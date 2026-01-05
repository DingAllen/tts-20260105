"""
T2S (Text-to-Semantic) 模型
基于自回归Transformer的文本到语义Token生成模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import SharedEmbedding, T2SInputBuilder
from ..layers.grl import GradientReversalLayer, SpeakerClassifier
from ...common import config
from ...common.utils import get_causal_mask


class T2SModel(nn.Module):
    """
    T2S自回归Transformer模型
    
    功能：
        1. 接收文本、说话人、情感、时长等输入
        2. 自回归生成语义Token序列
        3. 输出语义Token的logits和隐藏状态（用于S2M）
        4. 支持GRL进行情感解耦（Stage 2训练）
    
    架构：
        Input -> [Speaker+Emotion, Duration, <BT>, Text, <BA>, Semantic_Tokens] 
              -> Transformer Decoder 
              -> Output Logits + Hidden States
    """
    
    def __init__(
        self,
        vocab_size=config.SEMANTIC_VOCAB,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.T2S_NUM_LAYERS,
        num_heads=config.T2S_NUM_HEADS,
        ffn_dim=config.T2S_FFN_DIM,
        dropout=config.T2S_DROPOUT,
        max_position=2048,
        use_grl=False,
        grl_lambda=config.GRL_LAMBDA
    ):
        """
        初始化T2S模型
        
        Args:
            vocab_size: 语义Token词汇表大小
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 多头注意力头数
            ffn_dim: 前馈网络维度
            dropout: Dropout比率
            max_position: 最大位置数
            use_grl: 是否使用梯度反转层
            grl_lambda: GRL反转强度
        """
        super(T2SModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_grl = use_grl
        
        # 共享嵌入层（语义Token和时长控制）
        self.shared_embedding = SharedEmbedding(
            vocab_size=vocab_size,
            embed_dim=hidden_dim,
            max_position=max_position
        )
        
        # 输入构造器
        self.input_builder = T2SInputBuilder(embed_dim=hidden_dim)
        
        # Transformer Decoder层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        # 输出投影层（隐藏状态 -> 词汇表logits）
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # GRL和说话人分类器（用于Stage 2训练）
        if use_grl:
            self.grl = GradientReversalLayer(lambda_=grl_lambda)
            self.speaker_classifier = SpeakerClassifier(
                input_dim=hidden_dim,
                num_speakers=10  # 可配置
            )
        else:
            self.grl = None
            self.speaker_classifier = None
        
        # Layer Norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        if self.output_projection.bias is not None:
            nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        text_ids,
        semantic_ids,
        speaker_embed,
        emotion_embed,
        target_lengths=None,
        text_mask=None,
        semantic_mask=None,
        compute_speaker_loss=False
    ):
        """
        前向传播
        
        Args:
            text_ids: 文本Token IDs (B, Text_Len)
            semantic_ids: 语义Token IDs (B, Semantic_Len)
            speaker_embed: 说话人嵌入 (B, Speaker_Dim)
            emotion_embed: 情感嵌入 (B, Emotion_Dim)
            target_lengths: 目标长度 (B,)，用于时长控制
            text_mask: 文本mask (B, Text_Len)
            semantic_mask: 语义mask (B, Semantic_Len)
            compute_speaker_loss: 是否计算说话人分类loss
            
        Returns:
            outputs: 字典，包含：
                - logits: 语义Token预测logits (B, Semantic_Len, Vocab_Size)
                - hidden_states: 最后一层隐藏状态 (B, Total_Len, Hidden_Dim)
                - speaker_logits: 说话人分类logits（如果use_grl=True）
        """
        batch_size, semantic_len = semantic_ids.size()
        text_len = text_ids.size(1)
        
        # 1. 获取时长嵌入
        if target_lengths is None:
            # 如果没有指定目标长度，使用实际语义序列长度
            target_lengths = torch.full(
                (batch_size,), 
                semantic_len, 
                dtype=torch.long, 
                device=semantic_ids.device
            )
        
        duration_embed = self.shared_embedding.get_duration_embedding(target_lengths)
        
        # 2. 构造输入前缀（不包括语义Token）
        input_prefix = self.input_builder(
            text_ids=text_ids,
            semantic_ids=None,
            speaker_embed=speaker_embed,
            emotion_embed=emotion_embed,
            duration_embed=duration_embed
        )  # (B, 4 + Text_Len, Hidden_Dim)
        
        # 3. 获取语义Token的嵌入（用于自回归输入）
        # 自回归：输入是[BOS, sem_0, sem_1, ..., sem_{n-2}]
        # 目标是[sem_0, sem_1, sem_2, ..., sem_{n-1}]
        semantic_embed = self.shared_embedding(semantic_ids)  # (B, Semantic_Len, Hidden_Dim)
        
        # 4. 拼接完整输入序列
        # [input_prefix, semantic_embed]
        full_input = torch.cat([input_prefix, semantic_embed], dim=1)
        # (B, 4 + Text_Len + Semantic_Len, Hidden_Dim)
        
        total_len = full_input.size(1)
        
        # 5. 生成Causal Mask（自回归mask）
        causal_mask = get_causal_mask(total_len, device=semantic_ids.device)
        # Transformer需要的mask格式：(Seq_Len, Seq_Len)，True表示可以attend
        # 但PyTorch的Transformer用的是additive mask，False表示可以attend
        # 需要转换：True -> 0.0, False -> -inf
        attn_mask = torch.zeros(total_len, total_len, device=semantic_ids.device)
        attn_mask.masked_fill_(~causal_mask, float('-inf'))
        
        # 6. 通过Transformer Decoder
        # 注意：TransformerDecoder需要memory（encoder输出），但T2S是纯decoder
        # 所以我们使用自身作为memory（类似GPT）
        # 实际上，TransformerDecoder不太适合，应该用TransformerEncoder或直接用Decoder Layer
        # 这里为了简化，我们直接使用Decoder Layers手动构建
        
        hidden_states = full_input
        
        # 手动通过每一层
        for layer in self.transformer_decoder.layers:
            # 使用self-attention（将tgt和memory设为相同）
            hidden_states = layer(
                hidden_states,
                hidden_states,  # memory = tgt（自回归）
                tgt_mask=attn_mask,
                memory_mask=attn_mask
            )
        
        # Layer Norm
        hidden_states = self.layer_norm(hidden_states)
        # (B, Total_Len, Hidden_Dim)
        
        # 7. 提取语义Token部分的隐藏状态
        prefix_len = 4 + text_len
        semantic_hidden = hidden_states[:, prefix_len:, :]  # (B, Semantic_Len, Hidden_Dim)
        
        # 8. 投影到词汇表得到logits
        logits = self.output_projection(semantic_hidden)  # (B, Semantic_Len, Vocab_Size)
        
        # 9. 准备输出
        outputs = {
            'logits': logits,
            'hidden_states': hidden_states  # 完整的隐藏状态，用于S2M
        }
        
        # 10. 如果使用GRL，计算说话人分类
        if self.use_grl and compute_speaker_loss:
            # 通过GRL
            grl_features = self.grl(semantic_hidden.mean(dim=1))  # (B, Hidden_Dim)
            speaker_logits = self.speaker_classifier(grl_features)  # (B, Num_Speakers)
            outputs['speaker_logits'] = speaker_logits
        
        return outputs
    
    def generate(
        self,
        text_ids,
        speaker_embed,
        emotion_embed,
        target_length=None,
        max_length=500,
        temperature=1.0,
        top_k=None,
        top_p=None
    ):
        """
        自回归生成语义Token序列
        
        Args:
            text_ids: 文本Token IDs (B, Text_Len)
            speaker_embed: 说话人嵌入 (B, Speaker_Dim)
            emotion_embed: 情感嵌入 (B, Emotion_Dim)
            target_length: 目标长度 (B,)
            max_length: 最大生成长度
            temperature: 采样温度
            top_k: Top-K采样
            top_p: Nucleus采样
            
        Returns:
            generated_ids: 生成的语义Token IDs (B, Gen_Len)
        """
        batch_size = text_ids.size(0)
        device = text_ids.device
        
        # 初始化生成序列（从BOS开始）
        generated_ids = torch.full(
            (batch_size, 1),
            config.BOS_TOKEN_ID,
            dtype=torch.long,
            device=device
        )
        
        # 确定生成长度
        if target_length is None:
            gen_length = max_length
        else:
            gen_length = target_length.max().item()
        
        # 自回归生成
        for step in range(gen_length):
            # 前向传播
            outputs = self.forward(
                text_ids=text_ids,
                semantic_ids=generated_ids,
                speaker_embed=speaker_embed,
                emotion_embed=emotion_embed,
                target_lengths=target_length,
                compute_speaker_loss=False
            )
            
            # 获取最后一个位置的logits
            next_token_logits = outputs['logits'][:, -1, :]  # (B, Vocab_Size)
            
            # 应用temperature
            next_token_logits = next_token_logits / temperature
            
            # Top-K采样
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Top-P (Nucleus)采样
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # 采样
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # 拼接到生成序列
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # 检查是否所有序列都已经生成了EOS
            if (next_token == config.EOS_TOKEN_ID).all():
                break
        
        return generated_ids[:, 1:]  # 去掉BOS


# 测试代码
if __name__ == "__main__":
    print("=== 测试T2S模型 ===")
    
    # 创建模型
    model = T2SModel(use_grl=True)
    
    # 准备输入
    batch_size = 4
    text_ids = torch.randint(4, config.TEXT_VOCAB, (batch_size, 20))
    semantic_ids = torch.randint(4, config.SEMANTIC_VOCAB, (batch_size, 100))
    speaker_embed = torch.randn(batch_size, config.SPEAKER_EMBED_DIM)
    emotion_embed = torch.randn(batch_size, config.EMOTION_EMBED_DIM)
    target_lengths = torch.tensor([90, 100, 85, 95])
    
    # 前向传播
    outputs = model(
        text_ids=text_ids,
        semantic_ids=semantic_ids,
        speaker_embed=speaker_embed,
        emotion_embed=emotion_embed,
        target_lengths=target_lengths,
        compute_speaker_loss=True
    )
    
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    
    assert outputs['logits'].shape == (batch_size, 100, config.SEMANTIC_VOCAB)
    assert outputs['hidden_states'].shape[0] == batch_size
    assert outputs['hidden_states'].shape[2] == config.HIDDEN_DIM
    
    if 'speaker_logits' in outputs:
        print(f"Speaker logits shape: {outputs['speaker_logits'].shape}")
    
    # 测试Loss计算
    print("\n=== 测试Loss计算 ===")
    target = semantic_ids  # Ground truth
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)
    
    logits_flat = outputs['logits'].view(-1, config.SEMANTIC_VOCAB)
    target_flat = target.view(-1)
    loss = loss_fn(logits_flat, target_flat)
    
    print(f"CrossEntropy Loss: {loss.item():.4f}")
    
    # 测试反向传播
    loss.backward()
    print("✅ 反向传播成功")
    
    print("\n✅ T2S模型测试通过！")
