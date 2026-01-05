"""
S2M (Semantic-to-Mel) 完整模型
包含GPT潜变量融合层和完整的训练/推理流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion import ConditionalFlowMatching
from .solver import euler_solve, rk4_solve
from ...common import config
from ...common.utils import repeat_interleave


class LengthRegulator(nn.Module):
    """
    长度调节器
    
    用于将T2S的语义Token序列上采样到Mel帧的长度。
    实现方式：简单重复或插值
    """
    
    def __init__(self, method='repeat'):
        """
        Args:
            method: 'repeat'（重复）或 'interpolate'（插值）
        """
        super().__init__()
        self.method = method
    
    def forward(self, x, target_len=None, durations=None):
        """
        长度调节
        
        Args:
            x: 输入序列 (B, Seq_Len, Dim)
            target_len: 目标长度（如果指定）
            durations: 每个位置的持续时间 (B, Seq_Len)（如果指定）
            
        Returns:
            regulated: 调节后的序列 (B, Target_Len, Dim)
        """
        if durations is not None:
            # 使用持续时间进行重复
            # 这个功能比较复杂，这里简化处理
            # 在实际实现中，需要根据durations重复每个token
            raise NotImplementedError("Duration-based regulation not implemented")
        
        if target_len is None:
            # 如果没有指定目标长度，直接返回
            return x
        
        batch_size, seq_len, dim = x.shape
        
        if self.method == 'repeat':
            # 简单重复：均匀重复到目标长度
            repeat_factor = target_len // seq_len
            remainder = target_len % seq_len
            
            # 均匀重复
            regulated = x.repeat_interleave(repeat_factor, dim=1)
            
            # 处理余数
            if remainder > 0:
                extra = x[:, :remainder, :]
                regulated = torch.cat([regulated, extra], dim=1)
        
        elif self.method == 'interpolate':
            # 使用插值
            # (B, Seq_Len, Dim) -> (B, Dim, Seq_Len)
            x_transposed = x.transpose(1, 2)
            # 插值到目标长度
            regulated = F.interpolate(
                x_transposed, 
                size=target_len, 
                mode='linear', 
                align_corners=False
            )
            # (B, Dim, Target_Len) -> (B, Target_Len, Dim)
            regulated = regulated.transpose(1, 2)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return regulated


class GPTFusionLayer(nn.Module):
    """
    GPT潜变量融合层
    
    将T2S的隐藏状态（GPT特征）融合到S2M的语义嵌入中，
    增强发音和韵律信息。
    
    核心思想：
        1. 对齐：T2S输出长度 != Mel帧数，需要上采样
        2. 融合：Q_sem（S2M自己的语义特征）+ H_GPT（来自T2S）
        3. Dropout：训练时随机将H_GPT置零，避免过度依赖
    """
    
    def __init__(
        self,
        semantic_dim=config.HIDDEN_DIM,
        gpt_dim=config.HIDDEN_DIM,
        output_dim=config.HIDDEN_DIM,
        dropout=0.1
    ):
        """
        初始化GPT融合层
        
        Args:
            semantic_dim: S2M语义特征维度
            gpt_dim: GPT隐藏状态维度
            output_dim: 输出维度
            dropout: Dropout比率（用于随机丢弃GPT特征）
        """
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.gpt_dim = gpt_dim
        self.output_dim = output_dim
        
        # 长度调节器（将GPT特征上采样到Mel长度）
        self.length_regulator = LengthRegulator(method='interpolate')
        
        # 融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(semantic_dim + gpt_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # GPT Dropout（训练时随机丢弃）
        self.gpt_dropout = nn.Dropout(dropout)
    
    def forward(self, semantic_features, gpt_hidden_states, target_len=None, training=True):
        """
        前向传播
        
        Args:
            semantic_features: S2M的语义特征 (B, Mel_Len, Semantic_Dim)
            gpt_hidden_states: T2S的隐藏状态 (B, Seq_Len, GPT_Dim)
            target_len: 目标Mel长度（如果为None则使用semantic_features的长度）
            training: 是否训练模式
            
        Returns:
            fused_features: 融合后的特征 (B, Mel_Len, Output_Dim)
        """
        batch_size = semantic_features.size(0)
        mel_len = semantic_features.size(1)
        
        if target_len is None:
            target_len = mel_len
        
        # 1. 调节GPT隐藏状态的长度
        # gpt_hidden_states: (B, Seq_Len, GPT_Dim) -> (B, Mel_Len, GPT_Dim)
        gpt_aligned = self.length_regulator(gpt_hidden_states, target_len=target_len)
        
        # 2. 训练时随机Dropout GPT特征
        if training:
            gpt_aligned = self.gpt_dropout(gpt_aligned)
        
        # 3. 确保semantic_features长度匹配
        if semantic_features.size(1) != target_len:
            semantic_features = self.length_regulator(semantic_features, target_len=target_len)
        
        # 4. 拼接并融合
        fused = torch.cat([semantic_features, gpt_aligned], dim=-1)  # (B, Mel_Len, Semantic_Dim + GPT_Dim)
        fused_features = self.fusion_mlp(fused)  # (B, Mel_Len, Output_Dim)
        
        return fused_features


class S2MModel(nn.Module):
    """
    完整的S2M模型
    
    组合所有组件：
        1. 语义Token嵌入
        2. GPT融合层
        3. 条件流匹配模型
        4. ODE求解器（推理时）
    """
    
    def __init__(
        self,
        semantic_vocab=config.SEMANTIC_VOCAB,
        semantic_embed_dim=config.HIDDEN_DIM,
        mel_bins=config.MEL_BINS,
        gpt_dim=config.HIDDEN_DIM,
        hidden_dim=config.S2M_HIDDEN_DIM,
        num_layers=config.S2M_NUM_LAYERS,
        num_heads=config.S2M_NUM_HEADS,
        dropout=config.S2M_DROPOUT
    ):
        """
        初始化S2M模型
        
        Args:
            semantic_vocab: 语义Token词汇表大小
            semantic_embed_dim: 语义嵌入维度
            mel_bins: 梅尔频谱bins数量
            gpt_dim: GPT隐藏状态维度
            hidden_dim: 模型隐藏层维度
            num_layers: 模型层数
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super().__init__()
        
        self.semantic_vocab = semantic_vocab
        self.mel_bins = mel_bins
        
        # 语义Token嵌入（S2M自己的）
        self.semantic_embedding = nn.Embedding(semantic_vocab, semantic_embed_dim)
        
        # GPT融合层
        self.gpt_fusion = GPTFusionLayer(
            semantic_dim=semantic_embed_dim,
            gpt_dim=gpt_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # 条件流匹配模型
        self.cfm = ConditionalFlowMatching(
            mel_bins=mel_bins,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            condition_dim=hidden_dim
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.semantic_embedding.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        semantic_ids,
        gpt_hidden_states,
        mel_gt=None,
        mel_lengths=None,
        compute_loss=True
    ):
        """
        前向传播（训练模式）
        
        Args:
            semantic_ids: 语义Token IDs (B, Semantic_Len)
            gpt_hidden_states: T2S的隐藏状态 (B, Seq_Len, GPT_Dim)
            mel_gt: Ground truth梅尔频谱 (B, Mel_Bins, Frame_Len)
            mel_lengths: Mel长度 (B,)
            compute_loss: 是否计算Loss
            
        Returns:
            outputs: 字典，包含loss等信息
        """
        batch_size = semantic_ids.size(0)
        
        # 1. 语义Token嵌入
        semantic_embed = self.semantic_embedding(semantic_ids)  # (B, Semantic_Len, Embed_Dim)
        
        # 2. 确定目标Mel长度
        if mel_gt is not None:
            target_len = mel_gt.size(2)
        elif mel_lengths is not None:
            target_len = mel_lengths.max().item()
        else:
            # 默认：语义序列长度 * 2（假设上采样因子为2）
            target_len = semantic_ids.size(1) * 2
        
        # 3. 调整语义嵌入到目标长度
        semantic_embed_regulated = self.gpt_fusion.length_regulator(
            semantic_embed, target_len=target_len
        )
        
        # 4. GPT融合
        condition = self.gpt_fusion(
            semantic_features=semantic_embed_regulated,
            gpt_hidden_states=gpt_hidden_states,
            target_len=target_len,
            training=self.training
        )  # (B, Target_Len, Hidden_Dim)
        
        # 5. 计算Loss（如果需要）
        outputs = {}
        if compute_loss and mel_gt is not None:
            loss = self.cfm.compute_loss(mel_gt, condition)
            outputs['loss'] = loss
        
        outputs['condition'] = condition
        
        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        semantic_ids,
        gpt_hidden_states,
        target_len=None,
        num_steps=50,
        solver='euler',
        verbose=False
    ):
        """
        生成梅尔频谱（推理模式）
        
        Args:
            semantic_ids: 语义Token IDs (B, Semantic_Len)
            gpt_hidden_states: T2S的隐藏状态 (B, Seq_Len, GPT_Dim)
            target_len: 目标Mel帧数
            num_steps: ODE求解步数
            solver: 求解器类型（'euler', 'rk4', 'heun'）
            verbose: 是否显示进度
            
        Returns:
            mel: 生成的梅尔频谱 (B, Mel_Bins, Target_Len)
        """
        self.eval()
        
        batch_size = semantic_ids.size(0)
        
        # 1. 确定目标长度
        if target_len is None:
            target_len = semantic_ids.size(1) * 2
        
        # 2. 语义嵌入和GPT融合
        semantic_embed = self.semantic_embedding(semantic_ids)
        semantic_embed_regulated = self.gpt_fusion.length_regulator(
            semantic_embed, target_len=target_len
        )
        
        condition = self.gpt_fusion(
            semantic_features=semantic_embed_regulated,
            gpt_hidden_states=gpt_hidden_states,
            target_len=target_len,
            training=False
        )
        
        # 3. 初始化噪声
        z = torch.randn(batch_size, self.mel_bins, target_len, device=semantic_ids.device)
        
        # 4. ODE求解生成Mel
        if solver == 'euler':
            mel = euler_solve(self.cfm, z, condition, steps=num_steps, verbose=verbose)
        elif solver == 'rk4':
            mel = rk4_solve(self.cfm, z, condition, steps=num_steps, verbose=verbose)
        else:
            raise ValueError(f"Unknown solver: {solver}")
        
        return mel


# 测试代码
if __name__ == "__main__":
    print("=== 测试S2M模型 ===")
    
    # 创建模型
    model = S2MModel()
    
    # 准备输入
    batch_size = 4
    semantic_len = 100
    gpt_seq_len = 124  # T2S的输出长度（4 + text_len + semantic_len）
    mel_len = 200
    
    semantic_ids = torch.randint(4, config.SEMANTIC_VOCAB, (batch_size, semantic_len))
    gpt_hidden = torch.randn(batch_size, gpt_seq_len, config.HIDDEN_DIM)
    mel_gt = torch.randn(batch_size, config.MEL_BINS, mel_len)
    
    # 测试训练前向传播
    print("\n--- 测试训练模式 ---")
    model.train()
    outputs = model(
        semantic_ids=semantic_ids,
        gpt_hidden_states=gpt_hidden,
        mel_gt=mel_gt,
        compute_loss=True
    )
    
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Condition shape: {outputs['condition'].shape}")
    
    # 测试反向传播
    outputs['loss'].backward()
    print("✅ 训练模式反向传播成功")
    
    # 测试推理生成
    print("\n--- 测试推理模式 ---")
    model.eval()
    mel_generated = model.generate(
        semantic_ids=semantic_ids,
        gpt_hidden_states=gpt_hidden,
        target_len=mel_len,
        num_steps=10,
        solver='euler',
        verbose=False
    )
    
    print(f"Generated Mel shape: {mel_generated.shape}")
    assert mel_generated.shape == (batch_size, config.MEL_BINS, mel_len)
    print("✅ 推理模式生成成功")
    
    print("\n✅ S2M模型测试通过！")
