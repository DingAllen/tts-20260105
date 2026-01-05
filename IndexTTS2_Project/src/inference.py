"""
IndexTTS2 完整推理流水线
整合T2S和S2M模块，实现端到端的文本到语音生成
"""

import torch
import torch.nn as nn
from .modules.t2s.model import T2SModel
from .modules.s2m.model import S2MModel
from .common import config


class IndexTTS2Inference:
    """
    IndexTTS2推理引擎
    
    完整流程：
        文本 -> T2S -> 语义Token + 隐藏状态 -> S2M -> 梅尔频谱 -> Vocoder -> 波形
    
    当前实现：
        文本 -> T2S -> 语义Token + 隐藏状态 -> S2M -> 梅尔频谱
        (Vocoder部分为Mock实现)
    """
    
    def __init__(
        self,
        t2s_model=None,
        s2m_model=None,
        device='cpu',
        use_grl=False
    ):
        """
        初始化推理引擎
        
        Args:
            t2s_model: T2S模型（如果为None则创建新模型）
            s2m_model: S2M模型（如果为None则创建新模型）
            device: 设备
            use_grl: T2S是否使用GRL
        """
        self.device = device
        
        # 加载或创建T2S模型
        if t2s_model is None:
            self.t2s_model = T2SModel(use_grl=use_grl)
        else:
            self.t2s_model = t2s_model
        
        # 加载或创建S2M模型
        if s2m_model is None:
            self.s2m_model = S2MModel()
        else:
            self.s2m_model = s2m_model
        
        # 移动到设备
        self.t2s_model = self.t2s_model.to(device)
        self.s2m_model = self.s2m_model.to(device)
        
        # 设置为评估模式
        self.t2s_model.eval()
        self.s2m_model.eval()
    
    def load_models(self, t2s_path=None, s2m_path=None):
        """
        加载模型权重
        
        Args:
            t2s_path: T2S模型权重路径
            s2m_path: S2M模型权重路径
        """
        if t2s_path is not None:
            checkpoint = torch.load(t2s_path, map_location=self.device)
            self.t2s_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 已加载T2S模型: {t2s_path}")
        
        if s2m_path is not None:
            checkpoint = torch.load(s2m_path, map_location=self.device)
            self.s2m_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 已加载S2M模型: {s2m_path}")
    
    @torch.no_grad()
    def predict(
        self,
        text_ids,
        speaker_embed,
        emotion_embed,
        target_duration=None,
        t2s_temperature=1.0,
        t2s_top_k=None,
        s2m_steps=50,
        s2m_solver='euler',
        verbose=False
    ):
        """
        完整推理流程
        
        Args:
            text_ids: 文本Token IDs (B, Text_Len) 或 (Text_Len,)
            speaker_embed: 说话人嵌入 (B, Speaker_Dim) 或 (Speaker_Dim,)
            emotion_embed: 情感嵌入 (B, Emotion_Dim) 或 (Emotion_Dim,)
            target_duration: 目标时长（语义Token数量） (B,) 或 标量
            t2s_temperature: T2S采样温度
            t2s_top_k: T2S Top-K采样
            s2m_steps: S2M ODE求解步数
            s2m_solver: S2M求解器类型
            verbose: 是否显示进度
            
        Returns:
            outputs: 字典，包含：
                - semantic_tokens: 生成的语义Token (B, Semantic_Len)
                - mel_spectrogram: 生成的梅尔频谱 (B, Mel_Bins, Mel_Len)
                - waveform: Mock波形 (B, Audio_Len)（占位符）
        """
        # 处理输入维度（支持单样本输入）
        if text_ids.dim() == 1:
            text_ids = text_ids.unsqueeze(0)
        if speaker_embed.dim() == 1:
            speaker_embed = speaker_embed.unsqueeze(0)
        if emotion_embed.dim() == 1:
            emotion_embed = emotion_embed.unsqueeze(0)
        
        batch_size = text_ids.size(0)
        
        # 处理target_duration
        if target_duration is not None:
            if isinstance(target_duration, int):
                target_duration = torch.full((batch_size,), target_duration, device=self.device)
            elif isinstance(target_duration, torch.Tensor) and target_duration.dim() == 0:
                target_duration = target_duration.unsqueeze(0).expand(batch_size)
        
        # 移动到设备
        text_ids = text_ids.to(self.device)
        speaker_embed = speaker_embed.to(self.device)
        emotion_embed = emotion_embed.to(self.device)
        if target_duration is not None:
            target_duration = target_duration.to(self.device)
        
        if verbose:
            print("=" * 50)
            print("开始IndexTTS2推理")
            print("=" * 50)
            print(f"批次大小: {batch_size}")
            print(f"文本长度: {text_ids.size(1)}")
        
        # ===== 阶段1: T2S生成语义Token =====
        if verbose:
            print("\n[1/3] T2S: 文本 -> 语义Token")
        
        semantic_tokens = self.t2s_model.generate(
            text_ids=text_ids,
            speaker_embed=speaker_embed,
            emotion_embed=emotion_embed,
            target_length=target_duration,
            temperature=t2s_temperature,
            top_k=t2s_top_k
        )
        
        if verbose:
            print(f"      生成的语义Token数量: {semantic_tokens.size(1)}")
        
        # ===== 阶段2: 获取T2S隐藏状态 =====
        if verbose:
            print("\n[2/3] T2S: 提取隐藏状态（用于GPT融合）")
        
        # 使用生成的semantic_tokens获取隐藏状态
        t2s_outputs = self.t2s_model(
            text_ids=text_ids,
            semantic_ids=semantic_tokens,
            speaker_embed=speaker_embed,
            emotion_embed=emotion_embed,
            target_lengths=target_duration,
            compute_speaker_loss=False
        )
        
        gpt_hidden_states = t2s_outputs['hidden_states']
        
        if verbose:
            print(f"      隐藏状态形状: {gpt_hidden_states.shape}")
        
        # ===== 阶段3: S2M生成梅尔频谱 =====
        if verbose:
            print(f"\n[3/3] S2M: 语义Token -> 梅尔频谱 (ODE步数: {s2m_steps})")
        
        mel_spectrogram = self.s2m_model.generate(
            semantic_ids=semantic_tokens,
            gpt_hidden_states=gpt_hidden_states,
            target_len=None,  # 自动计算
            num_steps=s2m_steps,
            solver=s2m_solver,
            verbose=verbose
        )
        
        if verbose:
            print(f"      梅尔频谱形状: {mel_spectrogram.shape}")
        
        # ===== 阶段4: Vocoder生成波形（Mock） =====
        if verbose:
            print("\n[4/4] Vocoder: 梅尔频谱 -> 波形 (Mock)")
        
        # Mock Vocoder：简单生成随机波形作为占位符
        # 在实际应用中，这里应该调用真实的Vocoder模型（如HiFi-GAN）
        audio_len = mel_spectrogram.size(2) * 256  # 假设hop_size=256
        waveform = self.mock_vocoder(mel_spectrogram, audio_len)
        
        if verbose:
            print(f"      波形形状: {waveform.shape}")
            print("\n" + "=" * 50)
            print("推理完成！")
            print("=" * 50)
        
        # 返回所有结果
        outputs = {
            'semantic_tokens': semantic_tokens,
            'mel_spectrogram': mel_spectrogram,
            'waveform': waveform,
            'gpt_hidden_states': gpt_hidden_states
        }
        
        return outputs
    
    def mock_vocoder(self, mel, audio_len):
        """
        Mock Vocoder（占位符）
        
        在实际应用中，应该使用真实的Vocoder模型
        
        Args:
            mel: 梅尔频谱 (B, Mel_Bins, Mel_Len)
            audio_len: 目标音频长度
            
        Returns:
            waveform: 波形 (B, Audio_Len)
        """
        batch_size = mel.size(0)
        # 生成随机波形（范围[-1, 1]）
        waveform = torch.randn(batch_size, audio_len, device=mel.device) * 0.5
        return waveform
    
    def synthesize(
        self,
        text,
        speaker_id=0,
        emotion_id=0,
        target_duration=None,
        **kwargs
    ):
        """
        高级接口：从文本字符串生成语音
        
        注意：这个接口需要文本处理器（Tokenizer）
        目前为简化实现，假设text已经是token IDs
        
        Args:
            text: 文本（当前假设为token IDs的列表或张量）
            speaker_id: 说话人ID
            emotion_id: 情感ID
            target_duration: 目标时长
            **kwargs: 其他参数传递给predict
            
        Returns:
            outputs: predict的返回结果
        """
        # 文本预处理（简化版）
        if isinstance(text, list):
            text_ids = torch.tensor(text, dtype=torch.long)
        elif isinstance(text, torch.Tensor):
            text_ids = text
        else:
            raise ValueError("text should be a list or tensor")
        
        # 创建说话人和情感嵌入（简化版，应该从查找表获取）
        speaker_embed = torch.randn(config.SPEAKER_EMBED_DIM)
        emotion_embed = torch.randn(config.EMOTION_EMBED_DIM)
        
        # 调用predict
        return self.predict(
            text_ids=text_ids,
            speaker_embed=speaker_embed,
            emotion_embed=emotion_embed,
            target_duration=target_duration,
            **kwargs
        )


# 测试代码
if __name__ == "__main__":
    print("=== 测试IndexTTS2推理引擎 ===")
    
    # 创建推理引擎
    inference = IndexTTS2Inference(device='cpu')
    
    # 准备输入
    text_ids = torch.randint(4, config.TEXT_VOCAB, (20,))  # 单样本
    speaker_embed = torch.randn(config.SPEAKER_EMBED_DIM)
    emotion_embed = torch.randn(config.EMOTION_EMBED_DIM)
    
    # 推理
    print("\n--- 测试完整推理 ---")
    outputs = inference.predict(
        text_ids=text_ids,
        speaker_embed=speaker_embed,
        emotion_embed=emotion_embed,
        target_duration=100,
        s2m_steps=10,
        verbose=True
    )
    
    print("\n--- 输出结果 ---")
    print(f"语义Token: {outputs['semantic_tokens'].shape}")
    print(f"梅尔频谱: {outputs['mel_spectrogram'].shape}")
    print(f"波形: {outputs['waveform'].shape}")
    
    # 批量推理
    print("\n--- 测试批量推理 ---")
    batch_text_ids = torch.randint(4, config.TEXT_VOCAB, (4, 20))
    batch_speaker_embed = torch.randn(4, config.SPEAKER_EMBED_DIM)
    batch_emotion_embed = torch.randn(4, config.EMOTION_EMBED_DIM)
    
    outputs = inference.predict(
        text_ids=batch_text_ids,
        speaker_embed=batch_speaker_embed,
        emotion_embed=batch_emotion_embed,
        s2m_steps=5,
        verbose=False
    )
    
    print(f"批量语义Token: {outputs['semantic_tokens'].shape}")
    print(f"批量梅尔频谱: {outputs['mel_spectrogram'].shape}")
    print(f"批量波形: {outputs['waveform'].shape}")
    
    print("\n✅ IndexTTS2推理引擎测试通过！")
