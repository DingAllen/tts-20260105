"""
IndexTTS2 伪数据生成器 (Mock Data Factory)
用于生成符合真实数据分布但内容随机的张量，方便测试和开发
"""

import torch
import numpy as np
from ..common import config


class MockBatch:
    """
    伪数据批次生成器
    生成所有必需的训练数据字段，用于测试模型的形状变换和Loss计算
    """
    
    def __init__(
        self,
        batch_size=8,
        text_len=20,
        semantic_len=100,
        mel_frame_len=200,
        device='cpu'
    ):
        """
        初始化MockBatch
        
        Args:
            batch_size: 批次大小
            text_len: 文本序列最大长度
            semantic_len: 语义Token序列最大长度
            mel_frame_len: 梅尔频谱帧数最大长度
            device: 设备
        """
        self.batch_size = batch_size
        self.text_len = text_len
        self.semantic_len = semantic_len
        self.mel_frame_len = mel_frame_len
        self.device = device
        
        # 生成所有字段
        self._generate_data()
    
    def _generate_data(self):
        """生成所有必需的数据字段"""
        B = self.batch_size
        
        # ===== 文本相关 =====
        # 生成随机文本ID，范围 [4, TEXT_VOCAB)，避免特殊token
        self.text_ids = torch.randint(
            4, config.TEXT_VOCAB, 
            (B, self.text_len), 
            device=self.device
        )
        
        # 生成实际文本长度（变长）
        self.text_lengths = torch.randint(
            self.text_len // 2, self.text_len + 1,
            (B,),
            device=self.device
        )
        
        # ===== 语义Token相关 =====
        # 生成随机语义Token ID，范围 [4, SEMANTIC_VOCAB)
        self.semantic_ids = torch.randint(
            4, config.SEMANTIC_VOCAB,
            (B, self.semantic_len),
            device=self.device
        )
        
        # 生成实际语义序列长度（变长）
        self.semantic_lengths = torch.randint(
            self.semantic_len // 2, self.semantic_len + 1,
            (B,),
            device=self.device
        )
        
        # 目标长度（用于时长控制）
        self.target_len = self.semantic_lengths.clone()
        
        # ===== 梅尔频谱相关 =====
        # 生成随机梅尔频谱 Ground Truth
        # 形状: (B, MEL_BINS, Frame_Len)
        self.mel_gt = torch.randn(
            B, config.MEL_BINS, self.mel_frame_len,
            device=self.device
        )
        
        # 生成实际帧长度（变长）
        self.mel_lengths = torch.randint(
            self.mel_frame_len // 2, self.mel_frame_len + 1,
            (B,),
            device=self.device
        )
        
        # ===== 说话人和情感嵌入 =====
        # 生成随机说话人嵌入向量
        self.speaker_embed = torch.randn(
            B, config.SPEAKER_EMBED_DIM,
            device=self.device
        )
        
        # 生成随机情感嵌入向量
        self.emotion_embed = torch.randn(
            B, config.EMOTION_EMBED_DIM,
            device=self.device
        )
        
        # ===== 可选：说话人ID和情感ID（用于分类） =====
        self.speaker_ids = torch.randint(0, 10, (B,), device=self.device)
        self.emotion_ids = torch.randint(0, 5, (B,), device=self.device)
        
        # ===== T2S的隐藏状态（用于S2M的GPT融合） =====
        # 这个会在T2S forward时生成，这里先预留
        self.t2s_hidden_states = torch.randn(
            B, self.semantic_len, config.HIDDEN_DIM,
            device=self.device
        )
    
    def to(self, device):
        """将所有张量移动到指定设备"""
        self.device = device
        self.text_ids = self.text_ids.to(device)
        self.text_lengths = self.text_lengths.to(device)
        self.semantic_ids = self.semantic_ids.to(device)
        self.semantic_lengths = self.semantic_lengths.to(device)
        self.target_len = self.target_len.to(device)
        self.mel_gt = self.mel_gt.to(device)
        self.mel_lengths = self.mel_lengths.to(device)
        self.speaker_embed = self.speaker_embed.to(device)
        self.emotion_embed = self.emotion_embed.to(device)
        self.speaker_ids = self.speaker_ids.to(device)
        self.emotion_ids = self.emotion_ids.to(device)
        self.t2s_hidden_states = self.t2s_hidden_states.to(device)
        return self
    
    def __repr__(self):
        """打印批次信息"""
        info = f"MockBatch(\n"
        info += f"  batch_size={self.batch_size},\n"
        info += f"  text_ids={self.text_ids.shape},\n"
        info += f"  semantic_ids={self.semantic_ids.shape},\n"
        info += f"  mel_gt={self.mel_gt.shape},\n"
        info += f"  speaker_embed={self.speaker_embed.shape},\n"
        info += f"  emotion_embed={self.emotion_embed.shape},\n"
        info += f"  target_len={self.target_len.shape},\n"
        info += f"  device={self.device}\n"
        info += ")"
        return info


def create_mock_batch(batch_size=8, device='cpu', **kwargs):
    """
    便捷函数：创建Mock Batch
    
    Args:
        batch_size: 批次大小
        device: 设备
        **kwargs: 其他参数传递给MockBatch
        
    Returns:
        batch: MockBatch对象
    """
    return MockBatch(batch_size=batch_size, device=device, **kwargs)


# 用于快速测试的函数
if __name__ == "__main__":
    print("=== 测试MockBatch ===")
    batch = create_mock_batch(batch_size=4)
    print(batch)
    
    print("\n=== 测试字段访问 ===")
    print(f"Text IDs shape: {batch.text_ids.shape}")
    print(f"Semantic IDs shape: {batch.semantic_ids.shape}")
    print(f"Mel GT shape: {batch.mel_gt.shape}")
    print(f"Target lengths: {batch.target_len}")
    
    print("\n✅ MockBatch测试通过！")
