"""
T2S训练器
实现T2S模型的训练逻辑，包含三个Loss组件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...common import config


class T2STrainer:
    """
    T2S训练器
    
    实现三个训练阶段的Loss计算：
        1. loss_text: 文本预测Loss（可选）
        2. loss_semantic: 语义Token预测Loss（主Loss）
        3. loss_grl: 情感解耦Loss（仅Stage 2开启）
    
    总Loss：
        L_total = λ_text * L_text + λ_semantic * L_semantic + λ_grl * L_grl
    """
    
    def __init__(
        self,
        model,
        optimizer=None,
        device='cpu',
        lambda_text=0.0,
        lambda_semantic=1.0,
        lambda_grl=0.0
    ):
        """
        初始化训练器
        
        Args:
            model: T2S模型
            optimizer: 优化器
            device: 设备
            lambda_text: 文本预测Loss权重
            lambda_semantic: 语义预测Loss权重
            lambda_grl: GRL Loss权重
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lambda_text = lambda_text
        self.lambda_semantic = lambda_semantic
        self.lambda_grl = lambda_grl
        
        # Loss函数
        self.semantic_loss_fn = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)
        self.text_loss_fn = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)
        self.speaker_loss_fn = nn.CrossEntropyLoss()
    
    def compute_loss(
        self,
        text_ids,
        semantic_ids,
        speaker_embed,
        emotion_embed,
        target_lengths=None,
        speaker_ids=None,
        stage=1
    ):
        """
        计算训练Loss
        
        Args:
            text_ids: 文本Token IDs (B, Text_Len)
            semantic_ids: 语义Token IDs (B, Semantic_Len)
            speaker_embed: 说话人嵌入 (B, Speaker_Dim)
            emotion_embed: 情感嵌入 (B, Emotion_Dim)
            target_lengths: 目标长度 (B,)
            speaker_ids: 说话人ID (B,)，用于GRL分类
            stage: 训练阶段（1或2）
            
        Returns:
            loss_dict: 包含各个Loss的字典
        """
        # 前向传播
        compute_speaker = (stage == 2 and self.lambda_grl > 0)
        outputs = self.model(
            text_ids=text_ids,
            semantic_ids=semantic_ids,
            speaker_embed=speaker_embed,
            emotion_embed=emotion_embed,
            target_lengths=target_lengths,
            compute_speaker_loss=compute_speaker
        )
        
        logits = outputs['logits']  # (B, Semantic_Len, Vocab_Size)
        
        # 1. 语义Token预测Loss（主Loss）
        # 自回归：预测下一个token
        # 输入：[BOS, sem_0, sem_1, ..., sem_{n-2}]
        # 目标：[sem_0, sem_1, sem_2, ..., sem_{n-1}]
        # logits是对输入的每个位置的预测，所以要错位对齐
        
        # 实际上，我们的模型已经处理好了：
        # logits对应的是semantic_ids的预测
        # 所以直接计算loss
        logits_flat = logits.reshape(-1, logits.size(-1))  # (B * Semantic_Len, Vocab_Size)
        target_flat = semantic_ids.reshape(-1)  # (B * Semantic_Len)
        
        loss_semantic = self.semantic_loss_fn(logits_flat, target_flat)
        
        # 2. 文本预测Loss（可选，通常不用）
        loss_text = torch.tensor(0.0, device=self.device)
        
        # 3. GRL Loss（Stage 2）
        loss_grl = torch.tensor(0.0, device=self.device)
        if compute_speaker and 'speaker_logits' in outputs:
            speaker_logits = outputs['speaker_logits']  # (B, Num_Speakers)
            if speaker_ids is not None:
                loss_grl = self.speaker_loss_fn(speaker_logits, speaker_ids)
        
        # 总Loss
        total_loss = (
            self.lambda_text * loss_text +
            self.lambda_semantic * loss_semantic +
            self.lambda_grl * loss_grl
        )
        
        loss_dict = {
            'total_loss': total_loss,
            'loss_semantic': loss_semantic,
            'loss_text': loss_text,
            'loss_grl': loss_grl
        }
        
        return loss_dict
    
    def train_step(self, batch, stage=1):
        """
        单步训练
        
        Args:
            batch: MockBatch对象
            stage: 训练阶段
            
        Returns:
            loss_dict: Loss字典
        """
        self.model.train()
        
        # 计算Loss
        loss_dict = self.compute_loss(
            text_ids=batch.text_ids,
            semantic_ids=batch.semantic_ids,
            speaker_embed=batch.speaker_embed,
            emotion_embed=batch.emotion_embed,
            target_lengths=batch.target_len,
            speaker_ids=batch.speaker_ids,
            stage=stage
        )
        
        # 反向传播
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.MAX_GRAD_NORM
            )
            
            self.optimizer.step()
        
        # 转换为标量（用于日志）
        loss_dict_scalar = {
            k: v.item() if torch.is_tensor(v) else v 
            for k, v in loss_dict.items()
        }
        
        return loss_dict_scalar
    
    def set_stage(self, stage):
        """
        设置训练阶段
        
        Args:
            stage: 1或2
                - Stage 1: 只训练主任务（语义预测）
                - Stage 2: 加入GRL，进行情感解耦
        """
        if stage == 1:
            self.lambda_grl = 0.0
        elif stage == 2:
            self.lambda_grl = 1.0
        else:
            raise ValueError(f"Invalid stage: {stage}")
    
    def save_checkpoint(self, path):
        """保存checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'lambda_text': self.lambda_text,
            'lambda_semantic': self.lambda_semantic,
            'lambda_grl': self.lambda_grl
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """加载checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lambda_text = checkpoint.get('lambda_text', 0.0)
        self.lambda_semantic = checkpoint.get('lambda_semantic', 1.0)
        self.lambda_grl = checkpoint.get('lambda_grl', 0.0)


# 测试代码
if __name__ == "__main__":
    print("=== 测试T2S训练器 ===")
    
    from ..t2s.model import T2SModel
    from ...data.mock_factory import create_mock_batch
    
    # 创建模型和训练器
    model = T2SModel(use_grl=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = T2STrainer(
        model=model,
        optimizer=optimizer,
        device='cpu',
        lambda_semantic=1.0,
        lambda_grl=0.0
    )
    
    # 创建Mock数据
    batch = create_mock_batch(batch_size=4, device='cpu')
    
    # Stage 1训练（不使用GRL）
    print("\n--- Stage 1训练 ---")
    trainer.set_stage(1)
    loss_dict = trainer.train_step(batch, stage=1)
    
    print(f"Total Loss: {loss_dict['total_loss']:.4f}")
    print(f"Semantic Loss: {loss_dict['loss_semantic']:.4f}")
    print(f"GRL Loss: {loss_dict['loss_grl']:.4f}")
    
    # Stage 2训练（使用GRL）
    print("\n--- Stage 2训练 ---")
    trainer.set_stage(2)
    loss_dict = trainer.train_step(batch, stage=2)
    
    print(f"Total Loss: {loss_dict['total_loss']:.4f}")
    print(f"Semantic Loss: {loss_dict['loss_semantic']:.4f}")
    print(f"GRL Loss: {loss_dict['loss_grl']:.4f}")
    
    print("\n✅ T2S训练器测试通过！")
