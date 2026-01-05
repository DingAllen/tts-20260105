"""
IndexTTS2 主训练脚本
Mock训练循环，演示如何训练T2S和S2M模型
"""

import torch
import torch.nn as nn
from src.modules.t2s.model import T2SModel
from src.modules.t2s.trainer import T2STrainer
from src.modules.s2m.model import S2MModel
from src.data.mock_factory import create_mock_batch
from src.common import config


class IndexTTS2Trainer:
    """
    IndexTTS2完整训练器
    
    训练策略：
        1. Stage 1: 训练T2S主任务（语义预测）
        2. Stage 2: 训练T2S + GRL（情感解耦）
        3. Stage 3: 训练S2M（梅尔生成）
        4. 联合微调（可选）
    """
    
    def __init__(
        self,
        device='cpu',
        learning_rate=config.LEARNING_RATE,
        use_grl=True
    ):
        """
        初始化训练器
        
        Args:
            device: 训练设备
            learning_rate: 学习率
            use_grl: 是否使用GRL
        """
        self.device = device
        self.learning_rate = learning_rate
        
        # 创建模型
        self.t2s_model = T2SModel(use_grl=use_grl).to(device)
        self.s2m_model = S2MModel().to(device)
        
        # 创建优化器
        self.t2s_optimizer = torch.optim.Adam(
            self.t2s_model.parameters(),
            lr=learning_rate
        )
        self.s2m_optimizer = torch.optim.Adam(
            self.s2m_model.parameters(),
            lr=learning_rate
        )
        
        # 创建T2S训练器
        self.t2s_trainer = T2STrainer(
            model=self.t2s_model,
            optimizer=self.t2s_optimizer,
            device=device
        )
        
        # 训练统计
        self.stats = {
            't2s_losses': [],
            's2m_losses': []
        }
    
    def train_t2s_stage1(self, num_steps=10):
        """
        训练T2S Stage 1（主任务）
        
        Args:
            num_steps: 训练步数
        """
        print("\n" + "=" * 50)
        print("T2S Stage 1训练：语义预测")
        print("=" * 50)
        
        self.t2s_trainer.set_stage(1)
        
        for step in range(num_steps):
            # 生成Mock数据
            batch = create_mock_batch(
                batch_size=config.BATCH_SIZE,
                device=self.device
            )
            
            # 训练步
            loss_dict = self.t2s_trainer.train_step(batch, stage=1)
            
            # 记录
            self.stats['t2s_losses'].append(loss_dict['total_loss'])
            
            if (step + 1) % 5 == 0:
                print(f"Step {step+1}/{num_steps} | "
                      f"Loss: {loss_dict['total_loss']:.4f} | "
                      f"Semantic: {loss_dict['loss_semantic']:.4f}")
        
        print(f"\n✅ T2S Stage 1训练完成")
    
    def train_t2s_stage2(self, num_steps=10):
        """
        训练T2S Stage 2（情感解耦）
        
        Args:
            num_steps: 训练步数
        """
        print("\n" + "=" * 50)
        print("T2S Stage 2训练：情感解耦")
        print("=" * 50)
        
        self.t2s_trainer.set_stage(2)
        
        for step in range(num_steps):
            # 生成Mock数据
            batch = create_mock_batch(
                batch_size=config.BATCH_SIZE,
                device=self.device
            )
            
            # 训练步
            loss_dict = self.t2s_trainer.train_step(batch, stage=2)
            
            if (step + 1) % 5 == 0:
                print(f"Step {step+1}/{num_steps} | "
                      f"Loss: {loss_dict['total_loss']:.4f} | "
                      f"Semantic: {loss_dict['loss_semantic']:.4f} | "
                      f"GRL: {loss_dict['loss_grl']:.4f}")
        
        print(f"\n✅ T2S Stage 2训练完成")
    
    def train_s2m(self, num_steps=10):
        """
        训练S2M（梅尔生成）
        
        Args:
            num_steps: 训练步数
        """
        print("\n" + "=" * 50)
        print("S2M训练：梅尔频谱生成")
        print("=" * 50)
        
        self.s2m_model.train()
        
        for step in range(num_steps):
            # 生成Mock数据
            batch = create_mock_batch(
                batch_size=config.BATCH_SIZE,
                device=self.device
            )
            
            # 前向传播
            outputs = self.s2m_model(
                semantic_ids=batch.semantic_ids,
                gpt_hidden_states=batch.t2s_hidden_states,
                mel_gt=batch.mel_gt,
                compute_loss=True
            )
            
            loss = outputs['loss']
            
            # 反向传播
            self.s2m_optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.s2m_model.parameters(),
                config.MAX_GRAD_NORM
            )
            
            self.s2m_optimizer.step()
            
            # 记录
            self.stats['s2m_losses'].append(loss.item())
            
            if (step + 1) % 5 == 0:
                print(f"Step {step+1}/{num_steps} | Loss: {loss.item():.4f}")
        
        print(f"\n✅ S2M训练完成")
    
    def save_checkpoints(self, t2s_path='t2s_checkpoint.pth', s2m_path='s2m_checkpoint.pth'):
        """
        保存模型checkpoint
        
        Args:
            t2s_path: T2S模型保存路径
            s2m_path: S2M模型保存路径
        """
        # 保存T2S
        t2s_checkpoint = {
            'model_state_dict': self.t2s_model.state_dict(),
            'optimizer_state_dict': self.t2s_optimizer.state_dict()
        }
        torch.save(t2s_checkpoint, t2s_path)
        print(f"✅ T2S模型已保存到: {t2s_path}")
        
        # 保存S2M
        s2m_checkpoint = {
            'model_state_dict': self.s2m_model.state_dict(),
            'optimizer_state_dict': self.s2m_optimizer.state_dict()
        }
        torch.save(s2m_checkpoint, s2m_path)
        print(f"✅ S2M模型已保存到: {s2m_path}")


def main():
    """主训练流程"""
    print("=" * 50)
    print("IndexTTS2 Mock训练循环")
    print("=" * 50)
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    # 创建训练器
    trainer = IndexTTS2Trainer(device=device)
    
    # T2S Stage 1训练
    trainer.train_t2s_stage1(num_steps=10)
    
    # T2S Stage 2训练
    trainer.train_t2s_stage2(num_steps=10)
    
    # S2M训练
    trainer.train_s2m(num_steps=10)
    
    # 保存模型（可选，注释掉以避免实际保存）
    # trainer.save_checkpoints()
    
    # 打印训练统计
    print("\n" + "=" * 50)
    print("训练统计")
    print("=" * 50)
    
    if trainer.stats['t2s_losses']:
        avg_t2s_loss = sum(trainer.stats['t2s_losses']) / len(trainer.stats['t2s_losses'])
        print(f"T2S平均Loss: {avg_t2s_loss:.4f}")
    
    if trainer.stats['s2m_losses']:
        avg_s2m_loss = sum(trainer.stats['s2m_losses']) / len(trainer.stats['s2m_losses'])
        print(f"S2M平均Loss: {avg_s2m_loss:.4f}")
    
    print("\n" + "=" * 50)
    print("🎉 Mock训练循环完成！")
    print("=" * 50)
    print("\n注意：这是一个演示性的Mock训练循环。")
    print("在真实应用中，需要：")
    print("  1. 真实的数据加载器")
    print("  2. 更多的训练步数和epoch")
    print("  3. 验证集和测试集")
    print("  4. TensorBoard日志")
    print("  5. 学习率调度器")
    print("  6. 早停和模型保存策略")


if __name__ == "__main__":
    main()
