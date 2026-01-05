"""
梯度反转层 (Gradient Reversal Layer - GRL)
用于对抗训练，实现情感解耦功能
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    梯度反转函数
    前向传播：恒等映射（直接返回输入）
    反向传播：将梯度乘以 -lambda（反转梯度）
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        """
        前向传播
        
        Args:
            ctx: 上下文对象，用于保存反向传播所需的信息
            x: 输入张量
            lambda_: 梯度反转强度
            
        Returns:
            x: 直接返回输入（恒等映射）
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播
        
        Args:
            ctx: 上下文对象
            grad_output: 来自上游的梯度
            
        Returns:
            grad_input: 反转后的梯度
            None: lambda_参数不需要梯度
        """
        lambda_ = ctx.lambda_
        # 反转梯度
        grad_input = -lambda_ * grad_output
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    梯度反转层模块
    
    在T2S模型的Stage 2训练中使用，用于消除说话人信息，实现情感解耦。
    
    数学原理：
        L_total = L_AR - λ * L_SpeakerClassifier
    
    通过在反向传播时反转梯度，使得模型在优化主任务的同时，
    主动降低说话人分类器的准确率，从而消除说话人特征。
    """
    
    def __init__(self, lambda_=1.0):
        """
        初始化GRL层
        
        Args:
            lambda_: 梯度反转强度，默认为1.0
        """
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，任意形状
            
        Returns:
            x: 输出张量，形状与输入相同（恒等映射）
        """
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        """
        动态调整梯度反转强度
        
        Args:
            lambda_: 新的梯度反转强度
        """
        self.lambda_ = lambda_


class SpeakerClassifier(nn.Module):
    """
    说话人分类器
    
    用于在GRL之后，接受反转梯度的特征，
    尝试分类说话人ID。在对抗训练中，模型会学习到
    与说话人无关的表示。
    """
    
    def __init__(self, input_dim, num_speakers=10, hidden_dim=256):
        """
        初始化说话人分类器
        
        Args:
            input_dim: 输入特征维度
            num_speakers: 说话人数量
            hidden_dim: 隐藏层维度
        """
        super(SpeakerClassifier, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_speakers)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (B, *, input_dim)
            
        Returns:
            logits: 说话人分类logits (B, *, num_speakers)
        """
        return self.mlp(x)


class EmotionClassifier(nn.Module):
    """
    情感分类器（可选）
    
    与SpeakerClassifier类似，但用于分类情感
    """
    
    def __init__(self, input_dim, num_emotions=5, hidden_dim=256):
        """
        初始化情感分类器
        
        Args:
            input_dim: 输入特征维度
            num_emotions: 情感类别数量
            hidden_dim: 隐藏层维度
        """
        super(EmotionClassifier, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_emotions)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 (B, *, input_dim)
            
        Returns:
            logits: 情感分类logits (B, *, num_emotions)
        """
        return self.mlp(x)


# 测试代码
if __name__ == "__main__":
    print("=== 测试梯度反转层 ===")
    
    # 创建GRL
    grl = GradientReversalLayer(lambda_=1.0)
    
    # 测试前向传播
    x = torch.randn(4, 10, 512, requires_grad=True)
    y = grl(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Forward pass preserves input: {torch.allclose(x, y)}")
    
    # 测试反向传播
    loss = y.sum()
    loss.backward()
    
    print(f"Gradient exists: {x.grad is not None}")
    print(f"Gradient is reversed: {torch.allclose(x.grad, -torch.ones_like(x))}")
    
    # 测试说话人分类器
    print("\n=== 测试说话人分类器 ===")
    classifier = SpeakerClassifier(input_dim=512, num_speakers=10)
    
    # 通过GRL的特征
    x2 = torch.randn(4, 10, 512)
    features_through_grl = grl(x2)
    speaker_logits = classifier(features_through_grl)
    
    print(f"Speaker logits shape: {speaker_logits.shape}")
    assert speaker_logits.shape == (4, 10, 10), "Shape mismatch"
    
    print("\n✅ GRL和分类器测试通过！")
