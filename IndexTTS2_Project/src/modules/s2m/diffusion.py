"""
S2M模块的条件流匹配 (Conditional Flow Matching, CFM) 模型
基于流匹配的扩散生成模型，用于将语义Token转换为梅尔频谱
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ...common import config


class SinusoidalPositionEmbedding(nn.Module):
    """
    正弦位置编码（用于编码时间步t）
    """
    
    def __init__(self, dim):
        """
        Args:
            dim: 嵌入维度
        """
        super().__init__()
        self.dim = dim
    
    def forward(self, timesteps):
        """
        Args:
            timesteps: (B,) 时间步，范围[0, 1]
            
        Returns:
            embeddings: (B, dim) 时间步嵌入
        """
        device = timesteps.device
        half_dim = self.dim // 2
        
        # 使用对数空间的频率
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        
        # timesteps: (B,) -> (B, 1)
        # embeddings: (half_dim,) -> (1, half_dim)
        embeddings = timesteps.unsqueeze(1) * embeddings.unsqueeze(0)  # (B, half_dim)
        
        # sin和cos拼接
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        
        return embeddings


class VectorFieldEstimator(nn.Module):
    """
    向量场估计器（CFM的核心网络）
    
    目标：给定噪声Mel x_t、时间步t和条件c，预测速度向量v_t
    使用Transformer架构处理序列数据
    """
    
    def __init__(
        self,
        mel_bins=config.MEL_BINS,
        hidden_dim=config.S2M_HIDDEN_DIM,
        num_layers=config.S2M_NUM_LAYERS,
        num_heads=config.S2M_NUM_HEADS,
        dropout=config.S2M_DROPOUT,
        condition_dim=config.HIDDEN_DIM
    ):
        """
        初始化向量场估计器
        
        Args:
            mel_bins: 梅尔频谱的bins数量
            hidden_dim: 隐藏层维度
            num_layers: Transformer层数
            num_heads: 多头注意力头数
            dropout: Dropout比率
            condition_dim: 条件向量维度（来自语义特征）
        """
        super().__init__()
        
        self.mel_bins = mel_bins
        self.hidden_dim = hidden_dim
        
        # Mel频谱输入投影
        self.mel_proj = nn.Linear(mel_bins, hidden_dim)
        
        # 时间步嵌入
        self.time_embedding = SinusoidalPositionEmbedding(hidden_dim)
        
        # 条件投影（将语义特征投影到hidden_dim）
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影（hidden_dim -> mel_bins）
        self.output_proj = nn.Linear(hidden_dim, mel_bins)
        
        # Layer Norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x_t, t, condition):
        """
        前向传播
        
        Args:
            x_t: 噪声Mel频谱 (B, Mel_Bins, Frame_Len)
            t: 时间步 (B,)，范围[0, 1]
            condition: 条件特征 (B, Frame_Len, Condition_Dim)
            
        Returns:
            v_t: 速度向量 (B, Mel_Bins, Frame_Len)
        """
        batch_size, mel_bins, frame_len = x_t.shape
        
        # 1. Mel频谱转换为序列格式
        # (B, Mel_Bins, Frame_Len) -> (B, Frame_Len, Mel_Bins)
        x_t = x_t.transpose(1, 2)  # (B, Frame_Len, Mel_Bins)
        
        # 2. 投影到hidden_dim
        x_embed = self.mel_proj(x_t)  # (B, Frame_Len, Hidden_Dim)
        
        # 3. 时间步嵌入
        time_embed = self.time_embedding(t)  # (B, Hidden_Dim)
        # 广播到每个时间帧
        time_embed = time_embed.unsqueeze(1).expand(-1, frame_len, -1)  # (B, Frame_Len, Hidden_Dim)
        
        # 4. 条件投影
        cond_embed = self.condition_proj(condition)  # (B, Frame_Len, Hidden_Dim)
        
        # 5. 融合所有信息
        # 简单相加（也可以用更复杂的融合方式）
        features = x_embed + time_embed + cond_embed  # (B, Frame_Len, Hidden_Dim)
        features = self.layer_norm(features)
        
        # 6. 通过Transformer
        features = self.transformer(features)  # (B, Frame_Len, Hidden_Dim)
        
        # 7. 投影回Mel空间
        v_t = self.output_proj(features)  # (B, Frame_Len, Mel_Bins)
        
        # 8. 转换回原始格式
        v_t = v_t.transpose(1, 2)  # (B, Mel_Bins, Frame_Len)
        
        return v_t


class ConditionalFlowMatching(nn.Module):
    """
    条件流匹配模型
    
    实现CFM训练和推理：
        训练：回归向量场 v_t
        推理：通过ODE求解生成Mel频谱
    
    数学原理：
        x_t = (1-t) * x_0 + t * x_1
        v_t = x_1 - x_0
        Loss = E[||v_θ(x_t, t, c) - (x_1 - x_0)||^2]
    """
    
    def __init__(
        self,
        mel_bins=config.MEL_BINS,
        hidden_dim=config.S2M_HIDDEN_DIM,
        num_layers=config.S2M_NUM_LAYERS,
        num_heads=config.S2M_NUM_HEADS,
        dropout=config.S2M_DROPOUT,
        condition_dim=config.HIDDEN_DIM,
        sigma=config.CFM_SIGMA
    ):
        """
        初始化CFM模型
        
        Args:
            mel_bins: 梅尔频谱bins数量
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
            num_heads: 注意力头数
            dropout: Dropout比率
            condition_dim: 条件维度
            sigma: 噪声标准差
        """
        super().__init__()
        
        self.mel_bins = mel_bins
        self.sigma = sigma
        
        # 向量场估计器
        self.vector_field = VectorFieldEstimator(
            mel_bins=mel_bins,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            condition_dim=condition_dim
        )
    
    def get_interpolated_sample(self, x_0, x_1, t):
        """
        获取插值样本 x_t = (1-t) * x_0 + t * x_1
        
        Args:
            x_0: 起点（噪声） (B, Mel_Bins, Frame_Len)
            x_1: 终点（真实Mel） (B, Mel_Bins, Frame_Len)
            t: 时间步 (B,)
            
        Returns:
            x_t: 插值样本 (B, Mel_Bins, Frame_Len)
        """
        # 确保t的形状正确以进行广播
        # t: (B,) -> (B, 1, 1)
        t = t.view(-1, 1, 1)
        
        # 线性插值
        x_t = (1 - t) * x_0 + t * x_1
        
        return x_t
    
    def compute_loss(self, x_1, condition, x_0=None):
        """
        计算CFM训练Loss
        
        Args:
            x_1: 真实Mel频谱（目标） (B, Mel_Bins, Frame_Len)
            condition: 条件特征 (B, Frame_Len, Condition_Dim)
            x_0: 起始噪声（如果为None则随机生成） (B, Mel_Bins, Frame_Len)
            
        Returns:
            loss: CFM Loss
        """
        batch_size = x_1.size(0)
        device = x_1.device
        
        # 1. 如果没有提供x_0，随机生成噪声
        if x_0 is None:
            x_0 = torch.randn_like(x_1) * self.sigma
        
        # 2. 随机采样时间步 t ~ Uniform(0, 1)
        t = torch.rand(batch_size, device=device)
        
        # 3. 获取插值样本 x_t
        x_t = self.get_interpolated_sample(x_0, x_1, t)
        
        # 4. 计算目标速度向量（真实的v_t）
        target_v = x_1 - x_0  # (B, Mel_Bins, Frame_Len)
        
        # 5. 预测速度向量
        pred_v = self.vector_field(x_t, t, condition)
        
        # 6. 计算MSE Loss
        loss = F.mse_loss(pred_v, target_v)
        
        return loss
    
    def forward(self, x_t, t, condition):
        """
        前向传播（推理时使用）
        
        Args:
            x_t: 当前状态 (B, Mel_Bins, Frame_Len)
            t: 时间步 (B,)
            condition: 条件特征 (B, Frame_Len, Condition_Dim)
            
        Returns:
            v_t: 速度向量 (B, Mel_Bins, Frame_Len)
        """
        return self.vector_field(x_t, t, condition)


# 测试代码
if __name__ == "__main__":
    print("=== 测试CFM模型 ===")
    
    # 创建模型
    cfm = ConditionalFlowMatching()
    
    # 准备输入
    batch_size = 4
    frame_len = 200
    mel_bins = config.MEL_BINS
    
    x_1 = torch.randn(batch_size, mel_bins, frame_len)  # 真实Mel
    condition = torch.randn(batch_size, frame_len, config.HIDDEN_DIM)  # 条件
    
    # 测试训练Loss计算
    print("\n--- 测试训练Loss ---")
    loss = cfm.compute_loss(x_1, condition)
    print(f"CFM Loss: {loss.item():.4f}")
    
    # 测试反向传播
    loss.backward()
    print("✅ 反向传播成功")
    
    # 测试推理
    print("\n--- 测试推理 ---")
    cfm.eval()
    with torch.no_grad():
        x_0 = torch.randn(batch_size, mel_bins, frame_len)
        t = torch.rand(batch_size)
        x_t = cfm.get_interpolated_sample(x_0, x_1, t)
        v_t = cfm(x_t, t, condition)
        
        print(f"x_t shape: {x_t.shape}")
        print(f"v_t shape: {v_t.shape}")
        assert v_t.shape == x_t.shape
    
    print("\n✅ CFM模型测试通过！")
