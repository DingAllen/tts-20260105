"""
IndexTTS2 通用工具函数
包含Mask生成、序列处理等常用工具
"""

import torch
import numpy as np


def get_mask_from_lengths(lengths, max_len=None):
    """
    根据序列长度生成Padding Mask
    
    Args:
        lengths: (B,) 每个序列的实际长度
        max_len: 最大序列长度，如果为None则使用lengths中的最大值
        
    Returns:
        mask: (B, max_len) bool张量，True表示有效位置，False表示padding位置
    """
    if max_len is None:
        max_len = torch.max(lengths).item()
    
    batch_size = lengths.size(0)
    # 生成位置索引 (B, max_len)
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    # 比较生成mask
    mask = (ids < lengths.unsqueeze(1))
    
    return mask


def get_causal_mask(seq_len, device='cpu'):
    """
    生成因果注意力掩码（自回归mask）
    
    Args:
        seq_len: 序列长度
        device: 设备
        
    Returns:
        mask: (seq_len, seq_len) bool张量，上三角为False（不能看到未来）
    """
    # 生成下三角矩阵（包含对角线）
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    return mask


def get_attention_mask(padding_mask, causal_mask=None):
    """
    结合padding mask和causal mask生成完整的attention mask
    
    Args:
        padding_mask: (B, seq_len) padding mask
        causal_mask: (seq_len, seq_len) causal mask，可选
        
    Returns:
        mask: (B, seq_len, seq_len) 或 (B, 1, seq_len, seq_len) 注意力mask
    """
    batch_size, seq_len = padding_mask.size()
    
    # 扩展padding_mask到 (B, 1, seq_len)，然后广播到 (B, seq_len, seq_len)
    # key的mask：每个query位置都要检查哪些key是有效的
    key_padding_mask = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)  # (B, seq_len, seq_len)
    
    if causal_mask is not None:
        # 结合causal mask
        # causal_mask: (seq_len, seq_len)
        # 广播相乘（逻辑与）
        mask = key_padding_mask & causal_mask.unsqueeze(0)
    else:
        mask = key_padding_mask
    
    return mask


def sequence_mask(length, max_length=None):
    """
    创建序列掩码（numpy版本）
    
    Args:
        length: 序列长度数组 (B,)
        max_length: 最大长度
        
    Returns:
        mask: (B, max_length) bool数组
    """
    if max_length is None:
        max_length = length.max()
    x = np.arange(max_length, dtype=length.dtype)
    return x[None, :] < length[:, None]


def lengths_to_mask(lengths, max_len=None, dtype=torch.bool):
    """
    将长度张量转换为mask张量（更简洁的版本）
    
    Args:
        lengths: (B,) 长度张量
        max_len: 最大长度
        dtype: 返回的dtype
        
    Returns:
        mask: (B, max_len) mask张量
    """
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or lengths.max().item()
    mask = torch.arange(max_len, device=lengths.device, dtype=lengths.dtype).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = mask.to(dtype)
    return mask


def repeat_interleave(x, repeats, dim=1):
    """
    沿指定维度重复张量元素（用于Length Regulator）
    
    Args:
        x: 输入张量 (B, T, D)
        repeats: 每个元素的重复次数 (B, T)
        dim: 重复的维度
        
    Returns:
        repeated: 重复后的张量
    """
    if isinstance(repeats, int):
        return x.repeat_interleave(repeats, dim=dim)
    
    # 处理可变长度重复
    batch_size, seq_len = x.shape[0], x.shape[1]
    max_repeats = repeats.max().item()
    
    # 创建输出张量
    output_list = []
    for i in range(batch_size):
        batch_output = []
        for j in range(seq_len):
            repeat_times = repeats[i, j].item()
            if repeat_times > 0:
                batch_output.append(x[i:i+1, j:j+1].repeat(1, repeat_times, 1))
        if batch_output:
            output_list.append(torch.cat(batch_output, dim=1))
    
    if output_list:
        return torch.cat(output_list, dim=0)
    else:
        return torch.zeros(batch_size, 0, x.shape[-1], device=x.device)


def init_weights(module, mean=0.0, std=0.02):
    """
    初始化模型权重
    
    Args:
        module: nn.Module
        mean: 均值
        std: 标准差
    """
    if isinstance(module, torch.nn.Linear):
        module.weight.data.normal_(mean=mean, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=mean, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
