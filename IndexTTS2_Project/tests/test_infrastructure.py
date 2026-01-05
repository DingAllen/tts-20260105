"""
阶段一：基础设施测试
测试伪数据生成器、配置和工具函数
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from src.common import config
from src.common.utils import (
    get_mask_from_lengths,
    get_causal_mask,
    get_attention_mask,
    lengths_to_mask
)
from src.data.mock_factory import create_mock_batch, MockBatch


def test_config():
    """测试配置加载"""
    print("\n=== 测试配置加载 ===")
    assert config.TEXT_VOCAB == 5000
    assert config.SEMANTIC_VOCAB == 1024
    assert config.MEL_BINS == 80
    assert config.HIDDEN_DIM == 512
    print("✅ 配置加载测试通过")


def test_mask_generation():
    """测试Mask生成函数"""
    print("\n=== 测试Mask生成 ===")
    
    # 测试get_mask_from_lengths
    lengths = torch.tensor([3, 5, 2, 4])
    mask = get_mask_from_lengths(lengths, max_len=6)
    
    print(f"Lengths: {lengths}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask:\n{mask}")
    
    # 验证形状
    assert mask.shape == (4, 6), f"Expected shape (4, 6), got {mask.shape}"
    
    # 验证第一个序列（长度3）
    assert mask[0, 0] == True and mask[0, 2] == True and mask[0, 3] == False
    
    print("✅ get_mask_from_lengths 测试通过")
    
    # 测试causal mask
    causal = get_causal_mask(5)
    print(f"\nCausal mask (5x5):\n{causal}")
    assert causal.shape == (5, 5)
    assert causal[0, 1] == False  # 不能看到未来
    assert causal[1, 0] == True   # 可以看到过去
    assert causal[2, 2] == True   # 可以看到当前
    
    print("✅ get_causal_mask 测试通过")
    
    # 测试组合mask
    padding_mask = torch.ones(2, 5, dtype=torch.bool)
    padding_mask[0, 4] = False  # 第一个序列的最后一个位置是padding
    padding_mask[1, 3:] = False  # 第二个序列的后两个位置是padding
    
    attention_mask = get_attention_mask(padding_mask, causal)
    print(f"\nAttention mask shape: {attention_mask.shape}")
    assert attention_mask.shape == (2, 5, 5)
    
    print("✅ get_attention_mask 测试通过")


def test_mock_batch_creation():
    """测试伪数据生成"""
    print("\n=== 测试MockBatch创建 ===")
    
    batch = create_mock_batch(
        batch_size=4,
        text_len=20,
        semantic_len=100,
        mel_frame_len=200,
        device='cpu'
    )
    
    print(batch)
    
    # 验证所有必需字段存在
    assert hasattr(batch, 'text_ids')
    assert hasattr(batch, 'semantic_ids')
    assert hasattr(batch, 'mel_gt')
    assert hasattr(batch, 'speaker_embed')
    assert hasattr(batch, 'emotion_embed')
    assert hasattr(batch, 'target_len')
    
    # 验证形状
    assert batch.text_ids.shape == (4, 20), f"Expected (4, 20), got {batch.text_ids.shape}"
    assert batch.semantic_ids.shape == (4, 100), f"Expected (4, 100), got {batch.semantic_ids.shape}"
    assert batch.mel_gt.shape == (4, config.MEL_BINS, 200), f"Expected (4, {config.MEL_BINS}, 200), got {batch.mel_gt.shape}"
    assert batch.speaker_embed.shape == (4, config.SPEAKER_EMBED_DIM)
    assert batch.emotion_embed.shape == (4, config.EMOTION_EMBED_DIM)
    assert batch.target_len.shape == (4,)
    
    # 验证值的范围
    assert batch.text_ids.min() >= 4  # 应该避开特殊token
    assert batch.text_ids.max() < config.TEXT_VOCAB
    assert batch.semantic_ids.min() >= 4
    assert batch.semantic_ids.max() < config.SEMANTIC_VOCAB
    
    print("✅ MockBatch创建测试通过")


def test_mock_batch_device_transfer():
    """测试设备转移"""
    print("\n=== 测试设备转移 ===")
    
    batch = create_mock_batch(batch_size=2, device='cpu')
    assert batch.text_ids.device.type == 'cpu'
    
    # 如果CUDA可用，测试转移到CUDA
    if torch.cuda.is_available():
        batch = batch.to('cuda')
        assert batch.text_ids.device.type == 'cuda'
        print("✅ CUDA设备转移测试通过")
    else:
        print("⚠️  CUDA不可用，跳过CUDA测试")


def test_mask_with_mock_batch():
    """测试Mask与MockBatch的集成"""
    print("\n=== 测试Mask与MockBatch集成 ===")
    
    batch = create_mock_batch(batch_size=4, text_len=20, device='cpu')
    
    # 使用实际长度生成mask
    text_mask = get_mask_from_lengths(batch.text_lengths, max_len=batch.text_len)
    semantic_mask = get_mask_from_lengths(batch.semantic_lengths, max_len=batch.semantic_len)
    
    print(f"Text lengths: {batch.text_lengths}")
    print(f"Text mask shape: {text_mask.shape}")
    print(f"Semantic lengths: {batch.semantic_lengths}")
    print(f"Semantic mask shape: {semantic_mask.shape}")
    
    # 验证mask的正确性
    for i in range(4):
        text_len = batch.text_lengths[i].item()
        # 检查有效位置
        assert text_mask[i, :text_len].all(), f"Valid positions should be True"
        # 检查padding位置（如果有的话）
        if text_len < batch.text_len:
            assert not text_mask[i, text_len:].any(), f"Padding positions should be False"
    
    print("✅ Mask与MockBatch集成测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始运行阶段一基础设施测试")
    print("=" * 50)
    
    try:
        test_config()
        test_mask_generation()
        test_mock_batch_creation()
        test_mock_batch_device_transfer()
        test_mask_with_mock_batch()
        
        print("\n" + "=" * 50)
        print("🎉 Phase 1 Complete: All tests passed")
        print("=" * 50)
        return True
        
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"❌ 测试失败: {str(e)}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
