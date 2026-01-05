"""
阶段二：T2S模块测试
测试文本到语义Token生成模块
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.common import config
from src.data.mock_factory import create_mock_batch
from src.modules.layers.grl import GradientReversalLayer, SpeakerClassifier
from src.modules.t2s.embedding import SharedEmbedding, T2SInputBuilder
from src.modules.t2s.model import T2SModel
from src.modules.t2s.trainer import T2STrainer


def test_grl():
    """测试梯度反转层"""
    print("\n=== 测试GRL ===")
    
    # 创建GRL
    grl = GradientReversalLayer(lambda_=1.0)
    
    # 测试前向传播
    x = torch.randn(4, 10, 512, requires_grad=True)
    y = grl(x)
    
    assert torch.allclose(x, y), "GRL forward should be identity"
    print(f"✅ GRL前向传播是恒等映射")
    
    # 测试反向传播
    loss = y.sum()
    loss.backward()
    
    # 验证梯度是否反转
    assert x.grad is not None, "Gradient should exist"
    expected_grad = -torch.ones_like(x)
    assert torch.allclose(x.grad, expected_grad), "Gradient should be reversed"
    print(f"✅ GRL反向传播梯度反转正确")


def test_speaker_classifier():
    """测试说话人分类器"""
    print("\n=== 测试说话人分类器 ===")
    
    classifier = SpeakerClassifier(
        input_dim=512,
        num_speakers=10
    )
    
    x = torch.randn(4, 512)
    logits = classifier(x)
    
    assert logits.shape == (4, 10), f"Expected (4, 10), got {logits.shape}"
    print(f"Speaker classifier output shape: {logits.shape}")
    print(f"✅ 说话人分类器测试通过")


def test_shared_embedding():
    """测试共享嵌入层"""
    print("\n=== 测试共享嵌入层 ===")
    
    shared_emb = SharedEmbedding()
    
    # 测试Token嵌入
    token_ids = torch.randint(4, config.SEMANTIC_VOCAB, (4, 100))
    embeddings = shared_emb(token_ids)
    
    assert embeddings.shape == (4, 100, config.SEMANTIC_EMBED_DIM)
    print(f"Token embeddings shape: {embeddings.shape}")
    
    # 测试时长嵌入
    target_lengths = torch.tensor([50, 80, 120, 100])
    duration_embed = shared_emb.get_duration_embedding(target_lengths)
    
    assert duration_embed.shape == (4, config.SEMANTIC_EMBED_DIM)
    print(f"Duration embeddings shape: {duration_embed.shape}")
    
    # 验证共享权重
    # 时长嵌入应该等于position_embedding对应位置的权重
    for i, length in enumerate(target_lengths):
        expected = shared_emb.position_embedding.weight[length]
        assert torch.allclose(duration_embed[i], expected), "Duration embedding should use position embedding weights"
    
    print(f"✅ 共享嵌入层测试通过（权重共享正确）")


def test_input_builder():
    """测试输入构造器"""
    print("\n=== 测试T2S输入构造器 ===")
    
    input_builder = T2SInputBuilder()
    shared_emb = SharedEmbedding()
    
    # 准备输入
    batch_size = 4
    text_len = 20
    text_ids = torch.randint(4, config.TEXT_VOCAB, (batch_size, text_len))
    semantic_ids = torch.randint(4, config.SEMANTIC_VOCAB, (batch_size, 100))
    speaker_embed = torch.randn(batch_size, config.SPEAKER_EMBED_DIM)
    emotion_embed = torch.randn(batch_size, config.EMOTION_EMBED_DIM)
    target_lengths = torch.tensor([90, 100, 85, 95])
    duration_embed = shared_emb.get_duration_embedding(target_lengths)
    
    # 构造输入
    input_prefix = input_builder(
        text_ids, semantic_ids, speaker_embed, emotion_embed, duration_embed
    )
    
    # 验证形状：4 + Text_Len = 4 + 20 = 24
    expected_len = 4 + text_len
    assert input_prefix.shape == (batch_size, expected_len, config.HIDDEN_DIM)
    print(f"Input prefix shape: {input_prefix.shape}")
    print(f"✅ 输入构造器测试通过")


def test_t2s_model():
    """测试T2S模型"""
    print("\n=== 测试T2S模型 ===")
    
    # 创建模型
    model = T2SModel(use_grl=True)
    
    # 准备输入
    batch = create_mock_batch(batch_size=4, text_len=20, semantic_len=100)
    
    # 前向传播
    outputs = model(
        text_ids=batch.text_ids,
        semantic_ids=batch.semantic_ids,
        speaker_embed=batch.speaker_embed,
        emotion_embed=batch.emotion_embed,
        target_lengths=batch.target_len,
        compute_speaker_loss=True
    )
    
    # 验证输出
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    
    # 1. 验证Logits形状
    assert outputs['logits'].shape == (4, 100, config.SEMANTIC_VOCAB), \
        f"Expected (4, 100, {config.SEMANTIC_VOCAB}), got {outputs['logits'].shape}"
    print(f"✅ Logits形状正确: {outputs['logits'].shape}")
    
    # 2. 验证Hidden States形状
    assert outputs['hidden_states'].shape[0] == 4
    assert outputs['hidden_states'].shape[2] == config.HIDDEN_DIM
    print(f"✅ Hidden States形状正确: {outputs['hidden_states'].shape}")
    
    # 3. 验证Speaker Logits（如果有）
    if 'speaker_logits' in outputs:
        print(f"Speaker logits shape: {outputs['speaker_logits'].shape}")
        assert outputs['speaker_logits'].shape[0] == 4
        print(f"✅ Speaker Logits形状正确")


def test_t2s_backward():
    """测试T2S反向传播"""
    print("\n=== 测试T2S反向传播 ===")
    
    model = T2SModel(use_grl=False)
    batch = create_mock_batch(batch_size=4, text_len=20, semantic_len=100)
    
    # 前向传播
    outputs = model(
        text_ids=batch.text_ids,
        semantic_ids=batch.semantic_ids,
        speaker_embed=batch.speaker_embed,
        emotion_embed=batch.emotion_embed,
        target_lengths=batch.target_len
    )
    
    # 计算Loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)
    logits_flat = outputs['logits'].view(-1, config.SEMANTIC_VOCAB)
    target_flat = batch.semantic_ids.view(-1)
    loss = loss_fn(logits_flat, target_flat)
    
    print(f"Loss: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 验证梯度存在
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient for {name} should exist"
    
    print(f"✅ 反向传播成功，所有梯度计算正确")


def test_t2s_trainer():
    """测试T2S训练器"""
    print("\n=== 测试T2S训练器 ===")
    
    # 创建模型和训练器
    model = T2SModel(use_grl=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = T2STrainer(
        model=model,
        optimizer=optimizer,
        device='cpu',
        lambda_semantic=1.0
    )
    
    # 创建数据
    batch = create_mock_batch(batch_size=4)
    
    # Stage 1训练
    print("\n--- Stage 1训练 ---")
    trainer.set_stage(1)
    loss_dict = trainer.train_step(batch, stage=1)
    
    print(f"Total Loss: {loss_dict['total_loss']:.4f}")
    print(f"Semantic Loss: {loss_dict['loss_semantic']:.4f}")
    print(f"GRL Loss: {loss_dict['loss_grl']:.4f}")
    
    assert loss_dict['loss_grl'] == 0.0, "Stage 1 should not have GRL loss"
    assert loss_dict['loss_semantic'] > 0.0, "Semantic loss should be positive"
    print(f"✅ Stage 1训练测试通过")
    
    # Stage 2训练
    print("\n--- Stage 2训练 ---")
    trainer.set_stage(2)
    loss_dict = trainer.train_step(batch, stage=2)
    
    print(f"Total Loss: {loss_dict['total_loss']:.4f}")
    print(f"Semantic Loss: {loss_dict['loss_semantic']:.4f}")
    print(f"GRL Loss: {loss_dict['loss_grl']:.4f}")
    
    assert loss_dict['loss_grl'] > 0.0, "Stage 2 should have GRL loss"
    print(f"✅ Stage 2训练测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始运行阶段二T2S模块测试")
    print("=" * 50)
    
    try:
        test_grl()
        test_speaker_classifier()
        test_shared_embedding()
        test_input_builder()
        test_t2s_model()
        test_t2s_backward()
        test_t2s_trainer()
        
        print("\n" + "=" * 50)
        print("🎉 Phase 2 Complete: All tests passed")
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
