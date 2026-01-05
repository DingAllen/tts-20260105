"""
阶段三：S2M模块测试
测试语义到梅尔频谱生成模块
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.common import config
from src.data.mock_factory import create_mock_batch
from src.modules.s2m.diffusion import (
    SinusoidalPositionEmbedding,
    VectorFieldEstimator,
    ConditionalFlowMatching
)
from src.modules.s2m.solver import EulerSolver, RK4Solver, HeunSolver
from src.modules.s2m.model import LengthRegulator, GPTFusionLayer, S2MModel


def test_sinusoidal_embedding():
    """测试正弦位置编码"""
    print("\n=== 测试正弦位置编码 ===")
    
    embedding = SinusoidalPositionEmbedding(dim=512)
    
    # 测试单个时间步
    t = torch.tensor([0.0, 0.5, 1.0])
    embed = embedding(t)
    
    print(f"Time steps: {t}")
    print(f"Embeddings shape: {embed.shape}")
    assert embed.shape == (3, 512)
    
    # 验证不同时间步的嵌入不同
    assert not torch.allclose(embed[0], embed[1])
    assert not torch.allclose(embed[1], embed[2])
    
    print("✅ 正弦位置编码测试通过")


def test_vector_field_estimator():
    """测试向量场估计器"""
    print("\n=== 测试向量场估计器 ===")
    
    estimator = VectorFieldEstimator()
    
    # 准备输入
    batch_size = 4
    mel_bins = config.MEL_BINS
    frame_len = 100
    
    x_t = torch.randn(batch_size, mel_bins, frame_len)
    t = torch.rand(batch_size)
    condition = torch.randn(batch_size, frame_len, config.HIDDEN_DIM)
    
    # 前向传播
    v_t = estimator(x_t, t, condition)
    
    print(f"Input x_t shape: {x_t.shape}")
    print(f"Output v_t shape: {v_t.shape}")
    
    assert v_t.shape == x_t.shape, f"Expected {x_t.shape}, got {v_t.shape}"
    print("✅ 向量场估计器测试通过")


def test_cfm_loss():
    """测试CFM Loss计算"""
    print("\n=== 测试CFM Loss计算 ===")
    
    cfm = ConditionalFlowMatching()
    
    # 准备输入
    batch_size = 4
    mel_bins = config.MEL_BINS
    frame_len = 100
    
    x_1 = torch.randn(batch_size, mel_bins, frame_len)
    condition = torch.randn(batch_size, frame_len, config.HIDDEN_DIM)
    
    # 计算Loss
    loss = cfm.compute_loss(x_1, condition)
    
    print(f"CFM Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    # 测试反向传播
    loss.backward()
    print("✅ CFM Loss计算和反向传播测试通过")


def test_interpolation():
    """测试插值计算"""
    print("\n=== 测试插值计算 ===")
    
    cfm = ConditionalFlowMatching()
    
    batch_size = 2
    mel_bins = config.MEL_BINS
    frame_len = 50
    
    x_0 = torch.zeros(batch_size, mel_bins, frame_len)
    x_1 = torch.ones(batch_size, mel_bins, frame_len)
    
    # 测试不同时间步的插值
    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for t_val in t_values:
        t = torch.full((batch_size,), t_val)
        x_t = cfm.get_interpolated_sample(x_0, x_1, t)
        
        # 验证插值正确性
        expected = (1 - t_val) * 0 + t_val * 1
        assert torch.allclose(x_t, torch.full_like(x_t, expected), atol=1e-6)
        print(f"t={t_val:.2f}: x_t mean={x_t.mean().item():.2f} (expected {expected:.2f})")
    
    print("✅ 插值计算测试通过")


def test_ode_solvers():
    """测试ODE求解器"""
    print("\n=== 测试ODE求解器 ===")
    
    cfm = ConditionalFlowMatching()
    cfm.eval()
    
    batch_size = 2
    mel_bins = config.MEL_BINS
    frame_len = 50
    
    z = torch.randn(batch_size, mel_bins, frame_len)
    condition = torch.randn(batch_size, frame_len, config.HIDDEN_DIM)
    
    # 测试Euler求解器
    print("\n--- 测试Euler求解器 ---")
    euler_solver = EulerSolver(num_steps=10)
    with torch.no_grad():
        mel_euler = euler_solver.solve(cfm, z, condition)
    print(f"Euler result shape: {mel_euler.shape}")
    assert mel_euler.shape == z.shape
    print("✅ Euler求解器测试通过")
    
    # 测试RK4求解器
    print("\n--- 测试RK4求解器 ---")
    rk4_solver = RK4Solver(num_steps=10)
    with torch.no_grad():
        mel_rk4 = rk4_solver.solve(cfm, z, condition)
    print(f"RK4 result shape: {mel_rk4.shape}")
    assert mel_rk4.shape == z.shape
    print("✅ RK4求解器测试通过")
    
    # 测试Heun求解器
    print("\n--- 测试Heun求解器 ---")
    heun_solver = HeunSolver(num_steps=10)
    with torch.no_grad():
        mel_heun = heun_solver.solve(cfm, z, condition)
    print(f"Heun result shape: {mel_heun.shape}")
    assert mel_heun.shape == z.shape
    print("✅ Heun求解器测试通过")


def test_length_regulator():
    """测试长度调节器"""
    print("\n=== 测试长度调节器 ===")
    
    # 测试插值方法
    regulator = LengthRegulator(method='interpolate')
    
    x = torch.randn(4, 50, 512)  # (B, Seq_Len, Dim)
    target_len = 100
    
    regulated = regulator(x, target_len=target_len)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {regulated.shape}")
    assert regulated.shape == (4, target_len, 512)
    print("✅ 长度调节器（插值）测试通过")
    
    # 测试重复方法
    regulator_repeat = LengthRegulator(method='repeat')
    regulated_repeat = regulator_repeat(x, target_len=target_len)
    print(f"Repeat output shape: {regulated_repeat.shape}")
    assert regulated_repeat.shape == (4, target_len, 512)
    print("✅ 长度调节器（重复）测试通过")


def test_gpt_fusion():
    """测试GPT融合层"""
    print("\n=== 测试GPT融合层 ===")
    
    fusion = GPTFusionLayer()
    
    batch_size = 4
    mel_len = 200
    gpt_len = 124  # T2S输出长度
    
    semantic_features = torch.randn(batch_size, mel_len, config.HIDDEN_DIM)
    gpt_hidden = torch.randn(batch_size, gpt_len, config.HIDDEN_DIM)
    
    # 训练模式
    fusion.train()
    fused = fusion(semantic_features, gpt_hidden, target_len=mel_len, training=True)
    
    print(f"Semantic features shape: {semantic_features.shape}")
    print(f"GPT hidden shape: {gpt_hidden.shape}")
    print(f"Fused features shape: {fused.shape}")
    
    assert fused.shape == (batch_size, mel_len, config.HIDDEN_DIM)
    print("✅ GPT融合层测试通过")


def test_s2m_model():
    """测试完整S2M模型"""
    print("\n=== 测试S2M模型 ===")
    
    model = S2MModel()
    
    # 准备输入
    batch_size = 4
    semantic_len = 100
    gpt_len = 124
    mel_len = 200
    
    semantic_ids = torch.randint(4, config.SEMANTIC_VOCAB, (batch_size, semantic_len))
    gpt_hidden = torch.randn(batch_size, gpt_len, config.HIDDEN_DIM)
    mel_gt = torch.randn(batch_size, config.MEL_BINS, mel_len)
    
    # 测试训练模式
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
    
    assert 'loss' in outputs
    assert outputs['loss'].item() > 0
    print("✅ 训练模式测试通过")
    
    # 测试反向传播
    outputs['loss'].backward()
    print("✅ 反向传播测试通过")


def test_s2m_generation():
    """测试S2M生成"""
    print("\n=== 测试S2M生成 ===")
    
    model = S2MModel()
    model.eval()
    
    batch_size = 2
    semantic_len = 100
    gpt_len = 124
    target_len = 200
    
    semantic_ids = torch.randint(4, config.SEMANTIC_VOCAB, (batch_size, semantic_len))
    gpt_hidden = torch.randn(batch_size, gpt_len, config.HIDDEN_DIM)
    
    # 生成Mel频谱
    with torch.no_grad():
        mel = model.generate(
            semantic_ids=semantic_ids,
            gpt_hidden_states=gpt_hidden,
            target_len=target_len,
            num_steps=10,
            solver='euler',
            verbose=False
        )
    
    print(f"Generated Mel shape: {mel.shape}")
    assert mel.shape == (batch_size, config.MEL_BINS, target_len)
    print("✅ S2M生成测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 50)
    print("开始运行阶段三S2M模块测试")
    print("=" * 50)
    
    try:
        test_sinusoidal_embedding()
        test_vector_field_estimator()
        test_cfm_loss()
        test_interpolation()
        test_ode_solvers()
        test_length_regulator()
        test_gpt_fusion()
        test_s2m_model()
        test_s2m_generation()
        
        print("\n" + "=" * 50)
        print("🎉 Phase 3 Complete: All tests passed")
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
