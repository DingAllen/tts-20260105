"""
S2M模块的ODE求解器
用于CFM模型的推理阶段，通过数值求解ODE生成梅尔频谱
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from ...common import config


class EulerSolver:
    """
    欧拉方法ODE求解器
    
    最简单的ODE数值求解方法：
        x_{t+dt} = x_t + v_t * dt
    
    用于CFM模型的推理，从噪声逐步生成Mel频谱
    """
    
    def __init__(self, num_steps=config.CFM_TIME_STEPS):
        """
        初始化求解器
        
        Args:
            num_steps: ODE求解步数
        """
        self.num_steps = num_steps
    
    @torch.no_grad()
    def solve(self, model, z, condition, verbose=False):
        """
        求解ODE生成样本
        
        Args:
            model: CFM模型（VectorFieldEstimator）
            z: 初始噪声 (B, Mel_Bins, Frame_Len)
            condition: 条件特征 (B, Frame_Len, Condition_Dim)
            verbose: 是否显示进度条
            
        Returns:
            x: 生成的Mel频谱 (B, Mel_Bins, Frame_Len)
        """
        device = z.device
        batch_size = z.size(0)
        
        # 初始化：t=0时，x=z（纯噪声）
        x = z
        
        # 时间步长
        dt = 1.0 / self.num_steps
        
        # 迭代求解
        iterator = range(self.num_steps)
        if verbose:
            iterator = tqdm(iterator, desc="ODE Solving")
        
        for step in iterator:
            # 当前时间 t
            t = torch.full((batch_size,), step * dt, device=device)
            
            # 预测速度向量 v_t
            v_t = model(x, t, condition)
            
            # 欧拉更新
            x = x + v_t * dt
        
        return x


class RK4Solver:
    """
    四阶龙格-库塔(Runge-Kutta)方法ODE求解器
    
    相比欧拉方法更精确，但计算量更大：
        k1 = v(x_t, t)
        k2 = v(x_t + dt/2 * k1, t + dt/2)
        k3 = v(x_t + dt/2 * k2, t + dt/2)
        k4 = v(x_t + dt * k3, t + dt)
        x_{t+dt} = x_t + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    
    def __init__(self, num_steps=config.CFM_TIME_STEPS):
        """
        初始化求解器
        
        Args:
            num_steps: ODE求解步数
        """
        self.num_steps = num_steps
    
    @torch.no_grad()
    def solve(self, model, z, condition, verbose=False):
        """
        求解ODE生成样本
        
        Args:
            model: CFM模型
            z: 初始噪声 (B, Mel_Bins, Frame_Len)
            condition: 条件特征 (B, Frame_Len, Condition_Dim)
            verbose: 是否显示进度条
            
        Returns:
            x: 生成的Mel频谱 (B, Mel_Bins, Frame_Len)
        """
        device = z.device
        batch_size = z.size(0)
        
        # 初始化
        x = z
        dt = 1.0 / self.num_steps
        
        # 迭代求解
        iterator = range(self.num_steps)
        if verbose:
            iterator = tqdm(iterator, desc="RK4 Solving")
        
        for step in iterator:
            t = torch.full((batch_size,), step * dt, device=device)
            
            # k1 = f(x_t, t)
            k1 = model(x, t, condition)
            
            # k2 = f(x_t + dt/2 * k1, t + dt/2)
            t_mid = torch.full((batch_size,), (step + 0.5) * dt, device=device)
            k2 = model(x + dt * k1 / 2, t_mid, condition)
            
            # k3 = f(x_t + dt/2 * k2, t + dt/2)
            k3 = model(x + dt * k2 / 2, t_mid, condition)
            
            # k4 = f(x_t + dt * k3, t + dt)
            t_next = torch.full((batch_size,), (step + 1) * dt, device=device)
            k4 = model(x + dt * k3, t_next, condition)
            
            # RK4更新
            x = x + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return x


class HeunSolver:
    """
    Heun方法ODE求解器（二阶方法）
    
    介于欧拉和RK4之间的折中方案：
        k1 = v(x_t, t)
        k2 = v(x_t + dt * k1, t + dt)
        x_{t+dt} = x_t + dt/2 * (k1 + k2)
    """
    
    def __init__(self, num_steps=config.CFM_TIME_STEPS):
        """
        初始化求解器
        
        Args:
            num_steps: ODE求解步数
        """
        self.num_steps = num_steps
    
    @torch.no_grad()
    def solve(self, model, z, condition, verbose=False):
        """
        求解ODE生成样本
        
        Args:
            model: CFM模型
            z: 初始噪声 (B, Mel_Bins, Frame_Len)
            condition: 条件特征 (B, Frame_Len, Condition_Dim)
            verbose: 是否显示进度条
            
        Returns:
            x: 生成的Mel频谱 (B, Mel_Bins, Frame_Len)
        """
        device = z.device
        batch_size = z.size(0)
        
        # 初始化
        x = z
        dt = 1.0 / self.num_steps
        
        # 迭代求解
        iterator = range(self.num_steps)
        if verbose:
            iterator = tqdm(iterator, desc="Heun Solving")
        
        for step in iterator:
            t = torch.full((batch_size,), step * dt, device=device)
            
            # k1 = v(x_t, t)
            k1 = model(x, t, condition)
            
            # k2 = v(x_t + dt * k1, t + dt)
            t_next = torch.full((batch_size,), (step + 1) * dt, device=device)
            k2 = model(x + dt * k1, t_next, condition)
            
            # Heun更新
            x = x + dt / 2 * (k1 + k2)
        
        return x


def euler_solve(model, z, condition, steps=50, verbose=False):
    """
    便捷函数：使用欧拉方法求解ODE
    
    Args:
        model: CFM模型
        z: 初始噪声
        condition: 条件特征
        steps: 求解步数
        verbose: 是否显示进度
        
    Returns:
        生成的Mel频谱
    """
    solver = EulerSolver(num_steps=steps)
    return solver.solve(model, z, condition, verbose=verbose)


def rk4_solve(model, z, condition, steps=50, verbose=False):
    """
    便捷函数：使用RK4方法求解ODE
    
    Args:
        model: CFM模型
        z: 初始噪声
        condition: 条件特征
        steps: 求解步数
        verbose: 是否显示进度
        
    Returns:
        生成的Mel频谱
    """
    solver = RK4Solver(num_steps=steps)
    return solver.solve(model, z, condition, verbose=verbose)


def heun_solve(model, z, condition, steps=50, verbose=False):
    """
    便捷函数：使用Heun方法求解ODE
    
    Args:
        model: CFM模型
        z: 初始噪声
        condition: 条件特征
        steps: 求解步数
        verbose: 是否显示进度
        
    Returns:
        生成的Mel频谱
    """
    solver = HeunSolver(num_steps=steps)
    return solver.solve(model, z, condition, verbose=verbose)


# 测试代码
if __name__ == "__main__":
    print("=== 测试ODE求解器 ===")
    
    from .diffusion import ConditionalFlowMatching
    
    # 创建CFM模型
    cfm = ConditionalFlowMatching()
    cfm.eval()
    
    # 准备输入
    batch_size = 2
    frame_len = 100
    mel_bins = config.MEL_BINS
    
    z = torch.randn(batch_size, mel_bins, frame_len)
    condition = torch.randn(batch_size, frame_len, config.HIDDEN_DIM)
    
    # 测试欧拉求解器
    print("\n--- 测试欧拉求解器 ---")
    euler_solver = EulerSolver(num_steps=10)
    with torch.no_grad():
        mel_euler = euler_solver.solve(cfm, z, condition, verbose=False)
    print(f"Generated Mel (Euler) shape: {mel_euler.shape}")
    assert mel_euler.shape == z.shape
    print("✅ 欧拉求解器测试通过")
    
    # 测试RK4求解器
    print("\n--- 测试RK4求解器 ---")
    rk4_solver = RK4Solver(num_steps=10)
    with torch.no_grad():
        mel_rk4 = rk4_solver.solve(cfm, z, condition, verbose=False)
    print(f"Generated Mel (RK4) shape: {mel_rk4.shape}")
    assert mel_rk4.shape == z.shape
    print("✅ RK4求解器测试通过")
    
    # 测试Heun求解器
    print("\n--- 测试Heun求解器 ---")
    heun_solver = HeunSolver(num_steps=10)
    with torch.no_grad():
        mel_heun = heun_solver.solve(cfm, z, condition, verbose=False)
    print(f"Generated Mel (Heun) shape: {mel_heun.shape}")
    assert mel_heun.shape == z.shape
    print("✅ Heun求解器测试通过")
    
    # 比较不同求解器的结果
    print("\n--- 比较求解器结果 ---")
    print(f"Euler vs RK4 差异: {(mel_euler - mel_rk4).abs().mean().item():.6f}")
    print(f"Euler vs Heun 差异: {(mel_euler - mel_heun).abs().mean().item():.6f}")
    print(f"RK4 vs Heun 差异: {(mel_rk4 - mel_heun).abs().mean().item():.6f}")
    
    print("\n✅ 所有ODE求解器测试通过！")
