#!/usr/bin/env python3
"""
梯度流演示
==========

本代码演示 Wasserstein 空间中的梯度流概念。
包括 JKO 方案和 Fokker-Planck 方程。

参考: "Optimal Transport for Machine Learners" by Gabriel Peyré (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter1d


def jko_step(rho, F_func, grad_F_func, tau, epsilon=0.1, n_iter=100):
    """
    执行一步 JKO (Jordan-Kinderlehrer-Otto) 方案

    rho_{k+1} = argmin_rho F(rho) + (1/(2*tau)) * W_2(rho, rho_k)^2

    这里使用简化的实现，用梯度下降近似

    Parameters
    ----------
    rho : array (n,)
        当前分布
    F_func : callable
        能量泛函 F(rho)
    grad_F_func : callable
        F 关于 rho 的变分导数
    tau : float
        时间步长
    epsilon : float
        熵正则化（用于计算 Wasserstein 距离）
    n_iter : int
        内部优化迭代次数

    Returns
    -------
    rho_next : array (n,)
        下一步的分布
    """
    rho_current = rho.copy()
    lr = 0.01  # 学习率

    for _ in range(n_iter):
        # 计算梯度: grad F + (1/tau) * grad_W
        grad_F = grad_F_func(rho_current)

        # Wasserstein 梯度近似 (简化)
        # 实际应该用 continuity equation，这里用简单的扩散近似
        grad_W = (rho_current - rho) / tau

        # 总梯度
        grad = grad_F + grad_W

        # 梯度下降
        rho_current = rho_current - lr * grad

        # 投影到概率单纯形
        rho_current = np.maximum(rho_current, 0)
        rho_current /= rho_current.sum()

    return rho_current


def heat_equation_flow(rho0, n_steps=100, tau=0.01, sigma=1.0):
    """
    热方程作为熵的梯度流

    F(rho) = integral rho log rho  (负熵)
    梯度流: partial_t rho = Laplacian rho

    Parameters
    ----------
    rho0 : array (n,)
        初始分布
    n_steps : int
        时间步数
    tau : float
        时间步长
    sigma : float
        扩散系数

    Returns
    -------
    trajectory : list
        分布随时间的演化
    """
    rho = rho0.copy()
    trajectory = [rho.copy()]

    for _ in range(n_steps):
        # 热方程: rho_{t+1} = rho_t + tau * Laplacian(rho_t)
        # 使用高斯滤波模拟扩散
        rho = gaussian_filter1d(rho, sigma=sigma * np.sqrt(tau))
        rho /= rho.sum()  # 归一化
        trajectory.append(rho.copy())

    return trajectory


def fokker_planck_flow(rho0, potential, n_steps=100, tau=0.01, beta=1.0):
    """
    Fokker-Planck 方程作为 KL 散度的梯度流

    F(rho) = KL(rho || pi) where pi ∝ exp(-V)
    梯度流: partial_t rho = div(rho grad V) + (1/beta) Laplacian rho

    Parameters
    ----------
    rho0 : array (n,)
        初始分布
    potential : array (n,)
        势能函数 V(x)
    n_steps : int
        时间步数
    tau : float
        时间步长
    beta : float
        逆温度参数

    Returns
    -------
    trajectory : list
        分布随时间的演化
    """
    rho = rho0.copy()
    trajectory = [rho.copy()]
    n = len(rho)
    dx = 1.0 / n

    # 计算势能的梯度
    grad_V = np.gradient(potential, dx)

    for _ in range(n_steps):
        # 计算 rho 的梯度 (用于漂移项)
        # 漂移项: -div(rho * grad V)
        drift = -np.gradient(rho * grad_V, dx)

        # 扩散项: (1/beta) * Laplacian(rho)
        diffusion = (1/beta) * np.gradient(np.gradient(rho, dx), dx)

        # 更新
        rho = rho + tau * (drift + diffusion)
        rho = np.maximum(rho, 0)
        rho /= rho.sum()

        trajectory.append(rho.copy())

    return trajectory


def porous_medium_flow(rho0, m=2, n_steps=100, tau=0.001):
    """
    多孔介质方程

    F(rho) = (1/(m-1)) * integral rho^m
    梯度流: partial_t rho = Laplacian(rho^m)

    Parameters
    ----------
    rho0 : array (n,)
        初始分布
    m : float
        指数参数 (m > 1)
    n_steps : int
        时间步数
    tau : float
        时间步长

    Returns
    -------
    trajectory : list
        分布随时间的演化
    """
    rho = rho0.copy()
    trajectory = [rho.copy()]
    n = len(rho)
    dx = 1.0 / n

    for _ in range(n_steps):
        # 多孔介质项: Laplacian(rho^m)
        rho_m = rho**m
        laplacian = np.gradient(np.gradient(rho_m, dx), dx)

        # 更新
        rho = rho + tau * laplacian
        rho = np.maximum(rho, 0)
        rho /= rho.sum()

        trajectory.append(rho.copy())

    return trajectory


def demo_heat_equation():
    """演示热方程作为熵的梯度流"""
    print("=" * 60)
    print("演示 1: 热方程 (熵的梯度流)")
    print("=" * 60)

    # 定义初始分布 (高斯)
    x = np.linspace(-3, 3, 200)
    rho0 = np.exp(-x**2 / 0.1)
    rho0 /= rho0.sum()

    # 模拟热方程演化
    trajectory = heat_equation_flow(rho0, n_steps=200, tau=0.01, sigma=1.0)

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 初始分布
    ax = axes[0, 0]
    ax.plot(x, rho0, 'b-', linewidth=2)
    ax.set_title('初始分布')
    ax.set_xlabel('x')
    ax.set_ylabel('ρ(x)')
    ax.grid(True, alpha=0.3)

    # 中间状态
    ax = axes[0, 1]
    for i in [0, 50, 100, 150, 199]:
        alpha = 0.3 + 0.7 * (i / 200)
        ax.plot(x, trajectory[i], alpha=alpha, linewidth=2, label=f't={i*0.01:.2f}')
    ax.set_title('分布演化')
    ax.set_xlabel('x')
    ax.set_ylabel('ρ(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 熵随时间变化
    ax = axes[1, 0]
    entropies = []
    for rho in trajectory:
        # 计算微分熵 (数值稳定)
        rho_nonzero = rho[rho > 1e-10]
        entropy = -np.sum(rho_nonzero * np.log(rho_nonzero))
        entropies.append(entropy)
    ax.plot(entropies, 'b-', linewidth=2)
    ax.set_title('熵随时间增加')
    ax.set_xlabel('时间步')
    ax.set_ylabel('H(ρ) = -∫ ρ log ρ')
    ax.grid(True, alpha=0.3)

    # 方差随时间变化 (应该线性增长)
    ax = axes[1, 1]
    variances = []
    for rho in trajectory:
        mean = np.sum(x * rho)
        var = np.sum((x - mean)**2 * rho)
        variances.append(var)
    ax.plot(variances, 'r-', linewidth=2)
    ax.set_title('方差随时间增长')
    ax.set_xlabel('时间步')
    ax.set_ylabel('Var(ρ)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('heat_equation_flow.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 heat_equation_flow.png")
    plt.show()


def demo_fokker_planck():
    """演示 Fokker-Planck 方程"""
    print("\n" + "=" * 60)
    print("演示 2: Fokker-Planck 方程 (KL 散度的梯度流)")
    print("=" * 60)

    x = np.linspace(-3, 3, 200)
    dx = x[1] - x[0]

    # 定义势能函数 (双阱势)
    V = 0.5 * (x**2 - 1)**2

    # 目标平衡分布: pi ∝ exp(-V)
    pi = np.exp(-V)
    pi /= pi.sum()

    # 初始分布 (单峰)
    rho0 = np.exp(-(x + 1.5)**2 / 0.1)
    rho0 /= rho0.sum()

    # 模拟 Fokker-Planck 演化
    trajectory = fokker_planck_flow(rho0, V, n_steps=500, tau=0.001, beta=5.0)

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 势能和目标分布
    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.fill_between(x, pi, alpha=0.3, color='blue', label='平衡分布 π')
    ax2.plot(x, V, 'r-', linewidth=2, label='势能 V(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('π(x)', color='blue')
    ax2.set_ylabel('V(x)', color='red')
    ax.set_title('势能函数和平衡分布')
    ax.grid(True, alpha=0.3)

    # 演化过程
    ax = axes[0, 1]
    for i in [0, 100, 250, 499]:
        alpha = 0.3 + 0.7 * (i / 500)
        ax.plot(x, trajectory[i], alpha=alpha, linewidth=2, label=f't={i*0.001:.3f}')
    ax.plot(x, pi, 'k--', linewidth=2, label='平衡分布', alpha=0.5)
    ax.set_title('分布向平衡态演化')
    ax.set_xlabel('x')
    ax.set_ylabel('ρ(x)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # KL 散度随时间变化
    ax = axes[1, 0]
    kls = []
    for rho in trajectory:
        mask = (rho > 1e-10) & (pi > 1e-10)
        kl = np.sum(rho[mask] * np.log(rho[mask] / pi[mask]))
        kls.append(kl)
    ax.semilogy(kls, 'b-', linewidth=2)
    ax.set_title('KL(ρ||π) 随时间递减')
    ax.set_xlabel('时间步')
    ax.set_ylabel('KL 散度 (log scale)')
    ax.grid(True, alpha=0.3)

    # 能量分解
    ax = axes[1, 1]
    energies = []
    entropies = []
    for rho in trajectory:
        # 势能部分
        E_pot = np.sum(rho * V)
        # 熵部分
        mask = rho > 1e-10
        H = -np.sum(rho[mask] * np.log(rho[mask]))
        energies.append(E_pot)
        entropies.append(-H / 5.0)  # 缩放以便可视化
    ax.plot(energies, 'r-', linewidth=2, label='势能 ⟨V⟩')
    ax.plot(entropies, 'b-', linewidth=2, label='负熵 -H/β (缩放)')
    ax.set_title('自由能分解')
    ax.set_xlabel('时间步')
    ax.set_ylabel('能量')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fokker_planck_flow.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 fokker_planck_flow.png")
    plt.show()


def demo_comparison():
    """比较不同的梯度流"""
    print("\n" + "=" * 60)
    print("演示 3: 不同梯度流的比较")
    print("=" * 60)

    x = np.linspace(-3, 3, 200)

    # 相同的初始条件
    rho0 = np.exp(-x**2 / 0.1)
    rho0 /= rho0.sum()

    # 三种不同的演化
    traj_heat = heat_equation_flow(rho0, n_steps=100, tau=0.01, sigma=1.0)
    traj_porous = porous_medium_flow(rho0, m=2, n_steps=100, tau=0.001)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 热方程
    ax = axes[0]
    for i in [0, 25, 50, 99]:
        ax.plot(x, traj_heat[i], alpha=0.3 + 0.7*(i/100), linewidth=2)
    ax.set_title('热方程\n(熵的梯度流)')
    ax.set_xlabel('x')
    ax.set_ylabel('ρ(x)')
    ax.grid(True, alpha=0.3)

    # 多孔介质方程
    ax = axes[1]
    for i in [0, 25, 50, 99]:
        ax.plot(x, traj_porous[i], alpha=0.3 + 0.7*(i/100), linewidth=2)
    ax.set_title('多孔介质方程\n(m=2)')
    ax.set_xlabel('x')
    ax.grid(True, alpha=0.3)

    # 方差演化比较
    ax = axes[2]

    def compute_variance(traj, x):
        variances = []
        for rho in traj:
            mean = np.sum(x * rho)
            var = np.sum((x - mean)**2 * rho)
            variances.append(var)
        return variances

    var_heat = compute_variance(traj_heat, x)
    var_porous = compute_variance(traj_porous, x)

    ax.plot(var_heat, 'b-', linewidth=2, label='热方程')
    ax.plot(var_porous, 'r-', linewidth=2, label='多孔介质')
    ax.set_title('方差演化比较')
    ax.set_xlabel('时间步')
    ax.set_ylabel('方差')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gradient_flow_comparison.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 gradient_flow_comparison.png")
    plt.show()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Wasserstein 梯度流演示")
    print("基于: Optimal Transport for Machine Learners")
    print("作者: Gabriel Peyré (2025)")
    print("=" * 60)

    demo_heat_equation()
    demo_fokker_planck()
    demo_comparison()

    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("\n关键概念:")
    print("1. 热方程 = 熵的梯度流 (最大熵原理)")
    print("2. Fokker-Planck = KL 散度的梯度流")
    print("3. JKO 方案 = Wasserstein 空间中的梯度下降")
    print("=" * 60)
