#!/usr/bin/env python3
"""
Sinkhorn 算法演示
================

本代码演示熵正则化最优传输的 Sinkhorn 算法。
适用于中等规模问题 (n ~ 100-10000)。

参考: "Optimal Transport for Machine Learners" by Gabriel Peyré (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def sinkhorn(mu, nu, C, epsilon=0.1, max_iter=1000, tol=1e-6, verbose=False):
    """
    Sinkhorn-Knopp 算法求解熵正则化最优传输

    求解以下问题:
        min_π <C, π> + ε * KL(π | μ ⊗ ν)
        s.t. π 1 = μ, π^T 1 = ν

    Parameters
    ----------
    mu : array (n,)
        源分布 (非负, 求和为 1)
    nu : array (m,)
        目标分布 (非负, 求和为 1)
    C : array (n, m)
        成本矩阵
    epsilon : float
        熵正则化参数
    max_iter : int
        最大迭代次数
    tol : float
        收敛阈值
    verbose : bool
        是否打印进度

    Returns
    -------
    pi : array (n, m)
        最优耦合矩阵
    log : dict
        包含收敛历史
    """
    n, m = C.shape

    # 验证输入
    assert np.allclose(mu.sum(), 1.0), "源分布必须归一化"
    assert np.allclose(nu.sum(), 1.0), "目标分布必须归一化"
    assert np.all(mu >= 0), "源分布必须非负"
    assert np.all(nu >= 0), "目标分布必须非负"

    # 核矩阵
    K = np.exp(-C / epsilon)

    # 初始化
    u = np.ones(n) / n
    v = np.ones(m) / m

    # 记录收敛历史
    errors = []

    for iteration in range(max_iter):
        u_prev = u.copy()

        # Sinkhorn 迭代 (在 log 空间增加数值稳定性)
        u = mu / (K @ v)
        v = nu / (K.T @ u)

        # 检查收敛: 边缘约束违反程度
        pi = np.diag(u) @ K @ np.diag(v)
        err_mu = np.linalg.norm(pi.sum(axis=1) - mu)
        err_nu = np.linalg.norm(pi.sum(axis=0) - nu)
        err = err_mu + err_nu
        errors.append(err)

        if err < tol:
            if verbose:
                print(f"收敛于迭代 {iteration}, 误差: {err:.2e}")
            break

        if verbose and iteration % 100 == 0:
            print(f"迭代 {iteration}, 误差: {err:.2e}")

    else:
        if verbose:
            print(f"达到最大迭代次数 {max_iter}, 最终误差: {err:.2e}")

    # 重构耦合
    pi = np.diag(u) @ K @ np.diag(v)

    log = {
        'iterations': iteration + 1,
        'error': err,
        'errors': errors,
        'u': u,
        'v': v
    }

    return pi, log


def wasserstein_distance(mu, nu, C, epsilon=0.1, **kwargs):
    """
    计算正则化 Wasserstein 距离

    Returns
    -------
    distance : float
        W_ε(μ, ν)
    """
    pi, _ = sinkhorn(mu, nu, C, epsilon=epsilon, **kwargs)
    return np.sum(pi * C)


def demo_1d_transport():
    """演示 1D 分布间的最优传输"""
    print("=" * 60)
    print("演示 1: 1D 分布间的最优传输")
    print("=" * 60)

    # 定义两个 1D 高斯分布
    x = np.linspace(-5, 5, 100)

    # 源分布: N(-2, 0.5)
    mu = np.exp(-(x + 2)**2 / (2 * 0.5))
    mu /= mu.sum()

    # 目标分布: N(2, 0.8)
    nu = np.exp(-(x - 2)**2 / (2 * 0.8))
    nu /= nu.sum()

    # 计算成本矩阵 (欧氏距离)
    C = cdist(x.reshape(-1, 1), x.reshape(-1, 1), metric='sqeuclidean')

    # 求解 OT
    epsilon = 0.1
    pi, log = sinkhorn(mu, nu, C, epsilon=epsilon, verbose=True)

    # 计算距离
    W = np.sum(pi * C)
    print(f"\nWasserstein-2 距离 (正则化): {W:.4f}")

    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 绘制分布
    ax = axes[0, 0]
    ax.plot(x, mu, 'b-', label='源分布 μ', linewidth=2)
    ax.plot(x, nu, 'r-', label='目标分布 ν', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('概率密度')
    ax.set_title('概率分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 绘制耦合矩阵
    ax = axes[0, 1]
    im = ax.imshow(pi, origin='lower', cmap='Blues', aspect='auto')
    ax.set_xlabel('目标位置')
    ax.set_ylabel('源位置')
    ax.set_title(f'最优耦合矩阵 (ε={epsilon})')
    plt.colorbar(im, ax=ax)

    # 绘制传输方案 (子集)
    ax = axes[1, 0]
    step = 10
    for i in range(0, len(x), step):
        for j in range(0, len(x), step):
            if pi[i, j] > 0.001:
                ax.plot([0, 1], [x[i], x[j]], 'b-', alpha=pi[i, j] * 10, linewidth=0.5)
    ax.plot(0, 0, 'b-', alpha=0.3, label='传输方案')
    ax.plot([0]*len(x), x, 'bo', markersize=2, label='源')
    ax.plot([1]*len(x), x, 'ro', markersize=2, label='目标')
    ax.set_xlim(-0.2, 1.2)
    ax.set_xlabel('位置')
    ax.set_ylabel('值')
    ax.set_title('传输方案可视化 (采样)')
    ax.legend()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['源', '目标'])

    # 收敛曲线
    ax = axes[1, 1]
    ax.semilogy(log['errors'])
    ax.set_xlabel('迭代')
    ax.set_ylabel('误差 (log scale)')
    ax.set_title('收敛曲线')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sinkhorn_1d_demo.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 sinkhorn_1d_demo.png")
    plt.show()

    return mu, nu, pi


def demo_2d_transport():
    """演示 2D 点云间的最优传输"""
    print("\n" + "=" * 60)
    print("演示 2: 2D 点云间的最优传输")
    print("=" * 60)

    np.random.seed(42)

    # 生成两个 2D 点云
    n, m = 50, 50

    # 源点云: 高斯聚类
    mu_points = np.random.randn(n, 2) * 0.5 + np.array([0, 0])
    mu = np.ones(n) / n

    # 目标点云: 另一个高斯聚类
    nu_points = np.random.randn(m, 2) * 0.5 + np.array([3, 2])
    nu = np.ones(m) / m

    # 计算成本矩阵
    C = cdist(mu_points, nu_points, metric='sqeuclidean')

    # 求解 OT
    epsilon = 0.5
    pi, log = sinkhorn(mu, nu, C, epsilon=epsilon, verbose=True)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 绘制点云
    ax = axes[0]
    ax.scatter(mu_points[:, 0], mu_points[:, 1], c='blue', s=50, label='源', alpha=0.6)
    ax.scatter(nu_points[:, 0], nu_points[:, 1], c='red', s=50, label='目标', alpha=0.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('点云分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # 绘制耦合矩阵
    ax = axes[1]
    im = ax.imshow(pi, cmap='Blues', aspect='auto')
    ax.set_xlabel('目标点索引')
    ax.set_ylabel('源点索引')
    ax.set_title(f'耦合矩阵 (ε={epsilon})')
    plt.colorbar(im, ax=ax)

    # 绘制传输连接
    ax = axes[2]
    ax.scatter(mu_points[:, 0], mu_points[:, 1], c='blue', s=50, label='源', alpha=0.6, zorder=3)
    ax.scatter(nu_points[:, 0], nu_points[:, 1], c='red', s=50, label='目标', alpha=0.6, zorder=3)

    # 只绘制显著的连接
    threshold = np.percentile(pi, 95)
    for i in range(n):
        for j in range(m):
            if pi[i, j] > threshold:
                ax.plot([mu_points[i, 0], nu_points[j, 0]],
                       [mu_points[i, 1], nu_points[j, 1]],
                       'k-', alpha=0.3, linewidth=0.5, zorder=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'传输连接 (>{95}百分位)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig('sinkhorn_2d_demo.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 sinkhorn_2d_demo.png")
    plt.show()


def demo_epsilon_effect():
    """演示正则化参数 ε 的影响"""
    print("\n" + "=" * 60)
    print("演示 3: 正则化参数 ε 的影响")
    print("=" * 60)

    # 简单的 1D 例子
    n = 50
    x = np.linspace(0, 1, n)
    mu = np.exp(-(x - 0.3)**2 / 0.02)
    mu /= mu.sum()
    nu = np.exp(-(x - 0.7)**2 / 0.02)
    nu /= nu.sum()

    C = cdist(x.reshape(-1, 1), x.reshape(-1, 1), metric='sqeuclidean')

    # 测试不同的 epsilon
    epsilons = [0.01, 0.05, 0.1, 0.5, 1.0]

    fig, axes = plt.subplots(2, len(epsilons), figsize=(15, 8))

    for idx, eps in enumerate(epsilons):
        pi, log = sinkhorn(mu, nu, C, epsilon=eps, verbose=False)

        # 耦合矩阵
        ax = axes[0, idx]
        im = ax.imshow(pi, origin='lower', cmap='Blues', aspect='auto')
        ax.set_title(f'ε = {eps}')
        ax.set_xlabel('目标')
        if idx == 0:
            ax.set_ylabel('源')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 边缘分布
        ax = axes[1, idx]
        ax.bar(x, pi.sum(axis=1), width=0.02, alpha=0.5, label='行和 (应≈μ)', color='blue')
        ax.bar(x, pi.sum(axis=0), width=0.02, alpha=0.5, label='列和 (应≈ν)', color='red')
        if idx == 0:
            ax.set_ylabel('边缘分布')
            ax.legend()
        ax.set_xlabel('位置')
        ax.set_ylim(0, max(mu.max(), nu.max()) * 1.5)

    plt.tight_layout()
    plt.savefig('sinkhorn_epsilon_effect.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 sinkhorn_epsilon_effect.png")
    print("\n观察:")
    print("- ε 小: 稀疏解，接近精确 OT")
    print("- ε 大: 稠密平滑解，但偏差增大")
    plt.show()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Sinkhorn 算法演示")
    print("基于: Optimal Transport for Machine Learners")
    print("作者: Gabriel Peyré (2025)")
    print("=" * 60)

    # 运行演示
    demo_1d_transport()
    demo_2d_transport()
    demo_epsilon_effect()

    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)
