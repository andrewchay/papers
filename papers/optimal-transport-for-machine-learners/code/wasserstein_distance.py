#!/usr/bin/env python3
"""
Wasserstein 距离计算
===================

本代码演示不同类型的 Wasserstein 距离计算方法。
包括精确计算、近似计算、以及高斯分布的闭式解。

参考: "Optimal Transport for Machine Learners" by Gabriel Peyré (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy import linalg
import warnings


def wasserstein_1d_exact(mu, nu, x):
    """
    1D 分布的精确 Wasserstein-2 距离

    利用 1D 最优传输的闭式解：排序后的累积分布函数之差

    Parameters
    ----------
    mu, nu : array (n,)
        两个概率分布
    x : array (n,)
        支撑点（必须有序）

    Returns
    -------
    W2 : float
        Wasserstein-2 距离的平方
    """
    # 确保有序
    assert np.all(np.diff(x) >= 0), "x 必须有序"

    # 计算累积分布函数
    F_mu = np.cumsum(mu)
    F_nu = np.cumsum(nu)

    # 使用梯形法则积分
    diff = np.abs(F_mu - F_nu)
    W2 = np.sum(diff**2 * np.diff(x, prepend=x[0]))

    return np.sqrt(W2)


def bures_wasserstein(mu1, Sigma1, mu2, Sigma2, return_map=False):
    """
    计算两个高斯分布之间的 Bures-Wasserstein 距离

    公式:
    W_2^2 = ||mu1 - mu2||^2 + tr(Sigma1) + tr(Sigma2)
            - 2*tr((Sigma1^{1/2} Sigma2 Sigma1^{1/2})^{1/2})

    Parameters
    ----------
    mu1, mu2 : array (d,)
        均值向量
    Sigma1, Sigma2 : array (d, d)
        协方差矩阵
    return_map : bool
        是否返回最优传输映射

    Returns
    -------
    W2 : float
        Wasserstein-2 距离
    T : callable (optional)
        最优传输映射 T(x) = Ax + b
    """
    d = len(mu1)

    # 均值差异
    mean_diff = np.linalg.norm(mu1 - mu2)**2

    # 计算协方差矩阵的平方根 (使用特征值分解更稳定)
    def matrix_sqrt(M):
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        # 确保非负
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        return eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

    sqrt_Sigma1 = matrix_sqrt(Sigma1)

    # 计算 (Sigma1^{1/2} Sigma2 Sigma1^{1/2})^{1/2}
    product = sqrt_Sigma1 @ Sigma2 @ sqrt_Sigma1
    sqrt_product = matrix_sqrt(product)

    # Bures 距离
    cov_diff = np.trace(Sigma1) + np.trace(Sigma2) - 2 * np.trace(sqrt_product)
    W2 = np.sqrt(mean_diff + cov_diff)

    if return_map:
        # 最优传输映射: T(x) = A(x - mu1) + mu2
        # 其中 A = Sigma2^{1/2} (Sigma2^{1/2} Sigma1 Sigma2^{1/2})^{-1/2} Sigma2^{1/2}
        sqrt_Sigma2 = matrix_sqrt(Sigma2)
        product2 = sqrt_Sigma2 @ Sigma1 @ sqrt_Sigma2
        sqrt_product2_inv = linalg.inv(matrix_sqrt(product2))
        A = sqrt_Sigma2 @ sqrt_product2_inv @ sqrt_Sigma2

        def T(x):
            return A @ (x - mu1) + mu2

        return W2, T

    return W2


def sliced_wasserstein(mu_points, nu_points, mu_weights=None, nu_weights=None, n_projections=100):
    """
    切片 Wasserstein 距离

    通过随机投影到 1D，利用 1D OT 的闭式解近似高维 Wasserstein 距离

    Parameters
    ----------
    mu_points : array (n, d)
        源分布的样本点
    nu_points : array (m, d)
        目标分布的样本点
    mu_weights, nu_weights : array
        权重 (如果为 None，假设均匀)
    n_projections : int
        投影方向数

    Returns
    -------
    SW : float
        切片 Wasserstein 距离估计
    """
    d = mu_points.shape[1]

    if mu_weights is None:
        mu_weights = np.ones(len(mu_points)) / len(mu_points)
    if nu_weights is None:
        nu_weights = np.ones(len(nu_points)) / len(nu_points)

    distances = []

    for _ in range(n_projections):
        # 随机投影方向
        theta = np.random.randn(d)
        theta = theta / np.linalg.norm(theta)

        # 投影
        mu_proj = mu_points @ theta
        nu_proj = nu_points @ theta

        # 排序并计算 1D Wasserstein 距离
        mu_sorted_idx = np.argsort(mu_proj)
        nu_sorted_idx = np.argsort(nu_proj)

        mu_cumsum = np.cumsum(mu_weights[mu_sorted_idx])
        nu_cumsum = np.cumsum(nu_weights[nu_sorted_idx])

        # 计算经验 CDF 的差异
        W1_1d = np.sum(np.abs(mu_cumsum - nu_cumsum)) / len(mu_cumsum)
        distances.append(W1_1d)

    return np.mean(distances)


def wasserstein_barycenter(distributions, weights=None, epsilon=0.1, max_iter=100):
    """
    计算 Wasserstein 重心（简化版本）

    固定支撑，只优化权重

    Parameters
    ----------
    distributions : list of array
        分布列表，每个是 (n,) 数组
    weights : array
        每个分布的权重
    epsilon : float
        Sinkhorn 正则化参数
    max_iter : int
        最大迭代次数

    Returns
    -------
    barycenter : array
        Wasserstein 重心分布
    """
    n_distributions = len(distributions)
    n_points = distributions[0].shape[0]

    if weights is None:
        weights = np.ones(n_distributions) / n_distributions

    # 假设所有分布定义在相同的支撑上
    x = np.linspace(0, 1, n_points).reshape(-1, 1)
    C = cdist(x, x, metric='sqeuclidean')

    # 初始化重心为平均
    barycenter = np.mean(distributions, axis=0)
    barycenter /= barycenter.sum()

    # 迭代 Bregman 投影
    K = np.exp(-C / epsilon)

    for iteration in range(max_iter):
        barycenter_prev = barycenter.copy()

        # 对每个分布计算到重心的传输
        new_barycenter = np.zeros(n_points)
        for i, dist in enumerate(distributions):
            # Sinkhorn 一步
            u = np.ones(n_points)
            v = barycenter / (K.T @ u)
            u = dist / (K @ v)

            pi = np.diag(u) @ K @ np.diag(v)
            new_barycenter += weights[i] * pi.sum(axis=0)

        barycenter = new_barycenter
        barycenter /= barycenter.sum()

        # 检查收敛
        if np.linalg.norm(barycenter - barycenter_prev) < 1e-6:
            break

    return barycenter


def demo_gaussian_wasserstein():
    """演示高斯分布的 Bures-Wasserstein 距离"""
    print("=" * 60)
    print("演示: 高斯分布的 Bures-Wasserstein 距离")
    print("=" * 60)

    # 定义两个 2D 高斯分布
    mu1 = np.array([0, 0])
    Sigma1 = np.array([[1, 0.5],
                       [0.5, 1]])

    mu2 = np.array([3, 2])
    Sigma2 = np.array([[0.5, -0.3],
                       [-0.3, 0.8]])

    # 计算 Bures-Wasserstein 距离
    W2, T = bures_wasserstein(mu1, Sigma1, mu2, Sigma2, return_map=True)

    print(f"\n分布 1: μ={mu1}, Σ=\n{Sigma1}")
    print(f"\n分布 2: μ={mu2}, Σ=\n{Sigma2}")
    print(f"\nBures-Wasserstein 距离: {W2:.4f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 生成样本
    samples1 = np.random.multivariate_normal(mu1, Sigma1, 1000)
    samples2 = np.random.multivariate_normal(mu2, Sigma2, 1000)

    # 应用传输映射
    samples1_transformed = np.array([T(s) for s in samples1])

    ax = axes[0]
    ax.scatter(samples1[:, 0], samples1[:, 1], c='blue', alpha=0.3, s=10, label='源分布')
    ax.scatter(samples2[:, 0], samples2[:, 1], c='red', alpha=0.3, s=10, label='目标分布')
    ax.set_title('原始分布')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(samples1_transformed[:, 0], samples1_transformed[:, 1], c='blue', alpha=0.3, s=10, label='传输后的源')
    ax.scatter(samples2[:, 0], samples2[:, 1], c='red', alpha=0.3, s=10, label='目标分布')
    ax.set_title('应用最优传输映射后')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gaussian_wasserstein.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 gaussian_wasserstein.png")
    plt.show()


def demo_sliced_wasserstein():
    """演示切片 Wasserstein 距离"""
    print("\n" + "=" * 60)
    print("演示: 切片 Wasserstein 距离")
    print("=" * 60)

    np.random.seed(42)

    # 生成两个 2D 分布
    # 源: 两个高斯混合
    n = 500
    mask = np.random.rand(n) < 0.5
    mu_points = np.zeros((n, 2))
    mu_points[mask] = np.random.randn(mask.sum(), 2) * 0.3 + np.array([0, 0])
    mu_points[~mask] = np.random.randn((~mask).sum(), 2) * 0.3 + np.array([2, 2])

    # 目标: 平移后的分布
    nu_points = mu_points + np.array([3, 1]) + np.random.randn(n, 2) * 0.1

    # 计算切片 Wasserstein 距离
    n_projections_list = [10, 50, 100, 500]
    distances = []

    for n_proj in n_projections_list:
        SW = sliced_wasserstein(mu_points, nu_points, n_projections=n_proj)
        distances.append(SW)
        print(f"投影数: {n_proj:4d}, 切片 Wasserstein: {SW:.4f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(mu_points[:, 0], mu_points[:, 1], c='blue', alpha=0.5, s=10, label='源')
    ax.scatter(nu_points[:, 0], nu_points[:, 1], c='red', alpha=0.5, s=10, label='目标')
    ax.set_title('2D 点云分布')
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(n_projections_list, distances, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('投影方向数')
    ax.set_ylabel('切片 Wasserstein 距离')
    ax.set_title('距离 vs 投影数')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sliced_wasserstein.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 sliced_wasserstein.png")
    plt.show()


def demo_barycenter():
    """演示 Wasserstein 重心"""
    print("\n" + "=" * 60)
    print("演示: Wasserstein 重心")
    print("=" * 60)

    # 定义三个 1D 分布
    x = np.linspace(0, 1, 100)

    dists = []
    # 分布 1: 左偏
    d1 = np.exp(-((x - 0.3) / 0.1)**2)
    d1 /= d1.sum()
    dists.append(d1)

    # 分布 2: 右偏
    d2 = np.exp(-((x - 0.7) / 0.1)**2)
    d2 /= d2.sum()
    dists.append(d2)

    # 分布 3: 中间
    d3 = np.exp(-((x - 0.5) / 0.15)**2)
    d3 /= d3.sum()
    dists.append(d3)

    # 计算重心
    bary = wasserstein_barycenter(dists, weights=np.array([0.3, 0.3, 0.4]))

    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'red', 'green', 'black']
    labels = ['分布 1 (左)', '分布 2 (右)', '分布 3 (中)', 'Wasserstein 重心']

    for i, (d, color, label) in enumerate(zip(dists + [bary], colors, labels)):
        alpha = 0.6 if i < 3 else 1.0
        linewidth = 2 if i < 3 else 3
        ax.plot(x, d, color=color, linewidth=linewidth, alpha=alpha, label=label)

    ax.set_xlabel('x')
    ax.set_ylabel('概率密度')
    ax.set_title('Wasserstein 重心示例')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('wasserstein_barycenter.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 wasserstein_barycenter.png")
    plt.show()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Wasserstein 距离计算")
    print("基于: Optimal Transport for Machine Learners")
    print("作者: Gabriel Peyré (2025)")
    print("=" * 60)

    demo_gaussian_wasserstein()
    demo_sliced_wasserstein()
    demo_barycenter()

    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)
