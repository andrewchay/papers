"""
基础演示：熵与互信息
===================

本脚本演示信息论中最基础的两个概念：
1. 香农熵（Shannon Entropy）
2. 互信息（Mutual Information）

对应论文章节：第 II 章 "What Is Information?"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy


def discrete_entropy(samples, base=2):
    """
    计算离散香农熵

    公式: H(X) = -Σ p(x) log p(x)

    参数:
        samples: 观测样本序列
        base: 对数底（2=bits, e=nats）

    返回:
        H: 熵值
    """
    unique, counts = np.unique(samples, return_counts=True)
    probs = counts / len(samples)

    # 避免 log(0)
    probs_nonzero = probs[probs > 0]
    H = -np.sum(probs_nonzero * np.log(probs_nonzero))

    if base == 2:
        H /= np.log(2)

    return H


def discrete_mutual_information(x, y, base=2):
    """
    计算离散互信息

    公式: I(X;Y) = H(X) + H(Y) - H(X,Y)

    参数:
        x, y: 观测样本序列
        base: 对数底

    返回:
        mi: 互信息值
    """
    # 计算联合熵
    joint_samples = np.column_stack([x, y])
    H_joint = discrete_entropy(joint_samples[:, 0] * 100 + joint_samples[:, 1], base=base)

    H_x = discrete_entropy(x, base=base)
    H_y = discrete_entropy(y, base=base)

    # I(X;Y) = H(X) + H(Y) - H(X,Y)
    mi = H_x + H_y - H_joint

    return max(0, mi)


def demonstrate_entropy():
    """演示熵的概念"""
    print("=" * 60)
    print("演示 1: 熵与不确定性")
    print("=" * 60)

    # 公平硬币 vs 作弊硬币
    np.random.seed(42)
    fair_coin = np.random.choice([0, 1], size=10000)  # p=0.5
    biased_coin = np.random.choice([0, 1], size=10000, p=[0.1, 0.9])

    H_fair = discrete_entropy(fair_coin)
    H_biased = discrete_entropy(biased_coin)
    H_max = np.log2(2)  # 理论最大值

    print(f"\n公平硬币 (p=0.5):")
    print(f"  熵 = {H_fair:.4f} bits")
    print(f"  理论最大值 = {H_max:.4f} bits")
    print(f"  解释: 每次投掷都不确定")

    print(f"\n作弊硬币 (p=0.1, 0.9):")
    print(f"  熵 = {H_biased:.4f} bits")
    print(f"  理论最大值 = {H_max:.4f} bits")
    print(f"  解释: 更确定会出现 1")

    print(f'\n关键洞察: 熵度量"不确定性"。')
    print(f"  越确定 → 熵越低")
    print(f"  越不确定 → 熵越高")


def demonstrate_mutual_information():
    """演示互信息的概念"""
    print("\n" + "=" * 60)
    print("演示 2: 互信息与统计依赖")
    print("=" * 60)

    np.random.seed(42)
    n = 5000

    # 场景 1: 线性关系
    x = np.random.randn(n)
    y_linear = x + 0.1 * np.random.randn(n)

    # 场景 2: 非线性关系
    y_nonlinear = x**2 + 0.1 * np.random.randn(n)

    # 场景 3: 独立
    y_independent = np.random.randn(n)

    # 离散化后计算
    bins = 10
    x_d = np.digitize(x, np.histogram_bin_edges(x, bins=bins))
    y_l_d = np.digitize(y_linear, np.histogram_bin_edges(y_linear, bins=bins))
    y_nl_d = np.digitize(y_nonlinear, np.histogram_bin_edges(y_nonlinear, bins=bins))
    y_ind_d = np.digitize(y_independent, np.histogram_bin_edges(y_independent, bins=bins))

    mi_linear = discrete_mutual_information(x_d, y_l_d)
    mi_nonlinear = discrete_mutual_information(x_d, y_nl_d)
    mi_independent = discrete_mutual_information(x_d, y_ind_d)

    print(f"\n场景 1: 线性关系 (Y = X + noise)")
    print(f"  互信息 = {mi_linear:.4f} bits")
    print(f"  解释: 知道 X 可以预测 Y")

    print(f"\n场景 2: 非线性关系 (Y = X² + noise)")
    print(f"  互信息 = {mi_nonlinear:.4f} bits")
    print(f"  解释: 即使非线性，仍有统计依赖")

    print(f"\n场景 3: 独立变量")
    print(f"  互信息 = {mi_independent:.4f} bits")
    print(f"  解释: 知道 X 对预测 Y 没有帮助")

    print(f"\n关键洞察: 互信息捕捉任何形式的统计依赖（不限于线性）")


def demonstrate_xor_synergy():
    """演示 XOR 协同效应"""
    print("\n" + "=" * 60)
    print("演示 3: XOR 协同效应（论文核心示例）")
    print("=" * 60)

    np.random.seed(42)
    n = 10000

    # 生成随机输入
    x1 = np.random.randint(0, 2, n)
    x2 = np.random.randint(0, 2, n)
    y_xor = np.logical_xor(x1, x2).astype(int)

    # 计算各种互信息
    I_x1_y = discrete_mutual_information(x1, y_xor)
    I_x2_y = discrete_mutual_information(x2, y_xor)

    # 联合输入
    joint_input = x1 * 2 + x2
    I_joint_y = discrete_mutual_information(joint_input, y_xor)

    print(f"\nXOR 系统: Y = X₁ XOR X₂")
    print(f"  I(X₁; Y) = {I_x1_y:.4f} bits")
    print(f"  I(X₂; Y) = {I_x2_y:.4f} bits")
    print(f"  I(X₁, X₂; Y) = {I_joint_y:.4f} bits")

    print(f"\n关键发现:")
    print(f"  - 单独看 X₁ 或 X₂: 不能预测 Y")
    print(f"  - 一起看 X₁ 和 X₂: 完全预测 Y")
    print(f"  - 这是'协同信息'的经典示例！")

    print(f"\n实际意义:")
    print(f"  如果只分析成对关系，会完全错过系统的信息结构。")
    print(f"  这在神经科学、生态学中很常见。")


def visualize_entropy():
    """可视化熵的概念"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 公平分布
    probs_fair = [0.25, 0.25, 0.25, 0.25]
    H_fair = -sum(p * np.log2(p) for p in probs_fair if p > 0)

    axes[0].bar(range(4), probs_fair, color='blue', alpha=0.7)
    axes[0].set_title(f'均匀分布\n熵 = {H_fair:.2f} bits\n(最大不确定性)', fontsize=12)
    axes[0].set_xlabel('状态')
    axes[0].set_ylabel('概率')
    axes[0].set_ylim(0, 1)

    # 偏斜分布
    probs_skewed = [0.7, 0.1, 0.1, 0.1]
    H_skewed = -sum(p * np.log2(p) for p in probs_skewed if p > 0)

    axes[1].bar(range(4), probs_skewed, color='orange', alpha=0.7)
    axes[1].set_title(f'偏斜分布\n熵 = {H_skewed:.2f} bits\n(中等不确定性)', fontsize=12)
    axes[1].set_xlabel('状态')
    axes[1].set_ylim(0, 1)

    # 确定性分布
    probs_det = [1.0, 0, 0, 0]
    H_det = 0

    axes[2].bar(range(4), probs_det, color='green', alpha=0.7)
    axes[2].set_title(f'确定性分布\n熵 = {H_det:.2f} bits\n(无不确定性)', fontsize=12)
    axes[2].set_xlabel('状态')
    axes[2].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('entropy_visualization.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 entropy_visualization.png")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("信息论基础演示")
    print("基于 Varley (2023) 论文第 II 章")
    print("=" * 60)

    demonstrate_entropy()
    demonstrate_mutual_information()
    demonstrate_xor_synergy()
    visualize_entropy()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
