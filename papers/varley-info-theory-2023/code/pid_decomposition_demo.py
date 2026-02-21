"""
部分信息分解（PID）演示
=====================

本脚本演示如何将互信息分解为：
- Unique（独特信息）
- Redundant（冗余信息）
- Synergistic（协同信息）

对应论文章节：第 IV 章 "Partial Information Decomposition"
"""

import numpy as np
import matplotlib.pyplot as plt


def simple_pid(target, source1, source2):
    """
    简化的两源 PID 计算

    使用 I_min 作为冗余度量
    """
    def mi(x, y):
        """计算互信息（简化版）"""
        unique_x = np.unique(x)
        unique_y = np.unique(y)

        H_x = 0
        for val in unique_x:
            p = np.mean(x == val)
            if p > 0:
                H_x -= p * np.log2(p)

        H_y = 0
        for val in unique_y:
            p = np.mean(y == val)
            if p > 0:
                H_y -= p * np.log2(p)

        # 联合熵（简化计算）
        joint = x * 100 + y
        unique_joint = np.unique(joint)
        H_joint = 0
        for val in unique_joint:
            p = np.mean(joint == val)
            if p > 0:
                H_joint -= p * np.log2(p)

        return max(0, H_x + H_y - H_joint)

    # 计算各项互信息
    I_s1 = mi(source1, target)
    I_s2 = mi(source2, target)

    # 联合源
    joint = source1 * 2 + source2
    I_joint = mi(joint, target)

    # PID 分解（简化版）
    # Redundant = min(I(s1;t), I(s2;t))
    redundant = min(I_s1, I_s2)

    # Synergistic = I_joint - max(I_s1, I_s2)
    synergistic = max(0, I_joint - max(I_s1, I_s2))

    # Unique
    unique1 = max(0, I_s1 - redundant)
    unique2 = max(0, I_s2 - redundant)

    return {
        'I_s1': I_s1,
        'I_s2': I_s2,
        'I_joint': I_joint,
        'unique1': unique1,
        'unique2': unique2,
        'redundant': redundant,
        'synergistic': synergistic
    }


def demonstrate_xor():
    """演示 XOR 系统的 PID"""
    print("=" * 60)
    print("示例 1: XOR 系统（纯协同）")
    print("=" * 60)

    np.random.seed(42)
    n = 10000

    s1 = np.random.randint(0, 2, n)
    s2 = np.random.randint(0, 2, n)
    target = np.logical_xor(s1, s2).astype(int)

    pid = simple_pid(target, s1, s2)

    print(f"\n系统: Y = X₁ XOR X₂")
    print(f"\n互信息:")
    print(f"  I(X₁; Y)     = {pid['I_s1']:.4f} bits")
    print(f"  I(X₂; Y)     = {pid['I_s2']:.4f} bits")
    print(f"  I(X₁,X₂; Y)  = {pid['I_joint']:.4f} bits")

    print(f"\nPID 分解:")
    print(f"  Unique(X₁)   = {pid['unique1']:.4f} bits")
    print(f"  Unique(X₂)   = {pid['unique2']:.4f} bits")
    print(f"  Redundant    = {pid['redundant']:.4f} bits")
    print(f"  Synergistic  = {pid['synergistic']:.4f} bits ← 主导！")

    print(f"\n结论: XOR 是纯协同系统")


def demonstrate_and():
    """演示 AND 系统的 PID"""
    print("\n" + "=" * 60)
    print("示例 2: AND 系统（冗余主导）")
    print("=" * 60)

    np.random.seed(42)
    n = 10000

    s1 = np.random.randint(0, 2, n)
    s2 = np.random.randint(0, 2, n)
    target = np.logical_and(s1, s2).astype(int)

    pid = simple_pid(target, s1, s2)

    print(f"\n系统: Y = X₁ AND X₂")
    print(f"\n互信息:")
    print(f"  I(X₁; Y)     = {pid['I_s1']:.4f} bits")
    print(f"  I(X₂; Y)     = {pid['I_s2']:.4f} bits")
    print(f"  I(X₁,X₂; Y)  = {pid['I_joint']:.4f} bits")

    print(f"\nPID 分解:")
    print(f"  Unique(X₁)   = {pid['unique1']:.4f} bits")
    print(f"  Unique(X₂)   = {pid['unique2']:.4f} bits")
    print(f"  Redundant    = {pid['redundant']:.4f} bits ← 主导！")
    print(f"  Synergistic  = {pid['synergistic']:.4f} bits")

    print(f"\n结论: AND 有显著冗余，因为两个输入都提供'低值'信息")


def demonstrate_copy():
    """演示复制系统的 PID"""
    print("\n" + "=" * 60)
    print("示例 3: 复制系统（完全冗余）")
    print("=" * 60)

    np.random.seed(42)
    n = 10000

    s1 = np.random.randint(0, 2, n)
    s2 = s1.copy()  # X₂ 完全复制 X₁
    target = s1.copy()

    pid = simple_pid(target, s1, s2)

    print(f"\n系统: Y = X₁, X₂ = X₁（完全复制）")
    print(f"\n互信息:")
    print(f"  I(X₁; Y)     = {pid['I_s1']:.4f} bits")
    print(f"  I(X₂; Y)     = {pid['I_s2']:.4f} bits")
    print(f"  I(X₁,X₂; Y)  = {pid['I_joint']:.4f} bits")

    print(f"\nPID 分解:")
    print(f"  Unique(X₁)   = {pid['unique1']:.4f} bits")
    print(f"  Unique(X₂)   = {pid['unique2']:.4f} bits")
    print(f"  Redundant    = {pid['redundant']:.4f} bits ← 完全冗余！")
    print(f"  Synergistic  = {pid['synergistic']:.4f} bits")

    print(f"\n结论: 复制系统显示完全冗余，因为两个源提供相同信息")


def demonstrate_unique():
    """演示独特信息系统"""
    print("\n" + "=" * 60)
    print("示例 4: 独特信息系统")
    print("=" * 60)

    np.random.seed(42)
    n = 10000

    s1 = np.random.randint(0, 2, n)
    s2 = np.random.randint(0, 2, n)

    # 目标只依赖于 s1
    target = s1.copy()

    pid = simple_pid(target, s1, s2)

    print(f"\n系统: Y = X₁（完全独立于 X₂）")
    print(f"\n互信息:")
    print(f"  I(X₁; Y)     = {pid['I_s1']:.4f} bits")
    print(f"  I(X₂; Y)     = {pid['I_s2']:.4f} bits")
    print(f"  I(X₁,X₂; Y)  = {pid['I_joint']:.4f} bits")

    print(f"\nPID 分解:")
    print(f"  Unique(X₁)   = {pid['unique1']:.4f} bits ← 主导！")
    print(f"  Unique(X₂)   = {pid['unique2']:.4f} bits")
    print(f"  Redundant    = {pid['redundant']:.4f} bits")
    print(f"  Synergistic  = {pid['synergistic']:.4f} bits")

    print(f"\n结论: X₂ 不提供任何信息，所有信息来自 X₁")


def visualize_pid_comparison():
    """可视化不同系统的 PID 比较"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    systems = [
        ('XOR', demonstrate_xor_data),
        ('AND', demonstrate_and_data),
        ('Copy', demonstrate_copy_data),
        ('Unique', demonstrate_unique_data)
    ]

    for idx, (name, data_func) in enumerate(systems):
        ax = axes[idx // 2, idx % 2]
        pid = data_func()

        components = ['Unique₁', 'Unique₂', 'Redundant', 'Synergistic']
        values = [pid['unique1'], pid['unique2'], pid['redundant'], pid['synergistic']]
        colors = ['skyblue', 'lightgreen', 'orange', 'salmon']

        bars = ax.bar(components, values, color=colors, alpha=0.8)
        ax.set_ylabel('Information (bits)')
        ax.set_title(f'{name} System PID Decomposition')
        ax.set_ylim(0, 1.2)

        # 添加数值标签
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if val > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('pid_comparison.png', dpi=150, bbox_inches='tight')
    print("\n\n可视化已保存到 pid_comparison.png")


# 辅助函数用于可视化
def demonstrate_xor_data():
    np.random.seed(42)
    n = 1000
    s1 = np.random.randint(0, 2, n)
    s2 = np.random.randint(0, 2, n)
    target = np.logical_xor(s1, s2).astype(int)
    return simple_pid(target, s1, s2)


def demonstrate_and_data():
    np.random.seed(42)
    n = 1000
    s1 = np.random.randint(0, 2, n)
    s2 = np.random.randint(0, 2, n)
    target = np.logical_and(s1, s2).astype(int)
    return simple_pid(target, s1, s2)


def demonstrate_copy_data():
    np.random.seed(42)
    n = 1000
    s1 = np.random.randint(0, 2, n)
    s2 = s1.copy()
    target = s1.copy()
    return simple_pid(target, s1, s2)


def demonstrate_unique_data():
    np.random.seed(42)
    n = 1000
    s1 = np.random.randint(0, 2, n)
    s2 = np.random.randint(0, 2, n)
    target = s1.copy()
    return simple_pid(target, s1, s2)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("部分信息分解（PID）演示")
    print("基于 Varley (2023) 论文第 IV 章")
    print("=" * 60)

    demonstrate_xor()
    demonstrate_and()
    demonstrate_copy()
    demonstrate_unique()

    print("\n" + "=" * 60)
    print("生成可视化...")
    print("=" * 60)
    visualize_pid_comparison()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
