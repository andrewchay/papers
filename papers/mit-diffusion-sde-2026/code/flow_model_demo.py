"""
流模型（Flow Model）演示
=====================

基于 MIT 6.S184 课程
使用 Rectified Flow（线性插值）
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class FlowModel(nn.Module):
    """简单的流模型"""

    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

        # 时间编码
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )

        # 主网络
        self.net = nn.Sequential(
            nn.Linear(dim + 64, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x, t):
        """
        x: (batch, dim)
        t: (batch,) 或 scalar
        """
        if isinstance(t, (int, float)):
            t = torch.ones(x.shape[0], 1) * t
        else:
            t = t.view(-1, 1)

        # 时间编码
        t_emb = self.time_embed(t)

        # 拼接
        h = torch.cat([x, t_emb], dim=-1)

        return self.net(h)

    def sample(self, n_samples, n_steps=100):
        """使用欧拉方法采样"""
        self.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, self.dim)  # x_0 ~ N(0, I)
            dt = 1.0 / n_steps

            trajectory = [x.clone()]

            for i in range(n_steps):
                t = i * dt
                v = self(x, t)
                x = x + dt * v
                trajectory.append(x.clone())

        return x, trajectory


def generate_training_data(n_samples=1000):
    """生成训练数据：8个高斯分布组成的环"""
    n_modes = 8
    angles = torch.linspace(0, 2 * np.pi, n_modes + 1)[:-1]

    samples_per_mode = n_samples // n_modes
    data = []

    for angle in angles:
        center = torch.tensor([torch.cos(angle), torch.sin(angle)]) * 3
        noise = torch.randn(samples_per_mode, 2) * 0.3
        data.append(center + noise)

    return torch.cat(data, dim=0)


def flow_matching_loss(model, x_1):
    """
    条件流匹配损失（Rectified Flow）

    x_1: (batch, dim) 真实数据
    """
    batch_size = x_1.shape[0]

    # 采样时间和初始噪声
    t = torch.rand(batch_size, 1)
    x_0 = torch.randn_like(x_1)

    # 构建条件路径
    x_t = (1 - t) * x_0 + t * x_1

    # 条件向量场
    u_target = x_1 - x_0

    # 预测向量场
    u_pred = model(x_t, t.squeeze())

    # 损失
    loss = torch.mean((u_pred - u_target) ** 2)

    return loss


def train_flow_model():
    """训练流模型"""
    print("=" * 60)
    print("训练流模型")
    print("=" * 60)

    # 生成数据
    data = generate_training_data(10000)
    print(f"\n生成 {len(data)} 个训练样本")
    print(f"数据形状: {data.shape}")

    # 创建模型
    model = FlowModel(dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 训练
    batch_size = 256
    n_epochs = 1000

    print(f"\n训练 {n_epochs} epochs...")

    for epoch in range(n_epochs):
        # 随机采样批次
        idx = torch.randint(0, len(data), (batch_size,))
        x_1 = data[idx]

        # 计算损失
        loss = flow_matching_loss(model, x_1)

        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss.item():.6f}")

    print(f"\n训练完成！Final Loss: {loss.item():.6f}")

    return model, data


def visualize_results(model, data):
    """可视化结果"""
    print("\n" + "=" * 60)
    print("生成可视化")
    print("=" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 左图：训练数据
    data_np = data.numpy()
    axes[0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.5, s=10)
    axes[0].set_title('Training Data (8 Gaussians)')
    axes[0].set_xlim(-6, 6)
    axes[0].set_ylim(-6, 6)
    axes[0].set_aspect('equal')

    # 中图：生成样本
    samples, trajectory = model.sample(1000, n_steps=50)
    samples_np = samples.numpy()
    axes[1].scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, s=10, c='red')
    axes[1].set_title('Generated Samples (Flow Model)')
    axes[1].set_xlim(-6, 6)
    axes[1].set_ylim(-6, 6)
    axes[1].set_aspect('equal')

    # 右图：采样轨迹
    trajectory_np = torch.stack(trajectory).numpy()
    n_show = 10  # 显示10条轨迹
    for i in range(n_show):
        axes[2].plot(trajectory_np[:, i, 0], trajectory_np[:, i, 1],
                     alpha=0.5, linewidth=1)
    axes[2].set_title('Sampling Trajectories (ODE paths)')
    axes[2].set_xlim(-6, 6)
    axes[2].set_ylim(-6, 6)
    axes[2].set_aspect('equal')

    plt.tight_layout()
    plt.savefig('flow_model_results.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 flow_model_results.png")


def demonstrate_interpolation():
    """演示两个样本之间的插值"""
    print("\n" + "=" * 60)
    print("演示流模型的插值能力")
    print("=" * 60)

    model = FlowModel(dim=2)

    # 两个随机初始点
    x_a = torch.randn(1, 2)
    x_b = torch.randn(1, 2)

    # 生成
    n_steps = 50

    model.eval()
    with torch.no_grad():
        # 从 x_a 出发
        trajectory_a = [x_a.clone()]
        x = x_a.clone()
        for i in range(n_steps):
            t = i / n_steps
            x = x + (1.0 / n_steps) * model(x, t)
            trajectory_a.append(x.clone())

        # 从 x_b 出发
        trajectory_b = [x_b.clone()]
        x = x_b.clone()
        for i in range(n_steps):
            t = i / n_steps
            x = x + (1.0 / n_steps) * model(x, t)
            trajectory_b.append(x.clone())

    trajectory_a = torch.cat(trajectory_a).numpy()
    trajectory_b = torch.cat(trajectory_b).numpy()

    # 可视化
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.plot(trajectory_a[:, 0], trajectory_a[:, 1], 'b-', linewidth=2, label='From x_a')
    ax.plot(trajectory_b[:, 0], trajectory_b[:, 1], 'r-', linewidth=2, label='From x_b')
    ax.scatter(x_a[0, 0], x_a[0, 1], c='blue', s=100, marker='o', zorder=5)
    ax.scatter(x_b[0, 0], x_b[0, 1], c='red', s=100, marker='o', zorder=5)

    ax.set_title('Flow Model: Interpolation via ODE paths')
    ax.legend()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')

    plt.savefig('flow_interpolation.png', dpi=150, bbox_inches='tight')
    print("\n插值可视化已保存到 flow_interpolation.png")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MIT 6.S184: Flow Model Demo")
    print("=" * 60)

    # 训练模型
    model, data = train_flow_model()

    # 可视化结果
    visualize_results(model, data)

    # 演示插值
    demonstrate_interpolation()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
