"""
Epiplexity 估计器演示
====================

本脚本演示如何估计数据的 Epiplexity 和 Time-Bounded Entropy
基于 Finzi et al. (2026) 的 Prequential Coding 方法
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class SimpleLanguageModel(nn.Module):
    """简单的语言模型用于演示"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

    def loss(self, x, targets):
        """计算负对数似然"""
        logits = self(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction='mean'
        )
        return loss


def train_and_record_losses(model, data_loader, optimizer,
                           num_steps, device='cpu'):
    """
    训练模型并记录损失曲线

    参数:
        model: 神经网络模型
        data_loader: 数据加载器
        optimizer: 优化器
        num_steps: 训练步数

    返回:
        losses: 损失列表
    """
    model.train()
    losses = []

    step = 0
    epoch = 0

    while step < num_steps:
        for batch_x, batch_y in data_loader:
            if step >= num_steps:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            loss = model.loss(batch_x, batch_y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            step += 1

            if step % 100 == 0:
                print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")

        epoch += 1

    return losses


def estimate_epiplexity_preq(losses, final_model_loss=None):
    """
    使用 Prequential Coding 估计 Epiplexity

    公式: |P| ≈ Σᵢ (lossᵢ - loss_final)

    参数:
        losses: 训练损失列表
        final_model_loss: 最终模型在训练数据上的损失（可选）

    返回:
        epiplexity: Epiplexity 估计（单位：nats 或 bits）
        entropy: Time-Bounded Entropy 估计
    """
    if final_model_loss is None:
        # 使用最后10个损失的均值作为最终损失
        final_loss = np.mean(losses[-10:])
    else:
        final_loss = final_model_loss

    # 计算 Epiplexity：损失曲线下的面积
    epiplexity = sum(l - final_loss for l in losses)

    # Time-Bounded Entropy：最终损失乘以数据量
    entropy = final_loss * len(losses)

    return epiplexity, entropy, final_loss


def generate_synthetic_data(data_type='pattern', n_samples=10000, seq_len=50):
    """
    生成不同类型的合成数据

    参数:
        data_type: 'pattern', 'random', 'chaotic'
        n_samples: 样本数
        seq_len: 序列长度

    返回:
        data: (n_samples, seq_len) 数据
    """
    if data_type == 'pattern':
        # 简单重复模式：低 Epiplexity
        base_pattern = np.array([1, 2, 3, 4, 5])
        data = np.tile(base_pattern, (n_samples, seq_len // 5 + 1))[:, :seq_len]

    elif data_type == 'random':
        # 完全随机：高 Entropy，低 Epiplexity
        data = np.random.randint(0, 10, (n_samples, seq_len))

    elif data_type == 'chaotic':
        # 混沌系统（简化版）：中等 Epiplexity
        data = []
        for _ in range(n_samples):
            x = np.random.rand()
            seq = []
            for _ in range(seq_len):
                # Logistic map: x_{n+1} = r * x_n * (1 - x_n)
                x = 3.9 * x * (1 - x)
                seq.append(int(x * 10))
            data.append(seq)
        data = np.array(data)

    elif data_type == 'hierarchical':
        # 层次结构：高 Epiplexity
        data = []
        for _ in range(n_samples):
            seq = []
            for i in range(seq_len):
                if i % 10 == 0:
                    # 每10步一个大模式
                    val = np.random.randint(0, 5)
                else:
                    # 小模式依赖于大模式
                    val = (seq[-1] + 1) % 10 if seq else 0
                seq.append(val)
            data.append(seq)
        data = np.array(data)

    return data


def compare_data_epiplexity():
    """比较不同类型数据的 Epiplexity"""
    print("=" * 60)
    print("Epiplexity 比较实验")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")

    # 数据配置
    data_types = ['pattern', 'random', 'chaotic', 'hierarchical']
    n_samples = 1000
    seq_len = 50
    vocab_size = 10

    results = {}

    for data_type in data_types:
        print(f"\n{'-' * 60}")
        print(f"数据类型: {data_type}")
        print(f"{'-' * 60}")

        # 生成数据
        data = generate_synthetic_data(data_type, n_samples, seq_len)

        # 准备训练数据（预测下一个token）
        x_data = torch.LongTensor(data[:, :-1])
        y_data = torch.LongTensor(data[:, 1:])
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # 创建模型
        model = SimpleLanguageModel(vocab_size, embedding_dim=32,
                                   hidden_dim=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 训练
        num_steps = 500
        print(f"\n训练模型 ({num_steps} steps)...")
        losses = train_and_record_losses(model, dataloader, optimizer,
                                        num_steps, device)

        # 估计 Epiplexity
        epiplexity, entropy, final_loss = estimate_epiplexity_preq(losses)

        results[data_type] = {
            'epiplexity': epiplexity,
            'entropy': entropy,
            'final_loss': final_loss,
            'losses': losses
        }

        print(f"\n结果:")
        print(f"  Epiplexity (S_T): {epiplexity:.2f} nats")
        print(f"  Time-Bounded Entropy (H_T): {entropy:.2f} nats")
        print(f"  Final Loss: {final_loss:.4f}")

    # 可视化
    visualize_comparison(results)

    return results


def visualize_comparison(results):
    """可视化比较结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 图1：Epiplexity 和 Entropy 对比
    data_types = list(results.keys())
    epiplexities = [results[t]['epiplexity'] for t in data_types]
    entropies = [results[t]['entropy'] for t in data_types]

    x = np.arange(len(data_types))
    width = 0.35

    axes[0].bar(x - width/2, epiplexities, width, label='Epiplexity (S_T)',
               color='skyblue', alpha=0.8)
    axes[0].bar(x + width/2, entropies, width, label='Time-Bounded Entropy (H_T)',
               color='salmon', alpha=0.8)

    axes[0].set_xlabel('Data Type')
    axes[0].set_ylabel('Information (nats)')
    axes[0].set_title('Epiplexity vs Entropy by Data Type')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(data_types, rotation=15)
    axes[0].legend()

    # 图2：损失曲线
    for data_type in data_types:
        losses = results[data_type]['losses']
        axes[1].plot(losses, label=data_type, alpha=0.7)

    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss Curves')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('epiplexity_comparison.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到 epiplexity_comparison.png")


def demonstrate_paradox1():
    """
    演示悖论1：确定性过程可以产生信息

    使用混沌系统（Logistic Map）
    """
    print("\n" + "=" * 60)
    print("演示：确定性过程产生信息（混沌系统）")
    print("=" * 60)

    # Logistic Map: x_{n+1} = r * x_n * (1 - x_n)
    r = 3.9  # 混沌区域
    x0 = 0.5

    # 生成序列
    n_steps = 1000
    sequence = [x0]
    x = x0
    for _ in range(n_steps - 1):
        x = r * x * (1 - x)
        sequence.append(x)

    # 量化
    quantized = [int(x * 100) % 10 for x in sequence]

    # 分析
    print(f"\n生成 {n_steps} 步混沌序列")
    print(f"初始条件: x0 = {x0}")
    print(f"参数: r = {r}")

    # 计算熵
    from collections import Counter
    counts = Counter(quantized)
    probs = [c / len(quantized) for c in counts.values()]
    entropy = -sum(p * np.log(p) for p in probs if p > 0)

    print(f"\n序列统计:")
    print(f"  香农熵: {entropy:.4f} nats")
    print(f"  不同状态数: {len(counts)}")

    print(f"\n关键洞察:")
    print(f"  - 生成规则极其简单（一行代码）")
    print(f"  - 但序列看起来随机（高熵）")
    print(f"  - 对多项式时间观察者来说，这是'新'信息")

    return sequence


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Epiplexity 估计演示")
    print("基于 Finzi et al. (2026)")
    print("=" * 60)

    # 运行比较实验
    results = compare_data_epiplexity()

    # 演示悖论1
    demonstrate_paradox1()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
