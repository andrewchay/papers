# 流匹配（Flow Matching）

---

## 1. 核心问题

**生成模型的目标**：
将简单分布（如高斯分布）转换为复杂数据分布。

```
p_init  ──→  p_data
N(0,I)      图像/文本/...
```

**方法**：学习一个向量场 u_t，使得沿着这个场的粒子从 p_init 流向 p_data。

---

## 2. 数学框架

### 2.1 连续性方程

概率密度 p_t(x) 随时间的演化遵循**连续性方程**：

```
∂p_t/∂t + ∇ · (p_t u_t) = 0
```

其中：
- p_t(x): 时间 t 时的概率密度
- u_t(x): 向量场（概率流的速度）

**直观理解**：
- 散度 ∇ · (p_t u_t) 表示概率的流出
- 密度变化 = - 概率流出（质量守恒）

### 2.2 从路径到向量场

给定一族概率路径 {p_t}_{t∈[0,1]}，如何找到对应的向量场 u_t？

**Marginal Preserving Flow**：

如果 u_t 满足：
```
u_t(x) = E[Ẋ_t | X_t = x]
```

其中 (X_t)_{t∈[0,1]} 是从 p_t 采样的随机过程，则 u_t 保持边缘分布 p_t。

**证明**：见讲义 Appendix A

---

## 3. 流匹配损失

### 3.1 回归问题

**目标**：学习神经网络 u_θ 来近似 u_t

**流匹配损失**：
```
L_FM(θ) = E_{t,X_t} [||u_θ(X_t, t) - u_t(X_t)||²]
```

其中 t ~ Uniform[0,1], X_t ~ p_t。

### 3.2 条件流匹配

**问题**：我们不知道 u_t(X_t)！

**解决方案**：使用条件路径

给定数据点 x_1，定义条件路径：
```
X_t = a_t x_1 + b_t ε,  ε ~ N(0,I)
```

例如，线性插值（Rectified Flow）：
```
X_t = (1-t) x_0 + t x_1
```

**条件向量场**：
```
u_t(X_t | x_1) = (ẋ_t - σ̇_t ε) / ...
```

**条件流匹配损失**：
```
L_CFM(θ) = E_{t,x_1,X_t} [||u_θ(X_t, t) - u_t(X_t | x_1)||²]
```

### 3.3 等价性定理

**定理**：流匹配损失和条件流匹配损失在期望上等价（最多相差一个与 θ 无关的常数）。

```
∇_θ L_FM(θ) = ∇_θ L_CFM(θ)
```

**意义**：我们可以使用条件路径来计算梯度，而不需要知道真实的向量场 u_t。

---

## 4. 常见条件路径

### 4.1 Rectified Flow（线性插值）

**路径**：
```
X_t = (1-t) x_0 + t x_1
```

其中 x_0 ~ N(0,I), x_1 ~ p_data。

**向量场**：
```
u_t(X_t | x_1) = x_1 - x_0 = (X_t - x_0) / t
```

**特点**：
- 最简单
- 路径是直线
- 但可能不是最优传输

### 4.2 高斯路径（扩散路径）

**路径**：
```
X_t = α_t x_1 + σ_t ε
```

**选择**：
- VP-SDE: α_t = exp(-½∫β(s)ds), σ_t = √(1-α_t²)
- VE-SDE: α_t = 1, σ_t = √t

**向量场**：
```
u_t(X_t | x_1) = ẋ_t(x_1, ε) - σ̇_t ε
```

---

## 5. 实现细节

### 5.1 PyTorch 实现

```python
import torch
import torch.nn as nn

class FlowModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # 时间条件化的神经网络
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x, t):
        """
        x: (batch, dim)
        t: (batch,) or scalar
        """
        if isinstance(t, (int, float)):
            t = torch.ones(x.shape[0], 1) * t
        else:
            t = t.view(-1, 1)

        return self.net(torch.cat([x, t], dim=-1))

    def sample(self, n_samples, n_steps=100):
        """使用欧拉方法采样"""
        x = torch.randn(n_samples, self.dim)  # x_0 ~ N(0,I)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = i * dt
            x = x + dt * self(x, t)

        return x

def flow_matching_loss(model, x_1):
    """
    计算条件流匹配损失
    x_1: (batch, dim) 真实数据
    """
    batch_size = x_1.shape[0]

    # 采样时间和初始噪声
    t = torch.rand(batch_size, 1)
    x_0 = torch.randn_like(x_1)

    # 构建条件路径（Rectified Flow）
    x_t = (1 - t) * x_0 + t * x_1

    # 条件向量场
    u_target = x_1 - x_0

    # 预测向量场
    u_pred = model(x_t, t)

    # 损失
    loss = torch.mean((u_pred - u_target) ** 2)

    return loss
```

### 5.2 训练循环

```python
def train_flow_model(model, dataloader, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for x_1 in dataloader:
            loss = flow_matching_loss(model, x_1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return model
```

---

## 6. 与扩散模型的联系

### 6.1 概率流ODE

对于SDE：
```
dX_t = u_t(X_t) dt + σ_t dW_t
```

存在对应的**概率流ODE**：
```
dX_t = [u_t(X_t) - ½ σ_t² ∇ log p_t(X_t)] dt
```

**关键洞察**：
- SDE（随机）：更好的模式覆盖
- ODE（确定性）：相同的边缘分布，可逆

### 6.2 分数匹配

分数函数定义为：
```
s_t(x) = ∇ log p_t(x)
```

概率流ODE可以写成：
```
dX_t = [u_t(X_t) - ½ σ_t² s_t(X_t)] dt
```

**扩散模型 vs 流匹配**：

| 方法 | 学习目标 | 采样方式 |
|------|---------|---------|
| 流匹配 | u_t(X_t) | ODE（确定性） |
| 扩散模型 | s_t(X_t) | SDE（随机） |
| DDIM | 隐式分数 | ODE（确定性） |

**等价性**：
给定相同的路径，两种方法学习的向量场是等价的（最多相差一个与数据相关的项）。
