# MIT 6.S184: Generative AI With Stochastic Differential Equations

**MIT CSAIL, 2026**

---

## 这是什么课程？

这是MIT CSAIL开设的关于生成式AI的研究生课程讲义，从**随机微分方程（SDE）**和**常微分方程（ODE）**的角度全面介绍现代生成模型。

课程的独特之处在于：
- 统一的数学框架：所有生成模型都视为微分方程的模拟
- 理论与实践结合：从理论推导到PyTorch实现
- 前沿内容：包括Flow Matching、Consistency Models等最新进展

---

## 课程核心内容

### 第一部分：基础

1. **生成模型概述**
   - 从简单分布到复杂分布的变换
   - ODE vs SDE

2. **流匹配（Flow Matching）**
   - 向量场的学习
   - 概率密度演化（连续性方程）
   - 回归损失与流匹配损失的等价性

### 第二部分：扩散模型

3. **扩散模型基础**
   - 前向过程（加噪）
   - 反向过程（去噪）
   - Euler-Maruyama模拟方法

4. **ODE视角**
   - 概率流ODE
   - DDIM采样
   - 确定性采样路径

### 第三部分：高级主题

5. **最优传输（Optimal Transport）**
   - Monge问题
   - Kantorovich松弛
   - Wasserstein距离

6. **Flow Straightening**
   - 直化流（Rectified Flow）
   - 多步蒸馏
   - Reflow过程

7. **Consistency Models**
   - 一致性映射
   - 一致性蒸馏
   - 单步生成

---

## 核心数学框架

### 统一视角

所有生成模型都可以视为微分方程：

```
ODE:  dX_t = u_t(X_t) dt
SDE:  dX_t = u_t(X_t) dt + σ_t dW_t
```

**关键洞察**：
- **ODE**（概率流）：确定性路径，可逆变换
- **SDE**（扩散）：随机路径，更好的模式覆盖

### 学习目标

对于流匹配：
```
min E[||u_θ(X_t, t) - u_t(X_t, t)||²]
```

对于扩散模型：
```
min E[||s_θ(X_t, t) - ∇ log p(X_t|X_0)||²]
```

### 关键算法

**算法1：从流模型采样（ODE）**
```
1. 初始化 X_0 ~ p_init
2. for i = 1 to n:
3.   X_{t+h} = X_t + h * u_θ(X_t, t)
4. return X_1
```

**算法2：从扩散模型采样（SDE）**
```
1. 初始化 X_0 ~ p_init
2. for i = 1 to n:
3.   采样 ε ~ N(0, I)
4.   X_{t+h} = X_t + h * u_θ(X_t, t) + σ_t * √h * ε
5. return X_1
```

---

## 难度级别

**中高级** — 需要：
- 微积分（ODE/SDE基础）
- 概率论（高斯分布、条件概率）
- 深度学习（PyTorch、神经网络训练）
- 线性代数

---

## 如何阅读本资料

**推荐路径**：

1. **快速入门**：阅读本文档（概览）
2. **理论基础**：查看 `flow-matching.md`（流匹配详解）
3. **核心算法**：查看 `diffusion-models.md`（扩散模型）
4. **实现细节**：查看 `code/` 目录的示例
5. **前沿主题**：查看 `advanced-topics.md`（Flow Straightening、Consistency Models）

---

## 与其他资源的关系

| 资源 | 侧重点 | 与本课程的关系 |
|------|--------|---------------|
| **Sohl-Dickstein et al. (2015)** | 扩散模型原始论文 | 理论基础 |
| **Ho et al. (2020)** DDPM | 图像生成 | 实践应用 |
| **Song et al. (2021)** Score SDE | 分数匹配 | 数学等价 |
| **Lipman et al. (2023)** Flow Matching | 流匹配 | 本课程核心 |
| **Liu et al. (2023)** InstaFlow | 蒸馏 | 高级主题 |

---

## 核心代码示例

### PyTorch 实现流模型

```python
import torch
import torch.nn as nn

class FlowModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 256),  # +1 for time
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x, t):
        # x: (batch, dim)
        # t: (batch, 1)
        return self.net(torch.cat([x, t], dim=-1))

    def sample(self, n_samples, n_steps=100):
        x = torch.randn(n_samples, self.dim)  # p_init = N(0, I)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t = torch.ones(n_samples, 1) * i * dt
            x = x + dt * self(x, t)

        return x
```

---

## 学习资源

- **课程主页**: https://diffusion.csail.mit.edu/
- **讲义PDF**: https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf
- **实验代码**: 课程配套GitHub仓库

---

## 引用

```bibtex
@misc{mit2026diffusion,
  title={MIT 6.S184: Generative AI With Stochastic Differential Equations},
  author={MIT CSAIL},
  year={2026},
  howpublished={\url{https://diffusion.csail.mit.edu/}}
}
```
