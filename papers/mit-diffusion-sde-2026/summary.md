# 课程摘要：MIT 6.S184 生成式AI

---

## 课程概览

**MIT 6.S184: Generative AI With Stochastic Differential Equations**

这是一门将现代生成模型（扩散模型、流匹配、一致性模型）统一在**微分方程**框架下的研究生课程。

---

## 核心贡献

### 统一框架

所有生成模型都可以视为微分方程的模拟：

```
ODE（流模型）:    dX_t = u_t(X_t) dt
SDE（扩散模型）:  dX_t = u_t(X_t) dt + σ_t dW_t
```

**关键洞察**：
- ODE 提供确定性、可逆的变换
- SDE 提供随机性、更好的模式覆盖
- 两者共享相同的边缘分布 p_t(X_t)

### 理论-实践结合

| 理论概念 | 实践实现 |
|---------|---------|
| 连续性方程 | 神经网络参数化 |
| 向量场学习 | 流匹配损失 |
| 分数函数 | 噪声预测网络 |
| 概率流ODE | DDIM确定性采样 |

---

## 关键技术

### 1. Flow Matching（流匹配）

**核心思想**：直接学习将 p_init 转换为 p_data 的向量场 u_t。

**损失函数**：
```
L = E[||u_θ(X_t, t) - u_t(X_t)||²]
```

**关键创新**：条件流匹配损失，使得我们可以在不知道真实 u_t 的情况下训练模型。

**Rectified Flow（线性插值）**：
```
X_t = (1-t) X_0 + t X_1
u_t = X_1 - X_0
```

### 2. 扩散模型

**前向过程**（加噪）：
```
X_t = α_t X_0 + σ_t ε,  ε ~ N(0,I)
```

**反向过程**（去噪）：
```
p(X_{t-1}|X_t) = N(μ_θ(X_t, t), Σ_θ(X_t, t))
```

**三种等价视角**：
1. **噪声预测**：预测 ε
2. **分数匹配**：预测 ∇ log p(X_t)
3. **ODE视角**：概率流ODE

### 3. Flow Straightening

**问题**：标准流路径可能是弯曲的，需要多步采样。

**解决方案**：Reflow过程

```
Step 1: 训练初始流模型 u^(1)
Step 2: 生成配对数据 (X_0, X_1) 通过 u^(1)
Step 3: 在新数据上训练 u^(2)
Result: 更直的流路径
```

**理论保证**：多步Reflow后，流路径收敛到直线（最优传输）。

### 4. Consistency Models（一致性模型）

**核心思想**：学习一致性函数 f，使得：
```
f(X_t, t) = f(X_{t'}, t') = X_0  对所有 t, t'
```

**优点**：单步生成（无需多步采样）。

**训练**：一致性蒸馏或一致性训练。

---

## 与最优传输的联系

### Monge问题

寻找最优传输映射 T：
```
min E[||X - T(X)||²]
s.t. T#p_init = p_data
```

### 与流模型的关系

Rectified Flow 的极限（多步Reflow后）收敛到最优传输映射。

**实际意义**：
- 更直的流路径
- 更少的采样步数
- 更好的插值效果

---

## 实验结果

### 图像生成

- **CIFAR-10**：FID < 3.0（单步一致性模型）
- **ImageNet 64x64**：高质量样本

### 文本到图像

- 基于流的Stable Diffusion变体
- 更快采样（4-8步）

---

## 学习路径

### 基础知识
- [x] ODE/SDE基础
- [x] 概率论（条件概率、边缘分布）
- [x] 深度学习（PyTorch、神经网络）

### 核心内容
1. **流匹配**（第2-3章）
   - 连续性方程
   - 条件流匹配
   - Rectified Flow

2. **扩散模型**（第4-5章）
   - 前向/反向过程
   - DDPM/DDIM
   - 分数匹配

3. **高级主题**（第6-8章）
   - Flow Straightening
   - Consistency Models
   - 最优传输

### 实践
- 实现Flow Matching
- 训练扩散模型
- 蒸馏一致性模型

---

## 关键公式速查

### 流匹配
```python
# 条件路径
X_t = (1-t) * X_0 + t * X_1

# 目标向量场
u_target = X_1 - X_0

# 损失
L = ||u_θ(X_t, t) - u_target||²
```

### 扩散模型
```python
# 前向
X_t = sqrt(alpha_bar_t) * X_0 + sqrt(1 - alpha_bar_t) * eps

# 预测
eps_pred = model(X_t, t)

# 损失
L = ||eps - eps_pred||²
```

### DDIM采样
```python
# 预测X_0
X_0_pred = (X_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)

# 确定性更新
X_{t-1} = sqrt(alpha_{t-1}) * X_0_pred + sqrt(1-alpha_{t-1}) * eps
```

---

## 相关论文

### 基础
- **Sohl-Dickstein et al. (2015)**: 扩散模型原始论文
- **Ho et al. (2020)** DDPM: 去噪扩散概率模型
- **Song et al. (2021)** Score SDE: 分数SDE

### 流匹配
- **Lipman et al. (2023)**: Flow Matching for Generative Modeling
- **Liu et al. (2023)**: Flow Straight and Fast
- **Albergo & Vanden-Eijnden (2023)**: Building Normalizing Flows

### 蒸馏
- **Salimans & Ho (2022)**: Progressive Distillation
- **Song et al. (2023)**: Consistency Models
- **Liu et al. (2023)**: InstaFlow

---

## 一句话总结

> 这门课程提供了一个统一的数学框架，将所有现代生成模型（扩散模型、流匹配、一致性模型）视为微分方程的模拟，从而加深理解并促进新方法的开发。
