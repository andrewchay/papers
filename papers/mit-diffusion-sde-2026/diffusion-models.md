# 扩散模型详解

---

## 1. 扩散模型概述

### 1.1 前向过程（加噪）

逐渐向数据添加高斯噪声：
```
q(x_t | x_0) = N(x_t; α_t x_0, σ_t² I)
```

**方差调度（Variance Schedule）**：
- Linear: σ_t² = t
- Cosine: 平滑过渡
- VP (Variance Preserving): σ_t² = 1 - α_t²

### 1.2 反向过程（去噪）

从纯噪声逐渐去噪：
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

**关键问题**：如何学习 μ_θ 和 Σ_θ？

---

## 2. 三种等价视角

### 2.1 分数视角（Score Matching）

**分数函数**：
```
s(x_t, t) = ∇_{x_t} log p(x_t)
```

**去噪分数匹配**：
```
∇_x log p(x_t | x_0) = -(x_t - α_t x_0) / σ_t²
```

**学习目标**：
```
min E[||s_θ(x_t, t) - ∇_x log p(x_t | x_0)||²]
```

### 2.2 噪声视角（Noise Prediction）

**重参数化**：
```
x_t = α_t x_0 + σ_t ε,  ε ~ N(0, I)
```

**预测噪声**：
```
ε_θ(x_t, t) ≈ ε
```

**损失函数**（DDPM）：
```
L = E[||ε - ε_θ(α_t x_0 + σ_t ε, t)||²]
```

### 2.3 ODE视角（Probability Flow）

**概率流ODE**：
```
dx_t = [u_t(x_t) - ½ σ_t² ∇ log p_t(x_t)] dt
```

**与分数的关系**：
```
dx_t = -½ σ_t² s_θ(x_t, t) dt
```

---

## 3. 采样算法

### 3.1 DDPM采样

```python
def ddpm_sample(model, n_samples, n_steps=1000):
    x = torch.randn(n_samples, dim)

    for t in reversed(range(n_steps)):
        # 预测噪声
        eps = model(x, t)

        # 计算均值
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]

        x_0_pred = (x - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        mean = sqrt(alpha_{t-1}) * (1 - alpha_t) / (1 - alpha_bar_t) * x_0_pred + \
               sqrt(alpha_t) * (1 - alpha_{t-1}) / (1 - alpha_bar_t) * x

        # 添加噪声（除了最后一步）
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + sigma_t * noise
        else:
            x = mean

    return x
```

### 3.2 DDIM采样（确定性）

```python
def ddim_sample(model, n_samples, n_steps=50):
    x = torch.randn(n_samples, dim)

    for t in reversed(range(n_steps)):
        # 预测噪声
        eps = model(x, t)

        # 预测 x_0
        alpha_bar_t = alpha_bars[t]
        x_0 = (x - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)

        # 确定性更新
        if t > 0:
            alpha_bar_prev = alpha_bars[t-1]
            x = sqrt(alpha_bar_prev) * x_0 + sqrt(1 - alpha_bar_prev) * eps
        else:
            x = x_0

    return x
```

### 3.3 Euler-Maruyama（SDE）

```python
def sde_sample(model, n_samples, n_steps=1000):
    x = torch.randn(n_samples, dim)
    dt = 1.0 / n_steps

    for t in range(n_steps):
        # 分数估计
        score = model(x, t)

        # 漂移项 + 扩散项
        drift = -0.5 * sigma_t**2 * score
        diffusion = sigma_t * sqrt(dt) * torch.randn_like(x)

        x = x + drift * dt + diffusion

    return x
```

---

## 4. 训练目标详解

### 4.1 简化损失（DDPM）

```
L_simple = E_{t,x_0,ε}[||ε - ε_θ(x_t, t)||²]
```

**优点**：
- 简单有效
- 不需要学习方差

### 4.2 加权损失

```
L_λ = E[λ_t ||ε - ε_θ(x_t, t)||²]
```

**加权策略**：
- Uniform: λ_t = 1
- SNR-weighted: λ_t = SNR(t)
- Cosine: 平滑加权

### 4.3 与流匹配的关系

**等价性**：

如果定义向量场：
```
u_t(x) = (ẋ_t(x) - σ̇_t ε) / ...
```

则流匹配损失和噪声预测损失等价（最多相差常数）。

---

## 5. 实现要点

### 5.1 U-Net架构

```python
class UNet(nn.Module):
    def __init__(self, in_channels, time_emb_dim=256):
        super().__init__()
        # 时间编码
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # 下采样
        self.down = nn.ModuleList([
            DownBlock(in_channels, 64),
            DownBlock(64, 128),
            DownBlock(128, 256)
        ])

        # 中间
        self.mid = MidBlock(256)

        # 上采样
        self.up = nn.ModuleList([
            UpBlock(256 + 128, 128),
            UpBlock(128 + 64, 64),
            UpBlock(64 + in_channels, in_channels)
        ])

    def forward(self, x, t):
        # 时间编码
        t_emb = self.time_embed(t)

        # 下采样
        skips = []
        for block in self.down:
            x = block(x, t_emb)
            skips.append(x)

        # 中间
        x = self.mid(x, t_emb)

        # 上采样
        for block in self.up:
            x = block(torch.cat([x, skips.pop()], dim=1), t_emb)

        return x
```

### 5.2 时间步采样

```python
def sample_timesteps(batch_size, n_steps=1000):
    """均匀采样或重要性采样"""
    return torch.randint(0, n_steps, (batch_size,))
```

### 5.3 噪声调度

```python
def cosine_schedule(timesteps, s=0.008):
    """Cosine noise schedule"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)
```

---

## 6. 高级主题

### 6.1 Classifier Guidance

**条件生成**：
```
∇_x log p(x|y) = ∇_x log p(x) + ∇_x log p(y|x)
```

**Classifier-Free Guidance (CFG)**：
```
ε_θ(x_t, t, y) = ε_θ(x_t, t, ∅) + w * (ε_θ(x_t, t, y) - ε_θ(x_t, t, ∅))
```

其中 w > 1 增强条件效果。

### 6.2 引导采样

```python
def cfg_sample(model, n_samples, label, guidance_scale=7.5):
    x = torch.randn(n_samples, dim)

    for t in reversed(range(n_steps)):
        # 无条件预测
        eps_uncond = model(x, t, label=None)

        # 有条件预测
        eps_cond = model(x, t, label=label)

        # CFG
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # DDIM更新
        x = ddim_step(x, eps, t)

    return x
```

---

## 7. 总结

### 扩散模型 = 流匹配 + 随机噪声

| 特性 | 流匹配 (ODE) | 扩散模型 (SDE) |
|------|-------------|---------------|
| 训练 | 回归向量场 | 预测噪声/分数 |
| 采样 | 确定性 | 随机 |
| 可逆性 | 是 | 否（但有ODE对应） |
| 多样性 | 较低 | 较高 |
| 速度 | 快（少步数） | 慢（多步数） |

### 关键公式对照

```
流匹配:      u_t(x) = E[Ẋ_t | X_t = x]
扩散:        ε_θ(x, t) ≈ ε
分数:        s_θ(x, t) ≈ ∇ log p(x)
关系:        u = -σ² s = -σ² (x - αx_0)/σ² = (αx_0 - x)/σ
```
