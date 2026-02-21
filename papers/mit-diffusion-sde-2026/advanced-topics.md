# 高级主题

---

## 1. Flow Straightening（流直化）

### 1.1 问题动机

**标准流模型的路径可能是弯曲的**：
- 需要多步采样才能获得高质量样本
- 采样速度慢

**理想情况**：直线路径
- 单步或少数几步即可从 X_0 到达 X_1

### 1.2 Reflow 过程

**算法**：

```
初始化：训练初始流模型 u^(0)

对于 k = 1, 2, ..., K：
    1. 生成数据：
       - 采样 X_0 ~ N(0,I)
       - 通过 u^(k-1) 模拟得到 X_1
       - 得到配对 (X_0, X_1)

    2. 训练新流模型 u^(k)：
       - 在配对数据上使用 Rectified Flow
       - 最小化流匹配损失

输出：最终流模型 u^(K)
```

### 1.3 理论保证

**定理**（Liu et al., 2023）：

随着 Reflow 步数 K → ∞，流路径收敛到直线。

**最优传输解释**：

直化后的流对应于最优传输映射：
```
T* = arg min E[||X - T(X)||²]
```

### 1.4 实现代码

```python
class ReflowTrainer:
    def __init__(self, base_model):
        self.model = base_model
        self.models = [base_model]

    def generate_pairs(self, model, n_samples=10000):
        """使用当前模型生成 (X_0, X_1) 配对"""
        X_0 = torch.randn(n_samples, self.dim)
        X_1 = model.sample(n_samples)
        return X_0, X_1

    def train_reflow(self, n_reflow_steps=3):
        for k in range(n_reflow_steps):
            # 生成配对
            X_0, X_1 = self.generate_pairs(self.models[-1])

            # 在新数据上训练
            new_model = FlowModel(self.dim)
            train_flow_matching(new_model, X_0, X_1)

            self.models.append(new_model)

        return self.models[-1]
```

---

## 2. Consistency Models（一致性模型）

### 2.1 核心思想

**定义一致性函数** f(x, t)：
```
f(X_t, t) = X_0  对所有 t ∈ [0,1]
```

**关键性质**：
- 一步即可从任意 X_t 恢复 X_0
- 绕过多次采样步骤

### 2.2 一致性蒸馏

**从预训练扩散模型蒸馏**：

```
L = E[||f_θ(X_{t+Δt}, t+Δt) - f_θ(X_t, t)||²]
```

其中 X_t 和 X_{t+Δt} 来自ODE轨迹。

### 2.3 一致性训练

**直接训练（无需预训练模型）**：

```
L = E[λ(t) ||f_θ(X_t, t) - f_θ(X_{t+Δt}, t+Δt)||²]
```

其中 λ(t) 是加权函数。

### 2.4 多步采样

虽然一致性模型设计为单步，但多步采样可以提高质量：

```python
def multi_step_sampling(model, n_steps=4):
    x = torch.randn(batch_size, dim)

    # 分多步，每步用一致性模型
    for i in range(n_steps):
        t = i / n_steps
        x = model(x, t)
        # 加少量噪声（可选）
        x = x + 0.1 * torch.randn_like(x)

    return x
```

---

## 3. 最优传输（Optimal Transport）

### 3.1 Monge 问题

寻找最优传输映射 T：
```
min E[c(X, T(X))]
s.t. T#μ = ν
```

其中 c(x,y) 是代价函数（通常是 ||x-y||²）。

### 3.2 Kantorovich 松弛

允许"分割"传输：
```
min E_{π}[c(X,Y)]
s.t. π 的边缘是 μ 和 ν
```

### 3.3 Wasserstein 距离

```
W_p(μ, ν) = (min E[||X - Y||^p])^{1/p}
```

### 3.4 与生成模型的联系

**流模型学习传输映射**：
```
X_1 = T(X_0) = X_0 + ∫ u_t(X_t) dt
```

**Reflow 收敛到最优传输**：

多步 Reflow 后的流近似最优传输映射。

### 3.5 Sinkhorn 算法

**熵正则化最优传输**：
```
min E[c(X,Y)] + ε H(π)
```

其中 H(π) 是熵，ε 是正则化参数。

**迭代算法**：
```python
def sinkhorn(a, b, C, epsilon, n_iter=100):
    """
    a: 源分布 (n,)
    b: 目标分布 (m,)
    C: 代价矩阵 (n,m)
    epsilon: 正则化参数
    """
    K = np.exp(-C / epsilon)
    u = np.ones_like(a)

    for _ in range(n_iter):
        v = b / (K.T @ u)
        u = a / (K @ v)

    P = np.diag(u) @ K @ np.diag(v)
    return P
```

---

## 4. 引导采样技术

### 4.1 Classifier Guidance

**使用分类器引导**：
```
∇_x log p(x|y) = ∇_x log p(x) + ∇_x log p(y|x)
```

**实现**：
```python
def classifier_guidance(x, t, classifier, y, scale=7.5):
    # 分数估计
    score = score_model(x, t)

    # 分类器梯度
    with torch.enable_grad():
        x_in = x.detach().requires_grad_(True)
        log_prob = classifier(x_in, t, y)
        grad = torch.autograd.grad(log_prob.sum(), x_in)[0]

    # 引导分数
    guided_score = score + scale * grad

    return guided_score
```

### 4.2 Classifier-Free Guidance (CFG)

**无需预训练分类器**：

训练时随机丢弃条件：
```python
if random.random() < 0.1:
    y = None  # 无条件
```

采样时：
```python
def cfg_sampling(x, t, y, scale=7.5):
    # 无条件预测
    eps_uncond = model(x, t, y=None)

    # 有条件预测
    eps_cond = model(x, t, y=y)

    # 外推
    eps = eps_uncond + scale * (eps_cond - eps_uncond)

    return eps
```

### 4.3 负提示（Negative Prompting）

**指定不想要的属性**：
```python
def negative_prompt_sampling(x, t, positive_prompt, negative_prompt):
    # 正向引导
    eps_pos = cfg_sampling(x, t, positive_prompt, scale=7.5)

    # 负向引导
    eps_neg = cfg_sampling(x, t, negative_prompt, scale=7.5)

    # 结合
    eps = eps_pos - 0.5 * eps_neg

    return eps
```

---

## 5. 高效采样技术

### 5.1 步数蒸馏

**Progressive Distillation**：
```
教师模型：N 步采样
学生模型：N/2 步采样
```

迭代直到单步。

### 5.2 高阶求解器

**DPM-Solver**：
```python
def dpm_solver_step(x, t, s, model):
    """
    高阶ODE求解器
    比欧拉方法更快收敛
    """
    # 一阶
    eps_1 = model(x, t)

    # 二阶（中间步骤）
    t_mid = (t + s) / 2
    x_mid = x + (s - t) / 2 * eps_1
    eps_2 = model(x_mid, t_mid)

    # 更新
    x = x + (s - t) * ((1 + 1/(2r)) * eps_1 - 1/(2r) * eps_2)

    return x
```

### 5.3 自适应步长

根据局部曲率调整步长：
```python
def adaptive_step(x, t, model, tol=1e-3):
    # 尝试大步长
    x_large = euler_step(x, t, t+dt, model)

    # 尝试两步小步长
    x_small = euler_step(x, t, t+dt/2, model)
    x_small = euler_step(x_small, t+dt/2, t+dt, model)

    # 误差估计
    error = torch.norm(x_large - x_small)

    if error < tol:
        return x_large, dt * 2  # 接受并增大步长
    else:
        return x_small, dt / 2  # 拒绝并减小步长
```

---

## 6. 应用扩展

### 6.1 条件生成

**类别条件**：
```python
# 在输入中加入类别嵌入
x = torch.cat([x, class_emb], dim=-1)
```

**文本条件**（CLIP引导）：
```python
text_emb = clip.encode_text(prompt)
image_emb = clip.encode_image(x)
similarity = cosine_similarity(text_emb, image_emb)
```

### 6.2 图像编辑

**Inpainting**：
```python
def inpaint(x_known, mask, model):
    """
    mask: 1 表示已知，0 表示需要生成
    """
    x = torch.randn_like(x_known)

    for t in reversed(range(n_steps)):
        # 去噪
        x = denoise_step(x, t, model)

        # 替换已知区域
        x = x_known * mask + x * (1 - mask)

    return x
```

**Style Transfer**：
```python
# 内容损失
content_loss = ||features(x) - features(content)||²

# 风格损失
style_loss = ||gram(features(x)) - gram(features(style))||²
```

### 6.3 3D生成

**DreamFusion**：
- 使用2D扩散模型生成3D
- Score Distillation Sampling (SDS)

```python
def sds_loss(nerf_model, diffusion_model, text_emb):
    # 渲染图像
    image = nerf.render(camera_pose)

    # 加噪
    t = random.randint(0, T)
    noisy_image = add_noise(image, t)

    # 扩散模型预测
    pred = diffusion_model(noisy_image, t, text_emb)

    # 梯度更新NeRF
    loss = ||pred - noise||²

    return loss
```

---

## 7. 理论前沿

### 7.1 收敛性分析

**分数匹配的收敛**：
- 分数估计误差 → 采样质量
- Wasserstein 距离边界

### 7.2 泛化性

**扩散模型的泛化**：
- 训练分布外的样本
- 对抗鲁棒性

### 7.3 混合模型

**组合多个扩散模型**：
```
p(x) = p_1(x)^α * p_2(x)^(1-α) / Z
```

实现产品专家（Product of Experts）。

---

## 8. 实践建议

### 8.1 模型选择

| 场景 | 推荐模型 |
|------|---------|
| 最高质量 | 多步扩散模型 + CFG |
| 速度优先 | 一致性模型 |
| 可逆性需要 | ODE流模型 |
| 可控生成 | 条件扩散模型 |

### 8.2 超参数调优

**关键参数**：
- 学习率：1e-4 ~ 1e-3
- 批次大小：越大越好（考虑内存）
- 时间步数：1000（训练），50（采样）
- 噪声调度：Cosine > Linear

### 8.3 调试技巧

**检查清单**：
- [ ] 损失是否下降？
- [ ] 采样轨迹是否合理？
- [ ] 最终样本是否有模式坍塌？
- [ ] 不同种子的一致性？

**常见问题**：
- 训练不稳定：降低学习率
- 模式坍塌：增加多样性损失
- 采样噪声：增加采样步数
