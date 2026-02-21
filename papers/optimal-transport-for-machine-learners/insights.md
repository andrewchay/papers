# Optimal Transport 核心洞察

---

## 1. 核心思想：用几何方式比较概率分布

### 为什么传统方法不够？

**KL 散度**的问题：
```python
# 两个完全不重叠的分布
mu = Uniform([0, 0.5])
nu = Uniform([0.5, 1])

KL(mu || nu) = ∞  # 无穷大！
W_2(mu, nu) = 0.5  # 有意义的距离
```

**关键洞察**：OT 考虑分布之间的"最优移动成本"，而非仅仅是密度比值。

### 直观理解

想象你有两堆沙子：
- **Monge 问题**：每个沙粒从起点移动到终点的最优方案
- **Kantorovich 问题**：允许沙粒分裂和合并的最优方案

---

## 2. 为什么 Kantorovich 松弛如此重要？

### 数学原因

Monge 问题要求**确定性映射** $T$，这在离散情况下可能不存在：

```
μ = δ_{x1}  (单点质量)
ν = 0.5*δ_{y1} + 0.5*δ_{y2}  (两点)

不存在 T 使得 T#μ = ν！
```

Kantorovich 允许**概率耦合**，总是存在解。

### 计算原因

Kantorovich 形式化是一个**线性规划问题**：
- 凸优化：存在全局最优解
- 对偶形式：提供更高效的算法
- 可扩展性：熵正则化后可用 Sinkhorn 算法

---

## 3. Brenier 定理：凸性的力量

### 核心结论

对于二次成本，最优传输映射是某个**凸函数的梯度**：

$$T = \nabla\phi, \quad \phi \text{ 是凸函数}$$

### 为什么这很重要？

1. **存在性保证**：凸函数几乎处处可微
2. **唯一性**：凸性保证了唯一最优解
3. **计算可处理**：可以参数化为神经网络（输入凸网络）
4. **物理意义**：类比于流体动力学中的压力场

### 与深度学习的联系

**ICNN（Input Convex Neural Networks）**：
```python
class ICNN(nn.Module):
    """输入凸神经网络"""
    def forward(self, x):
        # 确保对输入 x 是凸的
        z = F.relu(self.W1 @ x + self.b1)
        z = F.relu(self.W2 @ z + self.b2)
        return self.final(z)

# 可以用来参数化最优传输映射
T = gradient(icnn)  # ∇φ
```

---

## 4. 对偶形式：从耦合到势函数

### Kantorovich 对偶

原始问题：在耦合空间上优化
对偶问题：在势函数空间上优化

$$W_c(\mu, \nu) = \sup_{\phi, \psi} \left\{ \mathbb{E}_\mu[\phi] + \mathbb{E}_\nu[\psi] \right\}$$

约束：$\phi(x) + \psi(y) \leq c(x,y)$

### 为什么对偶有用？

1. **降维**：从联合分布降到单变量函数
2. **计算效率**：某些情况下更容易求解
3. **理论洞察**：揭示最优传输的"价格"解释

### $c$-变换

定义：$\phi^c(y) = \inf_x c(x,y) - \phi(x)$

性质：最优势函数满足 $\psi = \phi^c$（对 $c$-变换封闭）

对于二次成本，这就是**Legendre-Fenchel 变换**！

---

## 5. 动态形式化：流体动力学视角

### Benamou-Brenier 公式

$$W_2(\mu, \nu)^2 = \inf_{\rho_t, v_t} \int_0^1 \int \|v_t(x)\|^2 \rho_t(x) dx dt$$

约束：连续性方程 $\partial_t \rho_t + \nabla \cdot (\rho_t v_t) = 0$

### 直观理解

Wasserstein 距离是沿着概率分布"流体"的最小动能路径。

类比：
- **静态 OT**：直接比较两个快照
- **动态 OT**：观察从一个分布到另一个分布的演化过程

### 与扩散模型的联系

扩散模型可以看作是在 Wasserstein 空间中的梯度流：

$$dX_t = -\nabla \log p_t(X_t) dt + \sqrt{2} dW_t$$

这近似于在 KL 散度下的梯度流，而 KL 散度在 Wasserstein 度量下的梯度流是 Fokker-Planck 方程！

---

## 6. 熵正则化：计算与统计的权衡

### Sinkhorn 算法

在 OT 目标中添加熵正则化：

$$\min_\pi \langle C, \pi \rangle + \varepsilon \text{KL}(\pi || \mu \otimes \nu)$$

### 为什么有效？

1. **严格凸**：保证唯一解
2. **闭式解**：解具有特定结构 $\pi^*_{ij} = a_i K_{ij} b_j$
3. **快速迭代**：Sinkhorn 算法 = 交替行列缩放

### 迭代公式

```python
def sinkhorn(K, a, b, epsilon, n_iter=100):
    """
    K: 核矩阵 K_{ij} = exp(-C_{ij}/epsilon)
    a, b: 边缘分布
    """
    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(n_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)

    return np.diag(u) @ K @ np.diag(v)
```

### 权衡

| 方法 | 计算复杂度 | 精度 | 平滑性 |
|------|-----------|------|--------|
| 精确 LP | $O(n^3)$ | 精确 | 不连续 |
| Sinkhorn | $O(n^2/\varepsilon)$ | $\varepsilon$-近似 | 平滑 |

---

## 7. 梯度流：在 Wasserstein 空间中优化

### 什么是 Wasserstein 梯度流？

给定泛函 $\mathcal{F}(\rho)$，其 Wasserstein 梯度流是：

$$\partial_t \rho + \nabla \cdot (\rho \nabla \frac{\delta \mathcal{F}}{\delta \rho}) = 0$$

### JKO 方案

离散时间近似：

$$\rho_{k+1} = \arg\min_\rho \mathcal{F}(\rho) + \frac{1}{2\tau} W_2(\rho, \rho_k)^2$$

这相当于在 Wasserstein 度量下的梯度下降！

### 在 ML 中的应用

1. **神经网络训练**：参数更新可以看作梯度流
2. **采样**：Langevin 动力学是 KL 散度的梯度流
3. **生成模型**：扩散模型是 score matching 的梯度流

---

## 8. Bures-Wasserstein：高斯世界的解析解

### 为什么高斯分布特殊？

对于高斯分布，Wasserstein-2 距离有**闭式解**：

$$B(\mu, \nu)^2 = \|m_\mu - m_\nu\|^2 + \text{tr}(\Sigma_\mu) + \text{tr}(\Sigma_\nu) - 2\text{tr}\left(\Sigma_\mu^{1/2}\Sigma_\nu\Sigma_\mu^{1/2}\right)^{1/2}$$

### 几何解释

- **均值部分**：欧氏距离
- **协方差部分**：类似于矩阵的"距离"

这定义了高斯分布上的一个**黎曼度量**。

### 应用

1. **高斯混合模型**：简化计算
2. **变分推断**：Wasserstein VI
3. **统计流形**：信息几何

---

## 9. 概念转变：从离散到连续再到离散

### 历史发展

1. **Monge (1781)**：连续问题，确定性映射
2. **Kantorovich (1942)**：松弛到耦合，线性规划
3. **现代 ML**：大规模离散问题，熵正则化

### 当前趋势

- **神经 OT**：用神经网络参数化映射
- **随机插值**：扩散模型与 OT 的结合
- **广义 OT**：非平衡 OT、Gromov-Wasserstein

---

## 10. 实用建议

### 何时使用 OT？

✅ **适合**：
- 生成分布与真实分布的比较
- 分布间的插值/形变
- 需要几何意义的距离
- 分布是低维的或具有结构

❌ **不适合**：
- 高维分布（维度诅咒）
- 只需要快速比较，不关心几何
- 样本量极大（考虑近似方法）

### 选择哪种方法？

| 场景 | 推荐方法 |
|------|----------|
| 小规模精确计算 | 线性规划 |
| 中等规模快速近似 | Sinkhorn |
| 大规模/在线 | 随机 Sinkhorn、神经 OT |
| 高维 | 切片 Wasserstein |
| 不同维度 | Gromov-Wasserstein |

---

## 11. 开放问题

1. **高维扩展性**：如何有效处理高维分布？
2. **采样复杂度**：需要多少样本才能准确估计 Wasserstein 距离？
3. **神经 OT**：神经网络能否学习到最优传输映射？
4. **扩散模型与 OT**：两者的精确关系是什么？
5. **非平衡 OT**：如何处理质量不守恒的情况？

---

## 12. 关键公式速查

### Wasserstein-p 距离
$$W_p(\mu, \nu) = \left(\inf_{\pi \in \Pi(\mu, \nu)} \int d(x,y)^p d\pi\right)^{1/p}$$

### Kantorovich 对偶
$$W_1(\mu, \nu) = \sup_{\|f\|_L \leq 1} \mathbb{E}_\mu[f] - \mathbb{E}_\nu[f]$$

### Brenier 势函数
$$T = \nabla\phi, \quad \phi \text{ 凸}$$

### Sinkhorn 目标
$$\mathcal{L}_\varepsilon(\pi) = \langle C, \pi \rangle + \varepsilon \text{KL}(\pi || \mu \otimes \nu)$$

### 梯度流
$$\partial_t \rho = \nabla \cdot \left(\rho \nabla \frac{\delta \mathcal{F}}{\delta \rho}\right)$$
