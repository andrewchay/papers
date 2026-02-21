# Optimal Transport 方法详解

---

## 1. 问题形式化

### 1.1 Monge 问题（1781）

**设置**：
- 源测度 $\mu \in \mathcal{P}(X)$
- 目标测度 $\nu \in \mathcal{P}(Y)$
- 成本函数 $c: X \times Y \to \mathbb{R}_+$

**优化问题**：
$$\inf_{T} \left\{ \int_X c(x, T(x)) d\mu(x) : T_\#\mu = \nu \right\}$$

其中 $T_\#\mu = \nu$ 表示推前测度：$\nu(A) = \mu(T^{-1}(A))$

**离散形式**（点云匹配）：

给定两个点云 $\{x_i\}_{i=1}^n$ 和 $\{y_j\}_{j=1}^m$，成本矩阵 $C_{ij} = c(x_i, y_j)$：

$$\min_{\sigma \in \text{Perm}(n)} \frac{1}{n}\sum_{i=1}^n C_{i,\sigma(i)}$$

其中 $\sigma$ 是排列（假设 $n=m$）。

**问题**：
- 当 $n \neq m$ 时无定义
- 非凸优化（排列空间是离散的）
- 可能不存在解

---

### 1.2 Kantorovich 松弛（1942）

**关键思想**：允许质量分裂

**优化问题**：
$$\inf_{\pi \in \Pi(\mu, \nu)} \int_{X \times Y} c(x,y) d\pi(x,y)$$

其中耦合集合：
$$\Pi(\mu, \nu) = \left\{ \pi \in \mathcal{P}(X \times Y) : \pi(A \times Y) = \mu(A), \pi(X \times B) = \nu(B) \right\}$$

**离散形式**：

$$\min_{\pi \in \mathbb{R}_+^{n \times m}} \sum_{i,j} C_{ij} \pi_{ij}$$

约束：
- $\sum_j \pi_{ij} = \mu_i$ （行和）
- $\sum_i \pi_{ij} = \nu_j$ （列和）

**优势**：
- 线性规划（凸优化）
- 总是存在解
- 对偶理论可用

---

## 2. 对偶形式

### 2.1 Kantorovich 对偶

**定理**：
$$W_c(\mu, \nu) = \sup_{\phi, \psi} \left\{ \int_X \phi d\mu + \int_Y \psi d\nu \right\}$$

约束：$\phi(x) + \psi(y) \leq c(x,y)$

**离散形式**：

$$\max_{u \in \mathbb{R}^n, v \in \mathbb{R}^m} \sum_i u_i \mu_i + \sum_j v_j \nu_j$$

约束：$u_i + v_j \leq C_{ij}$

### 2.2 $c$-变换

定义：
$$\phi^c(y) = \inf_x c(x,y) - \phi(x)$$

**性质**：
- 最优势函数满足 $\psi = \phi^c$
- 对二次成本，这就是 Legendre-Fenchel 变换

**$W_1$ 特殊情况**（Kantorovich-Rubinstein）：

$$W_1(\mu, \nu) = \sup_{\|f\|_L \leq 1} \mathbb{E}_\mu[f] - \mathbb{E}_\nu[f]$$

其中 $\|f\|_L$ 是 Lipschitz 常数。

---

## 3. 数值方法

### 3.1 线性规划求解

**网络单纯形法**：
- 复杂度：$O(n^3)$
- 精确解
- 适合小规模问题

**Python 实现**（使用 POT 库）：
```python
import ot

# 定义分布和成本矩阵
mu = np.array([0.5, 0.5])  # 源分布
nu = np.array([0.3, 0.7])  # 目标分布
C = np.array([[1, 2],      # 成本矩阵
              [3, 1]])

# 精确 LP 求解
pi = ot.emd(mu, nu, C)
W = np.sum(pi * C)
```

### 3.2 Sinkhorn 算法（熵正则化）

**正则化目标**：
$$\min_\pi \langle C, \pi \rangle + \varepsilon \text{KL}(\pi | \mu \otimes \nu)$$

**解的结构**：
$$\pi^*_{ij} = u_i K_{ij} v_j$$

其中 $K_{ij} = \exp(-C_{ij}/\varepsilon)$

**Sinkhorn 迭代**：

```python
def sinkhorn(K, mu, nu, epsilon, max_iter=1000, tol=1e-6):
    """
    Sinkhorn-Knopp 算法

    Parameters:
    -----------
    K : array (n, m)
        核矩阵 K_{ij} = exp(-C_{ij}/epsilon)
    mu : array (n,)
        源分布
    nu : array (m,)
        目标分布
    epsilon : float
        正则化参数
    """
    n, m = K.shape
    u = np.ones(n)
    v = np.ones(m)

    for iteration in range(max_iter):
        u_prev = u.copy()

        # 更新 u 和 v
        u = mu / (K @ v)
        v = nu / (K.T @ u)

        # 检查收敛
        if np.max(np.abs(u - u_prev)) < tol:
            break

    # 重构耦合
    pi = np.diag(u) @ K @ np.diag(v)
    return pi

# 使用示例
epsilon = 0.1
K = np.exp(-C / epsilon)
pi_sinkhorn = sinkhorn(K, mu, nu, epsilon)
```

**复杂度分析**：

| 方面 | 复杂度 |
|------|--------|
| 每次迭代 | $O(n^2)$ |
| 迭代次数 | $O(\|C\|_\infty / \varepsilon)$ |
| 总复杂度 | $O(n^2 \log n / \varepsilon)$ |
| 存储 | $O(n^2)$ |

**实际考虑**：
- 数值稳定性：使用 log-sum-exp 技巧
- 并行化：行列缩放高度并行
- GPU 加速：矩阵运算适合 GPU

---

## 4. 动态形式化

### 4.1 Benamou-Brenier 公式

**定理**：
$$W_2(\mu, \nu)^2 = \inf_{(\rho_t, v_t)} \int_0^1 \int \|v_t(x)\|^2 \rho_t(x) dx dt$$

约束：
- 连续性方程：$\partial_t \rho_t + \nabla \cdot (\rho_t v_t) = 0$
- 边界条件：$\rho_0 = \mu$, $\rho_1 = \nu$

**直观**：寻找从 $\mu$ 到 $\nu$ 的最小"动能"路径。

### 4.2 与静态 OT 的联系

**McCann 插值**：
$$\rho_t = ((1-t)\text{Id} + tT)_\#\mu$$

其中 $T$ 是最优传输映射。

对于二次成本，这是 Wasserstein 空间中的**测地线**。

---

## 5. 梯度流

### 5.1 Wasserstein 梯度流

**定义**：给定泛函 $\mathcal{F}(\rho)$，其 Wasserstein 梯度流满足：

$$\partial_t \rho = \nabla \cdot \left( \rho \nabla \frac{\delta \mathcal{F}}{\delta \rho} \right)$$

**关键例子**：

| 泛函 | 变分导数 | 梯度流 |
|------|----------|--------|
| $\int \rho \log \rho$ | $\log \rho + 1$ | 热方程 |
| $\text{KL}(\rho \| \pi)$ | $\log \rho - \log \pi$ | Fokker-Planck |
| $\frac{1}{2}W_2(\rho, \nu)^2$ | $\phi$ （Kantorovich 势） | 测度平移 |

### 5.2 JKO 方案

**离散时间近似**：

$$\rho_{k+1} = \arg\min_\rho \mathcal{F}(\rho) + \frac{1}{2\tau} W_2(\rho, \rho_k)^2$$

**解释**：
- 第一项：降低能量
- 第二项：保持在 Wasserstein 度量下"接近"前一个状态
- 步长 $\tau$ 控制稳定性

**Python 实现**（概念）：
```python
def jko_step(rho_k, F, tau):
    """
    一步 JKO 方案

    Parameters:
    -----------
    rho_k : 当前分布
    F : 能量泛函
    tau : 时间步长
    """
    def objective(rho):
        return F(rho) + (1/(2*tau)) * wasserstein_distance(rho, rho_k)**2

    rho_next = optimize(objective, init=rho_k)
    return rho_next
```

---

## 6. 高斯分布上的解析解

### 6.1 Bures-Wasserstein 距离

**定理**：对于高斯分布 $\mu = \mathcal{N}(m_\mu, \Sigma_\mu)$, $\nu = \mathcal{N}(m_\nu, \Sigma_\nu)$：

$$W_2(\mu, \nu)^2 = \|m_\mu - m_\nu\|^2 + B(\Sigma_\mu, \Sigma_\nu)^2$$

其中 Bures 距离：

$$B(\Sigma_\mu, \Sigma_\nu)^2 = \text{tr}(\Sigma_\mu) + \text{tr}(\Sigma_\nu) - 2\text{tr}\left(\Sigma_\mu^{1/2}\Sigma_\nu\Sigma_\mu^{1/2}\right)^{1/2}$$

**最优映射**：
$$T(x) = m_\nu + A(x - m_\mu)$$

其中 $A = \Sigma_\mu^{-1/2}(\Sigma_\mu^{1/2}\Sigma_\nu\Sigma_\mu^{1/2})^{1/2}\Sigma_\mu^{-1/2}$

**Python 实现**：
```python
def bures_wasserstein(mu1, sigma1, mu2, sigma2):
    """
    计算两个高斯分布之间的 Bures-Wasserstein 距离

    Parameters:
    -----------
    mu1, mu2 : array (d,)
        均值
    sigma1, sigma2 : array (d, d)
        协方差矩阵

    Returns:
    --------
    distance : float
        W_2 距离
    """
    # 均值部分
    mean_diff = np.linalg.norm(mu1 - mu2)**2

    # 协方差部分
    sqrt_sigma1 = scipy.linalg.sqrtm(sigma1)
    product = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    sqrt_product = scipy.linalg.sqrtm(product)

    cov_diff = np.trace(sigma1) + np.trace(sigma2) - 2*np.trace(sqrt_product)

    return np.sqrt(mean_diff + cov_diff)
```

---

## 7. 实现陷阱与建议

### 7.1 数值稳定性

**问题 1：Sinkhorn 中的数值溢出**

```python
# 不稳定：
K = np.exp(-C / epsilon)  # 当 C/epsilon 很大时上溢

# 稳定：使用 log-sum-exp
import scipy.special
log_K = -C / epsilon
# 在 log 空间进行计算
```

**问题 2：协方差矩阵的平方根**

```python
# 不稳定：直接计算
sqrt_sigma = scipy.linalg.sqrtm(sigma)

# 更稳定：使用特征值分解
eigenvalues, eigenvectors = np.linalg.eigh(sigma)
sqrt_sigma = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
```

### 7.2 超参数敏感性

**正则化参数 $\varepsilon$**：
- 太小：收敛慢，数值不稳定
- 太大：解过于平滑，偏离真实 OT
- 建议：从 0.1 开始，逐步减小

**Sinkhorn 迭代次数**：
- 通常 100-1000 次足够
- 监控收敛：检查边缘约束违反程度

### 7.3 可扩展性策略

| 场景 | 策略 |
|------|------|
| n ~ 1000 | 标准 Sinkhorn |
| n ~ 10^4 | 小批量 Sinkhorn |
| n ~ 10^6 | 神经 OT |
| 高维 | 切片 Wasserstein |
| 在线 | 随机 Sinkhorn |

---

## 8. 算法流程图

```
┌─────────────────────────────────────────────────────────────┐
│                   Optimal Transport 算法选择                 │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
      ┌──────────┐   ┌──────────┐   ┌──────────┐
      │ 精确求解  │   │ 熵正则化  │   │ 神经方法  │
      │   LP     │   │ Sinkhorn │   │  Neural  │
      └────┬─────┘   └────┬─────┘   └────┬─────┘
           │              │              │
           ▼              ▼              ▼
      ┌──────────┐   ┌──────────┐   ┌──────────┐
      │ 网络单纯形 │   │ 迭代缩放  │   │ 神经网络  │
      │ 复杂度 n³ │   │ 复杂度 n² │   │ 训练优化  │
      └──────────┘   └──────────┘   └──────────┘
```

---

## 9. 复现检查清单

复现 OT 实验时，确保：

- [ ] 正确定义概率分布（非负且归一化）
- [ ] 选择合适的成本函数（通常是欧氏距离）
- [ ] 对于 Sinkhorn，选择合适的正则化参数
- [ ] 验证解满足边缘约束（在数值误差范围内）
- [ ] 对于高维问题，考虑降维或近似方法
- [ ] 记录计算时间和内存使用

---

*详细的代码示例请参考 `code/` 目录。*
