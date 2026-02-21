# 方法论详解：信息论工具实战指南

---

## 1. 信息论基础计算

### 1.1 熵的计算

**离散情况（直方图估计）**：

```python
import numpy as np
from scipy.stats import entropy as scipy_entropy

def discrete_entropy(samples, base=2):
    """
    计算离散熵（香农熵）

    参数:
        samples: 观测样本序列
        base: 对数底（2=bits, e=nats, 10=hartleys）

    返回:
        H: 熵值
    """
    # 计算概率分布
    unique, counts = np.unique(samples, return_counts=True)
    probs = counts / len(samples)

    # 计算熵: H = -Σ p(x) log p(x)
    H = -np.sum(probs * np.log(probs + 1e-10))  # 加小值避免 log(0)

    # 转换单位
    if base == 2:
        H /= np.log(2)
    elif base == 10:
        H /= np.log(10)

    return H

# 示例：公平硬币 vs 作弊硬币
fair_coin = np.random.choice([0, 1], size=10000)  # p=0.5
biased_coin = np.random.choice([0, 1], size=10000, p=[0.1, 0.9])

print(f"公平硬币熵: {discrete_entropy(fair_coin):.3f} bits")
print(f"作弊硬币熵: {discrete_entropy(biased_coin):.3f} bits")
print(f"理论最大值: {np.log2(2):.3f} bits")
```

**连续情况（直方图粗粒化）**：

```python
def continuous_entropy_coarse_graining(samples, bins='auto'):
    """
    使用粗粒化方法估计连续熵

    注意：这实际上是离散熵，bin 大小影响结果
    """
    # 分箱离散化
    hist, bin_edges = np.histogram(samples, bins=bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]

    # 计算熵（注意 bin 宽度调整）
    probs = hist * bin_width
    probs = probs[probs > 0]  # 移除零概率

    H = -np.sum(probs * np.log2(probs)) - np.log2(bin_width)

    return H
```

### 1.2 互信息的计算

**离散互信息**：

```python
def discrete_mutual_information(x, y):
    """
    计算离散互信息 I(X;Y)

    公式: I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    # 计算联合分布
    joint_samples = np.column_stack([x, y])
    unique_joint, joint_counts = np.unique(joint_samples, axis=0, return_counts=True)
    joint_probs = joint_counts / len(x)

    # 边缘分布
    unique_x, counts_x = np.unique(x, return_counts=True)
    unique_y, counts_y = np.unique(y, return_counts=True)
    probs_x = counts_x / len(x)
    probs_y = counts_y / len(y)

    # 计算各项熵
    H_x = -np.sum(probs_x * np.log2(probs_x + 1e-10))
    H_y = -np.sum(probs_y * np.log2(probs_y + 1e-10))
    H_xy = -np.sum(joint_probs * np.log2(joint_probs + 1e-10))

    # I(X;Y) = H(X) + H(Y) - H(X,Y)
    I_xy = H_x + H_y - H_xy

    return max(0, I_xy)  # 确保非负

# 示例：线性关系 vs 独立
data_size = 5000
x = np.random.randn(data_size)
y_linear = x + 0.1 * np.random.randn(data_size)  # 强相关
y_indep = np.random.randn(data_size)  # 独立

print(f"线性关系 MI: {discrete_mutual_information(
    np.digitize(x, bins=10),
    np.digitize(y_linear, bins=10)
):.3f} bits")
print(f"独立变量 MI: {discrete_mutual_information(
    np.digitize(x, bins=10),
    np.digitize(y_indep, bins=10)
):.3f} bits")
```

**高斯估计器（连续变量的快速估计）**：

```python
def gaussian_mutual_information(x, y):
    """
    假设联合高斯分布的互信息

    公式: I(X;Y) = -0.5 * log2(1 - ρ²)
    其中 ρ 是皮尔逊相关系数

    优点：快速、无需分箱
    缺点：只能捕捉线性依赖
    """
    # 计算相关系数
    rho = np.corrcoef(x, y)[0, 1]

    # MI = -0.5 * log2(1 - rho^2)
    mi = -0.5 * np.log2(1 - rho**2 + 1e-10)

    return mi
```

---

## 2. 信息动力学

### 2.1 传递熵（Transfer Entropy）

传递熵是 Granger 因果的非线性推广：

```python
from sklearn.neighbors import KernelDensity

def transfer_entropy_kde(x, y, delay=1, k=4):
    """
    使用核密度估计计算传递熵

    T(X→Y) = I(Y_t; X_{t-delay} | Y_{t-delay})

    参数:
        x, y: 时间序列
        delay: 时间延迟
        k: 近邻数（用于 Kraskov 估计）
    """
    # 准备延迟嵌入
    y_past = y[:-delay]
    y_future = y[delay:]
    x_past = x[:-delay]

    # 对齐长度
    min_len = min(len(y_past), len(y_future), len(x_past))
    y_past = y_past[:min_len]
    y_future = y_future[:min_len]
    x_past = x_past[:min_len]

    # 计算条件互信息
    # TE = I(Y_future; X_past | Y_past)
    # 简化为: TE = H(Y_future, Y_past) + H(X_past, Y_past)
    #            - H(Y_past) - H(Y_future, X_past, Y_past)

    # 注意：这里使用直方图估计作为简化示例
    # 实际应用中应使用 Kraskov-Stögbauer-Grassberger (KSG) 估计器

    bins = min(20, int(np.sqrt(min_len) / 5))

    def joint_entropy(*arrays):
        """计算联合熵"""
        joint = np.column_stack(arrays)
        # 多维直方图
        hist, edges = np.histogramdd(joint, bins=bins)
        probs = hist.flatten() / np.sum(hist)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    H_yf_yp = joint_entropy(y_future, y_past)
    H_xp_yp = joint_entropy(x_past, y_past)
    H_yp = discrete_entropy(y_past)
    H_all = joint_entropy(y_future, x_past, y_past)

    te = H_yf_yp + H_xp_yp - H_yp - H_all

    return max(0, te)

# 示例：因果驱动系统
np.random.seed(42)
t = np.arange(1000)
x = np.sin(0.1 * t) + 0.3 * np.random.randn(1000)
y = 0.5 * np.roll(x, 3) + 0.5 * np.sin(0.15 * t) + 0.3 * np.random.randn(1000)

print(f"X → Y 的传递熵: {transfer_entropy_kde(x, y, delay=3):.4f} bits")
print(f"Y → X 的传递熵: {transfer_entropy_kde(y, x, delay=3):.4f} bits")
```

### 2.2 主动信息存储（Active Information Storage）

```python
def active_information_storage(x, k=3):
    """
    计算主动信息存储

    A(X) = I(X_t; X_{t-1}, ..., X_{t-k})

    参数:
        x: 时间序列
        k: 历史长度
    """
    # 构建延迟嵌入
    embedded = np.array([x[i:i+k] for i in range(len(x) - k)])
    future = x[k:]

    # 简化为成对 MI 的和（近似）
    ais = 0
    for i in range(k):
        past_i = embedded[:, i]
        mi = discrete_mutual_information(
            np.digitize(future, bins=10),
            np.digitize(past_i, bins=10)
        )
        ais += mi

    return ais

# 示例：高自相关 vs 低自相关
high_memory = np.cumsum(np.random.randn(1000))  # 随机游走，高自相关
low_memory = np.random.randn(1000)  # 白噪声，低自相关

print(f"高记忆系统 AIS: {active_information_storage(high_memory):.3f} bits")
print(f"低记忆系统 AIS: {active_information_storage(low_memory):.3f} bits")
```

---

## 3. 部分信息分解（PID）

### 3.1 简化版 PID 实现

```python
def pid_two_sources(target, source1, source2, bins=10):
    """
    两个源变量的简化 PID 计算

    返回:
        dict: 包含 unique1, unique2, redundant, synergistic
    """
    # 离散化
    t = np.digitize(target, np.histogram_bin_edges(target, bins=bins))
    s1 = np.digitize(source1, np.histogram_bin_edges(source1, bins=bins))
    s2 = np.digitize(source2, np.histogram_bin_edges(source2, bins=bins))

    # 计算各类互信息
    I_s1_t = discrete_mutual_information(s1, t)
    I_s2_t = discrete_mutual_information(s2, t)

    # 联合源
    joint_source = np.column_stack([s1, s2])
    joint_unique, joint_counts = np.unique(joint_source, axis=0, return_counts=True)

    # 计算 I({s1,s2}; t) - 需要映射回整数
    joint_map = {tuple(row): i for i, row in enumerate(joint_unique)}
    joint_int = np.array([joint_map[tuple(row)] for row in joint_source])
    I_joint_t = discrete_mutual_information(joint_int, t)

    # 条件互信息
    # 简化为分组计算
    unique_s2 = np.unique(s2)
    I_s1_t_given_s2 = 0
    for val in unique_s2:
        mask = s2 == val
        if np.sum(mask) > 10:  # 确保有足够的样本
            I_s1_t_given_s2 += (np.sum(mask) / len(s2)) * \
                discrete_mutual_information(s1[mask], t[mask])

    # PID 分解（使用 I_min 近似）
    # Redundant = min(I(s1;t), I(s2;t))
    redundant = min(I_s1_t, I_s2_t)

    # Unique = 总信息 - 其他
    # Synergistic = I({s1,s2};t) - max(I(s1;t), I(s2;t))
    synergistic = max(0, I_joint_t - max(I_s1_t, I_s2_t))

    # Unique = 个体 - redundant
    unique1 = max(0, I_s1_t - redundant)
    unique2 = max(0, I_s2_t - redundant)

    return {
        'total': I_joint_t,
        'unique1': unique1,
        'unique2': unique2,
        'redundant': redundant,
        'synergistic': synergistic,
        'raw_I_s1': I_s1_t,
        'raw_I_s2': I_s2_t
    }

# 示例：XOR 协同
np.random.seed(42)
s1 = np.random.randint(0, 2, 10000)
s2 = np.random.randint(0, 2, 10000)
target_xor = np.logical_xor(s1, s2).astype(int)

pid_xor = pid_two_sources(target_xor, s1, s2, bins=2)
print("XOR 系统的 PID 分解:")
for key, val in pid_xor.items():
    print(f"  {key}: {val:.4f} bits")

# 示例：冗余系统（AND）
target_and = np.logical_and(s1, s2).astype(int)
pid_and = pid_two_sources(target_and, s1, s2, bins=2)
print("\nAND 系统的 PID 分解:")
for key, val in pid_and.items():
    print(f"  {key}: {val:.4f} bits")
```

---

## 4. 网络推断

### 4.1 功能网络构建

```python
import networkx as nx

def build_functional_network(data, threshold=0.1, method='mi'):
    """
    从多变量时间序列构建功能连接网络

    参数:
        data: (time, variables) 数组
        threshold: 连边阈值
        method: 'mi' 或 'correlation'

    返回:
        G: NetworkX 图
    """
    n_vars = data.shape[1]
    G = nx.Graph()

    # 添加节点
    for i in range(n_vars):
        G.add_node(i)

    # 计算成对连接
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if method == 'mi':
                # 离散化后计算 MI
                mi = discrete_mutual_information(
                    np.digitize(data[:, i], bins=10),
                    np.digitize(data[:, j], bins=10)
                )
                weight = mi
            else:
                # 相关系数
                weight = abs(np.corrcoef(data[:, i], data[:, j])[0, 1])

            if weight > threshold:
                G.add_edge(i, j, weight=weight)

    return G

# 可视化函数
def plot_network(G, title="Functional Network"):
    """绘制网络图"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=3, iterations=50)

    # 节点
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                          node_size=500, alpha=0.9)
    nx.draw_networkx_labels(G, pos)

    # 边（权重决定粗细）
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights],
                          alpha=0.5, edge_color='gray')

    plt.title(title)
    plt.axis('off')
    return plt

# 示例：模拟 5 节点网络
np.random.seed(42)
np.random.seed(42)
n_nodes = 5
t = np.arange(1000)

# 创建相关的时间序列
data = np.zeros((1000, n_nodes))
data[:, 0] = np.sin(0.1 * t) + 0.2 * np.random.randn(1000)  # 驱动节点
for i in range(1, n_nodes):
    data[:, i] = 0.5 * data[:, i-1] + 0.3 * np.sin(0.1 * t + i) + 0.2 * np.random.randn(1000)

G = build_functional_network(data, threshold=0.05)
print(f"网络节点数: {G.number_of_nodes()}")
print(f"网络边数: {G.number_of_edges()}")
print(f"平均聚类系数: {nx.average_clustering(G):.3f}")
```

---

## 5. 复杂性度量

### 5.1 TSE 复杂性

```python
def tse_complexity(data, max_k=None):
    """
    计算 TSE 复杂性

    C_TSE = Σ_{k=1}^{N-1} I(X_1,...,X_k; X_{k+1}) / k

    对于大数据集，计算量很大。这里使用近似方法。
    """
    n_vars = data.shape[1]
    if max_k is None:
        max_k = min(n_vars - 1, 5)  # 限制计算量

    complexity = 0
    for k in range(1, max_k + 1):
        # 随机采样 k 个变量
        vars_sample = np.random.choice(n_vars, size=min(k+1, n_vars), replace=False)

        if len(vars_sample) >= 2:
            # 计算 I(X_1...X_k; X_{k+1})
            subset = data[:, vars_sample]

            # 简化为成对 MI 的平均
            mi_sum = 0
            for i in range(len(vars_sample) - 1):
                mi = discrete_mutual_information(
                    np.digitize(subset[:, i], bins=10),
                    np.digitize(subset[:, -1], bins=10)
                )
                mi_sum += mi

            complexity += mi_sum / k

    return complexity / max_k

# 示例：比较不同系统的复杂性
# 1. 独立系统
indep_data = np.random.randn(1000, 5)

# 2. 完全相关系统
common = np.random.randn(1000, 1)
correlated_data = np.hstack([common] * 5) + 0.1 * np.random.randn(1000, 5)

# 3. 中等相关系统
mixed_data = np.zeros((1000, 5))
for i in range(5):
    mixed_data[:, i] = 0.5 * common.flatten() + 0.5 * np.random.randn(1000)

print(f"独立系统 TSE: {tse_complexity(indep_data):.3f}")
print(f"相关系统 TSE: {tse_complexity(correlated_data):.3f}")
print(f"混合系统 TSE: {tse_complexity(mixed_data):.3f}")
```

### 5.2 O-Information

```python
def o_information(data):
    """
    计算 O-Information

    Ω(X) = (N-1) * H(X) - Σ H(X\\{i})

    Ω > 0: 冗余主导
    Ω < 0: 协同主导
    """
    n_vars = data.shape[1]

    # 计算联合熵（近似：使用第一个主成分）
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    joint_repr = pca.fit_transform(data).flatten()
    H_joint = discrete_entropy(np.digitize(joint_repr, bins=20))

    # 计算各边缘熵
    sum_marginals = 0
    for i in range(n_vars):
        H_i = discrete_entropy(np.digitize(data[:, i], bins=20))
        sum_marginals += H_i

    # O-Information
    omega = (n_vars - 1) * H_joint - sum_marginals

    return omega

# 示例
print(f"\n独立系统 O-Info: {o_information(indep_data):.3f}")
print(f"相关系统 O-Info: {o_information(correlated_data):.3f}")
print(f"混合系统 O-Info: {o_information(mixed_data):.3f}")
```

---

## 6. 实用工具推荐

### 6.1 推荐软件包

| 包名 | 语言 | 优势 | 适用场景 |
|------|------|------|----------|
| **JIDT** | Java | 成熟稳定，算法全面 | 学术研究，生产环境 |
| **IDTxl** | Python | 因果推断导向 | 大规模网络分析 |
| **DIT** | Python | PID 分析专业 | 信息分解研究 |
| **NPEET** | Python | 非参数估计 | 连续变量分析 |

### 6.2 安装与使用示例

```python
# IDTxl 安装与使用示例
"""
pip install idtxl
"""

# 基本使用框架
"""
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data

# 准备数据
data = Data(np_array, dim_order='sp')  # samples x processes

# 初始化分析器
network_analysis = MultivariateTE()

# 运行分析
results = network_analysis.analyse_network(
    settings={
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 200,
        'n_perm_min_stat': 200,
        'alpha_max_stat': 0.05,
        'alpha_min_stat': 0.05
    },
    data=data
)
"""
```

---

## 7. 实现注意事项

### 7.1 偏差与方差权衡

| 估计器类型 | 偏差 | 方差 | 适用场景 |
|-----------|------|------|----------|
| 直方图 | 高 | 低 | 大数据集，离散变量 |
| 核密度 (KDE) | 中 | 中 | 连续变量，中等数据 |
| K-近邻 (KSG) | 低 | 高 | 小数据集，高维变量 |
| 高斯 | 低* | 低 | 近似高斯分布的数据 |

*只有线性依赖

### 7.2 统计显著性检验

```python
def permutation_test_mi(x, y, n_permutations=1000):
    """
    置换检验互信息的显著性

    H0: X 和 Y 独立
    """
    # 观测值
    observed_mi = discrete_mutual_information(x, y)

    # 置换分布
    permuted_mis = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        perm_mi = discrete_mutual_information(x, y_perm)
        permuted_mis.append(perm_mi)

    # p-value
    p_value = np.mean(np.array(permuted_mis) >= observed_mi)

    return observed_mi, p_value, permuted_mis
```

### 7.3 常见陷阱

1. **维度灾难**：高维变量需要指数级增加的数据量
   - 解决：降维、分层分析、连续估计器

2. **非平稳性**：时间序列统计特性变化
   - 解决：滑动窗口、分段分析、自适应估计

3. **有限样本偏差**：小样本导致熵低估
   - 解决：Miller-Madow 校正、收缩估计器

4. **虚假相关性**：多变量检验导致假阳性
   - 解决：多重检验校正（FDR、Bonferroni）
