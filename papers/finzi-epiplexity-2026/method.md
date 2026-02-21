# 方法论：Epiplexity 的数学定义与估计

---

## 1. 预备知识

### 1.1 时间有界概率模型

**定义 7**（时间有界概率模型）：

设 T: ℕ → ℕ 是非递减的时间可构造函数，U 是固定的前缀自由通用图灵机。

程序 P 是 T-时间概率模型，如果：

**评估**：输入 (0, x)，U(P, (0,x)) 在 T(n) 步内停机，输出概率值 Prob_P(x) ∈ [0,1]

**采样**：输入 (1, u)，u ∈ {0,1}^∞ 是无限随机带，U(P, (1,u)) 在 T(n) 步内停机，输出样本 Sample_P(u) ∈ {0,1}^n

且满足一致性：
```
Σ_{x∈{0,1}^n} Prob_P(x) = 1
Pr_{u~U_∞}[Sample_P(u) = x] = Prob_P(x)
```

**直观理解**：
- P 既可以在时间 T 内计算概率
- 也可以在时间 T 内生成样本
- 包括大多数序列模型（Transformer等）

---

## 2. Epiplexity 与 Time-Bounded Entropy 的定义

### 2.1 核心定义

**定义 8**（Epiplexity 和 Time-Bounded Entropy）：

对于随机变量 X ∈ {0,1}^n，计算约束 T：

```
P* = arg min_{P ∈ P_T} { |P| + E[log 1/P(X)] }

S_T(X) := |P*|                    （Epiplexity）
H_T(X) := E[log 1/P*(X)]          （Time-Bounded Entropy）
```

其中 |P| 是程序 P 的长度（比特），期望取自 X。

**直观理解**：
- **S_T(X)**：描述数据分布所需的程序长度（结构信息）
- **H_T(X)**：使用该程序编码数据所需的期望长度（随机信息）
- **总和**：时间有界 MDL（最小描述长度）

### 2.2 与均匀分布的对比

对于均匀随机变量 U_n：
```
S_T(U_n) ≤ c_2                    （常数级，只需uniform分布程序）
H_T(U_n) ≥ n                      （最大熵）
MDL_T(U_n) ≤ n + c_2
```

这表明均匀分布有：
- 很低的 Epiplexity（没有结构）
- 很高的 Time-Bounded Entropy（完全随机）

### 2.3 条件版本

**定义 11**（条件 Epiplexity 和 Time-Bounded Entropy）：

对于随机变量对 (X, Y)：

```
P*_{Y|X} = arg min_{P ∈ P_T^X} { |P| + E_{(X,Y)}[-log P(Y|X)] }

S_T(Y|X) := |P*_{Y|X}|
H_T(Y|X) := E_{(X,Y)}[-log P*_{Y|X}(Y|X)]
```

**应用**：
- 图像分类：X=图像，Y=标签
- 语言建模：X=前缀，Y=下一个token
- 条件模型只关心 Y|X 的信息

---

## 3. 理论性质

### 3.1 基本性质

```
(1) S_T(X) ≥ 0, H_T(X) ≥ 0                           （非负性）

(2) H(X) ≤ S_T(X) + H_T(X) ≤ n + c_1                （有界性）
    H(X) 是香农熵，MDL_T(X) 上界为 n + 常数

(3) MDL_T'(X) ≤ MDL_T(X) 当 T' ≥ T                  （单调性）
    更多计算时间 → 更短的描述长度

(4) MDL_T'(f^{-1}(X)) ≤ MDL_T(X) + |f| + c_2        （变换）
    T'(n) = T(n) + Time(f)
    注意：f 和 f^{-1} 的计算复杂度可能不对称！
```

### 3.2 伪随机数生成器的关键定理

**定理 9**（CSPRNG 的 Epiplexity）：

对于任何 T ∈ Poly(n)，G ∈ CSPRNG 拉伸 k 比特到 n = poly(k) 比特：

```
H_T(G(U_k)) > n - 2 - nε(k)        （高 Time-Bounded Entropy）
S_T(G(U_k)) ≤ c + nε(k)            （低 Epiplexity）
```

其中 ε(k) 是可忽略函数。

**证明思路**：
1. 如果 H_T 不接近 n，则可以区分 G 的输出和真随机
2. 这与 CSPRNG 的定义矛盾
3. Epiplexity 低是因为 G 本身程序很短

**对比其他复杂度度量**：

| 度量 | CSPRNG 输出 |
|------|------------|
| 香农熵 H(X) | k（种子长度） |
| 柯尔莫哥洛夫复杂度 K(X) | ≤ k + c |
| 时间有界柯氏复杂度 | ≤ k + c |
| Levin 复杂度 | 小 |
| **Time-Bounded Entropy H_T** | **≈ n（大）** |
| **Epiplexity S_T** | **≈ c（小）** |

**关键洞察**：Epiplexity 是唯一正确刻画 CSPRNG 的度量——对多项式时间观察者来说，输出"看起来随机"（高 Entropy），没有可学习的结构（低 Epiplexity）。

### 3.3 存在性定理

**定理 10**（高 Epiplexity 随机变量的存在性）：

假设存在对非均匀概率多项式时间敌手安全的单向函数，则存在随机变量序列 {X_n}：

```
S_Poly(X_n) = Ω(log n)
```

**证明思路**：
1. 使用单向函数构造伪随机函数族
2. 随机选择函数索引
3. 对多项式时间观察者来说，函数看起来随机
4. 但函数本身有结构（可由多项式时间计算）

**局限**：这只是对数增长，远低于自然数据中观察到的幂律增长。更强的下界是开放问题。

---

## 4. 实际估计方法

### 4.1 Prequential Coding（预quential编码）

**核心思想**：通过训练过程编码模型

**过程**：
1. 随机初始化模型 P_0
2. 对于每个训练样本 Z_i：
   - 使用当前模型 P_i 编码 Z_i（需要 log 1/P_i(Z_i) 比特）
   - 训练模型得到 P_{i+1}

**总描述长度**：
```
L(Z_{0:M}, P_M) = Σ_{i=0}^{M-1} log 1/P_i(Z_i)
```

**估计模型长度**（启发式）：
```
|P_preq| ≈ Σ_{i=0}^{M-1} (log 1/P_i(Z_i) - log 1/P_M(Z_i))
```

**直观解释**：
- 损失曲线下的面积（最终损失之上）
- 训练过程中"压缩"的信息量

**优点**：
- 容易计算（已有训练运行即可）
- 有直观解释

**局限**：
- 不是严格上界
- 依赖于启发式假设

### 4.2 Requential Coding（requential编码）

**核心思想**：使用师生模型显式编码

**设置**：
- 教师模型 P_t：在真实数据上训练
- 学生模型 P_s：在教师生成的合成数据上训练

**编码**：
```
|P_req| = Σ KL(P_s || P_t)
```

**过程**：
1. 教师模型生成合成数据
2. 学生模型学习匹配教师分布
3. 累积的 KL 散度就是模型描述长度

**与 Prequential Coding 的关系**：

两者通常给出相似的排名，但：
- Prequential：通常更大（高估）
- Requential：更严格

### 4.3 计算最优配置

**目标**：在给定计算预算 T 下最小化 MDL

**计算模型**：
- 训练时间：≈ 6ND FLOPs（N参数，D tokens）
- 推理时间：≈ 2ND FLOPs

**优化问题**：
```
min_{N, D, lr, ...} |P| + E[log 1/P(X)]
s.t. 6ND + 2ND ≤ T
```

**使用 μP（Maximal Update Parameterization）**：
- 确保最优学习率跨模型规模一致
- 简化超参数调优

**Pareto 前沿**：
- 对不同计算预算 T，找到最优配置
- 追踪 MDL_T 随 T 的变化

---

## 5. 神经网络实现

### 5.1 模型类

使用神经网络作为概率模型 P：

```python
class NeuralProbabilisticModel:
    """
    时间 T 内可评估和采样的神经网络模型
    """
    def __init__(self, architecture, n_params):
        self.net = architecture(n_params)
        self.n_params = n_params

    def evaluate(self, x):
        """在 T 步内计算 P(x)"""
        return self.net.probability(x)

    def sample(self, random_tape):
        """在 T 步内生成样本"""
        return self.net.generate(random_tape)
```

### 5.2 程序长度估计

**Prequential 版本**：

```python
class PrequentialEstimator:
    def estimate_model_length(self, training_losses):
        """
        从训练损失曲线估计 |P|

        training_losses: [L_0, L_1, ..., L_M]
        L_i = E[log 1/P_i(Z)]
        """
        final_loss = training_losses[-1]
        model_length = sum(L_i - final_loss for L_i in training_losses)
        return model_length

    def estimate_entropy(self, validation_data, final_model):
        """估计 H_T(X)"""
        return np.mean([final_model.negative_log_likelihood(x)
                       for x in validation_data])
```

**Requential 版本**：

```python
class RequentialEstimator:
    def estimate_model_length(self, teacher, student, num_steps):
        """
        使用师生模型估计 |P|

        累积 KL 散度
        """
        model_length = 0
        for _ in range(num_steps):
            # 教师生成数据
            synthetic_data = teacher.sample()

            # 学生在教师数据上训练
            student.train_on(synthetic_data)

            # 计算 KL 散度
            kl = compute_kl(student, teacher, synthetic_data)
            model_length += kl

        return model_length
```

---

## 6. 代码示例

### 6.1 基础 Epiplexity 估计

```python
import numpy as np
import torch
import torch.nn as nn

def estimate_epiplexity_preq(model_class, data, max_tokens,
                              model_sizes, device='cuda'):
    """
    使用 Prequential Coding 估计 Epiplexity

    参数:
        model_class: 模型类
        data: 数据集
        max_tokens: 最大训练tokens
        model_sizes: 要尝试的参数数量列表

    返回:
        best_config: 最优配置
        S_T: Epiplexity 估计
        H_T: Time-Bounded Entropy 估计
    """
    best_mdl = float('inf')
    best_config = None

    for N in model_sizes:
        # 计算对应的数据量
        D = max_tokens // (6 * N)  # 计算约束

        # 训练模型
        model = model_class(N).to(device)
        losses = train_model(model, data[:D])

        # 估计 |P|
        final_loss = losses[-1]
        model_length = sum(l - final_loss for l in losses)

        # 估计 H_T
        entropy = evaluate_loss(model, data[D:])

        # 计算 MDL
        mdl = model_length + entropy

        if mdl < best_mdl:
            best_mdl = mdl
            best_config = {
                'N': N, 'D': D,
                'S_T': model_length,
                'H_T': entropy,
                'MDL': mdl
            }

    return best_config
```

### 6.2 比较不同数据的 Epiplexity

```python
def compare_datasets(datasets, model_class, max_tokens):
    """
    比较多个数据集的 Epiplexity

    参数:
        datasets: {name: data} 字典
        model_class: 模型类
        max_tokens: 计算预算

    返回:
        results: {name: {'S_T': ..., 'H_T': ...}}
    """
    results = {}

    for name, data in datasets.items():
        config = estimate_epiplexity_preq(
            model_class, data, max_tokens,
            model_sizes=[1e6, 1e7, 1e8]
        )

        results[name] = {
            'Epiplexity': config['S_T'],
            'Entropy': config['H_T'],
            'Total_MDL': config['MDL']
        }

    return results


# 示例：比较不同数据
if __name__ == "__main__":
    datasets = {
        'random_noise': generate_random_data(),
        'simple_patterns': generate_pattern_data(),
        'natural_language': load_text_data(),
        'images': load_image_data()
    }

    results = compare_datasets(datasets, TransformerModel, 1e12)

    # 打印结果
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Epiplexity: {metrics['Epiplexity']:.2f} MB")
        print(f"  Entropy: {metrics['Entropy']:.2f} bits/token")
```

---

## 7. 实际考虑

### 7.1 超参数调优

**关键超参数**：
- 模型规模 N
- 训练数据量 D
- 学习率 lr
- 批次大小 batch_size

**使用 μP**：
- 确保最优 lr 跨模型规模一致
- 简化调优过程

### 7.2 计算资源

**估计 Epiplexity 的成本**：
- 需要训练多个模型（不同规模）
- 对每种数据类型都需要单独估计
- 可能需要 GPU 集群

**加速技巧**：
- 使用较小的代理模型
- 提前停止（如果损失不再下降）
- 并行训练多个配置

### 7.3 数值稳定性

**损失曲线平滑**：
- 使用移动平均
- 处理异常值

**数值精度**：
- 使用 float64 计算累积和
- 防止溢出/下溢

---

## 8. 与其他度量的比较

| 度量 | 计算约束 | 是否可计算 | 捕捉结构 |
|------|---------|-----------|---------|
| 香农熵 H(X) | 无 | 估计困难 | 否 |
| 柯氏复杂度 K(X) | 无 | 不可计算 | 部分 |
| 时间有界柯氏复杂度 K^t(X) | 有 | 困难 | 部分 |
| Sophistication | 无 | 不可计算 | 是 |
| **Epiplexity S_T(X)** | **有** | **可估计** | **是** |

Epiplexity 的独特优势：
1. **计算约束**：符合实际AI系统
2. **可估计**：通过神经网络训练
3. **捕捉结构**：区分随机和结构信息
