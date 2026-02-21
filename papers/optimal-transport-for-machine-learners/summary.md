# Optimal Transport for Machine Learners - 内容总结

---

## 1. 背景与动机

### 什么是最优传输？

最优传输（Optimal Transport, OT）是连接**优化**、**偏微分方程**和**概率论**的基础数学理论。它提供了一种比较概率分布的强大框架，近年来在机器学习中变得尤为重要，特别是在设计和评估**生成模型**方面。

### 为什么机器学习需要 OT？

1. **分布比较**：传统方法（如 KL 散度）无法捕捉分布的几何结构
2. **生成模型**：GANs、扩散模型都涉及分布间的映射
3. **神经网络训练**：可以将参数更新视为梯度流
4. **Transformer 动态**：token 分布的演化可以用 OT 分析

---

## 2. 核心问题陈述

### Monge 问题（1781）

给定两个概率测度 $\mu$ 和 $\nu$，以及成本函数 $c(x,y)$，寻找确定性映射 $T$ 使得：

$$\inf_T \left\{ \int c(x, T(x)) d\mu(x) : T_\#\mu = \nu \right\}$$

其中 $T_\#\mu = \nu$ 表示 $T$ 将 $\mu$ 推前到 $\nu$。

### Kantorovich 松弛（1942）

由于 Monge 问题在离散情况下可能没有解，Kantorovich 引入了耦合（coupling）的概念：

$$\inf_\pi \left\{ \int c(x,y) d\pi(x,y) : \pi \in \Pi(\mu, \nu) \right\}$$

其中 $\Pi(\mu, \nu)$ 是所有边缘分布为 $\mu$ 和 $\nu$ 的联合分布集合。

---

## 3. 主要贡献

本课程笔记系统性地涵盖了：

### 理论方面

| 主题 | 关键内容 |
|------|----------|
| **Monge 问题** | 离散点云匹配、最优分配问题 |
| **Kantorovich 形式化** | 松弛问题、耦合的存在性 |
| **Brenier 定理** | 凸势能、最优映射的存在唯一性 |
| **对偶形式化** | Kantorovich 对偶、$c$-变换 |
| **动态形式化** | Benamou-Brenier 公式、连续性方程 |
| **Bures 度量** | 高斯分布上的 Wasserstein 距离 |
| **梯度流** | JKO 方案、Wasserstein 空间中的梯度流 |

### 计算方法

| 方法 | 描述 |
|------|------|
| **线性规划** | 离散 OT 作为网络流问题 |
| **半离散求解器** | 一个连续分布到离散点云 |
| **熵正则化** | Sinkhorn 算法、快速近似求解 |

### 机器学习应用

| 应用 | 说明 |
|------|------|
| **神经网络训练** | 参数空间中的梯度流视角 |
| **Transformer Token 动态** | 跨层的 token 分布演化 |
| **GANs 结构** | 生成器-判别器的博弈与 OT |
| **扩散模型** | 与最优传输的深刻联系 |

---

## 4. 关键结果

### 理论结果

1. **Brenier 定理**：对于二次成本 $c(x,y) = \frac{1}{2}\|x-y\|^2$，最优映射存在且唯一，形式为 $T = \nabla\phi$，其中 $\phi$ 是凸函数。

2. **Kantorovich 对偶**：
   $$W_c(\mu, \nu) = \sup_{\phi, \psi} \left\{ \int \phi d\mu + \int \psi d\nu : \phi(x) + \psi(y) \leq c(x,y) \right\}$$

3. **Benamou-Brenier 公式**（动态形式化）：
   $$W_2(\mu, \nu)^2 = \inf_{(\rho_t, v_t)} \int_0^1 \int \|v_t(x)\|^2 d\rho_t(x) dt$$

### 计算结果

- **Sinkhorn 算法**：通过熵正则化将 OT 转化为矩阵缩放问题，收敛速度快
- 复杂度：从 $O(n^3)$ 降低到 $O(n^2 \log n / \varepsilon)$

---

## 5. 定量指标

### Wasserstein 距离

- **定义**：$W_p(\mu, \nu) = \left( \inf_\pi \int d(x,y)^p d\pi \right)^{1/p}$
- **$W_1$（Earth Mover's Distance）**：具有对偶形式，计算方便
- **$W_2$**：与梯度流理论联系最紧密

### Bures-Wasserstein 距离

对于高斯分布 $\mu = \mathcal{N}(m_\mu, \Sigma_\mu)$，$\nu = \mathcal{N}(m_\nu, \Sigma_\nu)$：

$$B(\mu, \nu)^2 = \|m_\mu - m_\nu\|^2 + \text{tr}(\Sigma_\mu) + \text{tr}(\Sigma_\nu) - 2\text{tr}\left(\Sigma_\mu^{1/2}\Sigma_\nu\Sigma_\mu^{1/2}\right)^{1/2}$$

---

## 6. 与其他工作的关系

### 对比 Peyré & Cuturi [23]

| 方面 | 本笔记 | Peyré & Cuturi |
|------|--------|-------------------|
| 重点 | 数学理论 | 计算方法 |
| 深度 | 中等 | 深入 |
| ML 应用 | 概念层面 | 实现细节 |
| 数学严谨性 | 高 | 中等 |

### 对比 Santambrogio [25]

| 方面 | 本笔记 | Santambrogio |
|------|--------|--------------|
| 目标读者 | ML 研究者 | 应用数学家 |
| 数学深度 | 适中 | 非常高 |
| 应用示例 | ML 为主 | 物理、经济为主 |

---

## 7. 结论

本课程笔记为机器学习研究者提供了一个**平衡的**最优传输学习路径：

- ✅ **足够的数学深度**：理解理论基础和证明
- ✅ **聚焦 ML 应用**：神经网络、Transformer、生成模型
- ✅ **计算方法**：从线性规划到 Sinkhorn 算法
- ✅ **前沿连接**：扩散模型与最优传输

**适合人群**：
- 想要理解生成模型数学原理的研究者
- 对分布学习和概率建模感兴趣的学生
- 希望将 OT 工具应用到新问题的工程师
