# 基于反事实机制演化的因子挖掘

## 研究动机

现有自动因子挖掘通常遵循：

$$
\mathrm{Generation}
\rightarrow
\mathrm{Evaluation}
\rightarrow
\mathrm{Selection}
$$

候选因子最终得到一个整体评价：

$$
f\rightarrow R(f)
$$

但 factor-level reward 无法进一步回答：

> **一个因子为什么有效？哪些结构机制真正贡献了预测能力？这些机制应如何被保留、替换或重新组合？**

同时，高质量 Alpha 往往依赖多个结构组件之间的组合关系，仅围绕单个算子、窗口或 subtree 进行局部搜索，难以产生足够大的结构跃迁。

因此，本工作将 **Structural Counterfactual Reasoning** 用于机制诊断，并与 evolutionary search 结合：

> **Counterfactual reasoning 识别有效机制，Evolutionary search 在机制层面进行新的结构探索。**

整体流程为：

$$
\mathrm{Factor\ Pool}
\rightarrow
\mathrm{Counterfactual\ Diagnosis}
\rightarrow
\mathrm{Mechanism\ Credit}
\rightarrow
\mathrm{Macro\ Evolution}
\rightarrow
\mathrm{Pool\ Update}
$$

---

## 核心 Idea

对于因子 $f$，将其分解为若干具有完整金融或数学含义的结构机制：

$$
M(f)=\{m_1,m_2,\ldots,m_K\}
$$

每个 mechanism 包含：

- 一个简洁的语义描述；
- 一个对应的 expression subtree。

例如：

```text
Mechanism:
short-term reversal

Expression:
-Delta(close, 5)
```

或：

```text
Mechanism:
abnormal-volume confirmation

Expression:
volume / TSMean(volume, 20)
```

Mechanism 是 counterfactual attribution、mutation、replacement 和 crossover 的基本结构单元。

对 mechanism $m_i$ 构造 ablation：

$$
T(f,m_i)
$$

其中 $T$ 表示 Remove / Neutralize，并尽量保持其他结构不变。

重新评价得到：

$$
\Delta_i^{\mathrm{cf}}
=
R(T(f,m_i))-R(f)
$$

其中：

- $\Delta_i^{\mathrm{cf}}<0$：移除后表现下降，该机制具有正贡献；
- $\Delta_i^{\mathrm{cf}}\approx0$：该机制贡献有限；
- $\Delta_i^{\mathrm{cf}}>0$：移除后表现改善，该机制可被进一步重构。

这些结构证据进一步用于机制级搜索，而不是直接作为单点局部编辑。

---

# 基础定义

## Factor Reward

对于因子 $f$，在每个交易日计算横截面 RankIC：

$$
IC_t(f)
=
\operatorname{Spearman}
\left(
f_t,\,
r_{t\rightarrow t+h}
\right)
$$

定义：

$$
R(f)
=
\frac{
\operatorname{Mean}(IC_t(f))
}{
\operatorname{Std}(IC_t(f))+\epsilon
}
$$

即以 RankICIR 作为基础 factor reward。

---

## Pool Utility

对于因子池：

$$
P=\{f_1,f_2,\ldots,f_K\}
$$

对各因子的横截面输出进行 rank normalization，记为 $\tilde f_{i,t}$，并构造等权组合信号：

$$
F_{P,t}
=
\frac{1}{K}
\sum_{i=1}^{K}
\tilde f_{i,t}
$$

定义：

$$
U(P)
=
\operatorname{RankICIR}(F_P)
$$

候选因子对当前 pool 的边际贡献为：

$$
r_{\mathrm{pool}}(f\mid P)
=
U(P\cup\{f\})-U(P)
$$

---

## Factor Selection

Pool selection 统一考虑单因子质量、组合贡献、多样性和 signal cost：

$$
S(f\mid P)
=
\alpha R(f)
+
\beta r_{\mathrm{pool}}(f\mid P)
+
\gamma D(f,P)
-
\lambda C_{\mathrm{cost}}(f)
$$

多样性定义为：

$$
\rho(f,g)
=
\frac{1}{T}
\sum_t
\left|
\operatorname{Spearman}(f_t,g_t)
\right|
$$

$$
D(f,P)
=
1-\max_{g\in P}\rho(f,g)
$$

signal cost 使用相邻交易日 factor ranking 的变化作为 turnover proxy：

$$
C_{\mathrm{cost}}(f)
=
\frac{1}{2}
\left[
1-
\frac{1}{T-1}
\sum_t
\operatorname{Spearman}(f_t,f_{t-1})
\right]
$$

实际 selection 时，各项可在当前 candidate set 内统一归一化。

记：

$$
\operatorname{Select}_K(\mathcal C)
$$

为依据上述原则从候选集合 $\mathcal C$ 中构建大小为 $K$ 的 factor pool。

---

# 方法整体设计

## Step 1：Initial Pool Construction

初始提供约：

$$
|\mathcal C_0|\approx200
$$

个候选因子，可来自现有搜索算法、LLM generation、人工因子库或其混合。

通过统一的 pool-aware selection 构建工作池：

$$
P_0
=
\operatorname{Select}_{60}(\mathcal C_0)
$$

之后整个 evolutionary process 始终维护：

$$
|P_t|=60
$$

未进入 $P_0$ 的因子不再参与后续搜索。

---

## Step 2：Counterfactual Diagnosis

每轮从当前工作池 $P_t$ 中选择一部分具有较高质量或探索价值的 factor 作为 parent：

$$
E_t\subset P_t
$$

对于每个 $f\in E_t$，LLM 将其分解为：

$$
M(f)=\{m_1,m_2,\ldots,m_K\}
$$

随后对主要 mechanism 依次执行 ablation：

$$
T(f,m_i)
$$

并计算：

$$
\Delta_i^{\mathrm{cf}}
=
R(T(f,m_i))-R(f)
$$

由此得到 factor 内部不同机制的结构贡献信息。

---

## Step 3：Mechanism Credit

单因子贡献并不完全等价于其对整体 factor pool 的价值。

对于 $f\in P$，定义：

$$
P_{-f}=P\setminus\{f\}
$$

则 mechanism $m$ 的 pool-level credit 为：

$$
C_{\mathrm{pool}}(m;f,P)
=
U(P)
-
U\left(
P_{-f}\cup\{T(f,m)\}
\right)
$$

该量衡量移除 mechanism $m$ 后整个 pool utility 的变化。

因此每个 mechanism 同时具有两类证据：

$$
\Delta^{\mathrm{cf}}
\qquad\text{and}\qquad
C_{\mathrm{pool}}
$$

分别描述其对单因子表现和整体 pool 的作用。

---

## Step 4：Mechanism Memory

Counterfactual diagnosis 的结果保存为结构化 Mechanism Memory：

| Factor | Mechanism | Expression | $\Delta^{\mathrm{cf}}$ | $C_{\mathrm{pool}}$ | Action |
|---|---|---|---:|---:|---|
| $f_1$ | short-term reversal | `-Delta(close,5)` | -0.021 | 0.014 | Preserve |
| $f_1$ | volume confirmation | `volume/Mean(volume,20)` | -0.006 | 0.018 | Preserve |
| $f_2$ | volatility normalization | `x/Std(ret,20)` | 0.004 | -0.001 | Replace |

Mechanism Memory 记录已经被真实 intervention 验证过的结构证据，并作为后续 evolution 的上下文。

---

## Step 5：Macro Evolution

利用当前 parent 的 counterfactual evidence 和 Mechanism Memory 生成新的 factor。

主要包含三类操作。

### Mechanism Mutation

保留高价值机制，对其他部分进行结构重写：

$$
f'
=
\operatorname{Mutate}
(
m_{\mathrm{keep}},
m_{\mathrm{replace}}
)
$$

### Mechanism Replacement

将某个完整机制替换为新的经济或数学结构，例如：

$$
\mathrm{Momentum}
\rightarrow
\mathrm{Reversal}
$$

### Credit-Aware Crossover

对于两个 parent：

$$
f_A,\qquad f_B
$$

选择其中具有较强结构证据的 mechanism 进行重组：

$$
f_{\mathrm{child}}
=
\operatorname{Combine}
\left(
M_A^{\mathrm{selected}},
M_B^{\mathrm{selected}}
\right)
$$

生成本轮 offspring：

$$
O_t
$$

---

## Step 6：Offspring Refinement

新生成的 factor 首先使用默认参数进行基础评价。

对于表现较好的候选，对 operator 中的离散常量进行小规模枚举。例如：

$$
W_{\mathrm{window}}
=
\{5,10,20,40,60\}
$$

对于参数化因子：

$$
f(x;\theta)
$$

在预定义合法集合 $\Theta$ 中选择：

$$
\theta^*
=
\arg\max_{\theta\in\Theta}
R(f(\cdot;\theta))
$$

得到 refined offspring。

---

## Step 7：Pool Update

将当前工作池与 offspring 合并：

$$
\mathcal C_{t+1}
=
P_t\cup O_t
$$

并重新执行统一的 pool selection：

$$
P_{t+1}
=
\operatorname{Select}_{60}(\mathcal C_{t+1})
$$

因此每一轮 population evolution 为：

$$
P_t
\rightarrow
P_t\cup O_t
\rightarrow
P_{t+1}
$$

重复若干轮后得到最终工作池：

$$
P_G
$$

---

## Step 8：Final Selection

搜索完成后，从最终工作池中选择约 30 个最终因子：

$$
P_{\mathrm{final}}
=
\operatorname{Select}_{30}(P_G)
$$

整个 pool 生命周期为：

$$
200
\rightarrow
60
\rightarrow
60
\rightarrow
\cdots
\rightarrow
60
\rightarrow
30
$$

其中 60 为 evolutionary working pool，30 为最终输出的 factor pool。

---

# LLM 在其中的作用

LLM 主要负责：

### Mechanism Decomposition

将 factor 分解为少量具有明确金融或数学语义、并能映射到 expression subtree 的 mechanism。

### Counterfactual Proposal

为 mechanism 构造对应的 Remove / Neutralize intervention。

### Evolution Proposal

结合当前 factor 的 counterfactual evidence 与 Mechanism Memory，执行 mutation、replacement 和 crossover。

以下部分由真实数据和程序完成：

- factor execution；
- reward calculation；
- counterfactual evaluation；
- pool utility；
- parameter enumeration；
- factor selection。

基本原则为：

> **LLM 提出结构假设，真实市场数据提供结构证据。**

---

# Data Protocol

数据划分为：

$$
D_{\mathrm{train}}=2010\text{-}2021
$$

$$
D_{\mathrm{valid}}=2022
$$

$$
D_{\mathrm{test}}=2023\text{-}2026.04
$$

Factor search、counterfactual diagnosis、parameter refinement 和 iterative pool evolution 使用 Train。

Validation 用于搜索完成后的 final factor selection：

$$
60\rightarrow30
$$

Test 保持封存，用于最终实验评价。

---

# Research Questions

## RQ1：Mechanism Attribution

**Can structural counterfactual interventions identify predictive and transferable mechanisms inside alpha factors?**

## RQ2：Search Efficiency

**Can counterfactual mechanism evidence improve evolutionary alpha search under a fixed evaluation budget?**

重点比较：

$$
\mathrm{Blind\ Evolution}
\quad\mathrm{vs.}\quad
\mathrm{Counterfactual\text{-}Guided\ Evolution}
$$

## RQ3：Pool Synergy

**Can pool-level counterfactual credit discover more complementary alpha factors?**

重点考察：

$$
C_{\mathrm{pool}}
$$

是否能够帮助构建具有更强组合表现和更低冗余度的 factor pool。

---

# 初步实验设计

主要 baseline：

- Random / GP Search；
- AlphaGen；
- AlphaSAGE；
- LLM-based Evolution；
- Counterfactual-Guided Evolution。

主要消融：

- w/o Counterfactual Diagnosis；
- w/o Pool-Level Credit；
- Blind / Random Crossover；
- w/o Mechanism Memory；
- w/o Pool-Aware Selection；
- w/o Parameter Refinement。

---

# 核心研究目标

本工作利用 counterfactual intervention 识别 factor 内部真正有效的结构机制，并将这些机制作为 evolutionary search 的结构先验。

最终形成：

$$
\mathrm{Factor\ Search}
\rightarrow
\mathrm{Mechanism\ Understanding}
\rightarrow
\mathrm{Mechanism\ Evolution}
$$

即从公式级搜索进一步转向：

> **基于结构证据的机制级 Alpha Discovery。**
