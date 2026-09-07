# 基于反事实逻辑推演的因子挖掘

## 研究动机

现有自动因子挖掘方法通常遵循：Search→Generation→Evaluation\.\\text\{Search\}\.

候选因子最终通常都会得到一个整体评价：f→R\(f\)，其中 R\(f\)可以是 IC、RankIC、ICIR 或组合收益等指标。

这种方式无法进一步回答：

> **这个因子为什么好？具体是哪一部分结构产生了有效的 Alpha？下一步应该修改哪里？**
> 
> 

例如，一个有效因子可能同时包含多个 feature、operator、时间窗口和子表达式。即使最终 IC 很高，现有方法也很难判断究竟是哪一个结构决策真正贡献了预测能力。

因此，可以认为当前因子挖掘存在两个相关问题：

1. **Credit Assignment 过于粗粒度**：factor\-level reward 难以分解到具体结构决策；

2. **Search Efficiency 较低**：由于不知道哪些结构值得保留或修改，后续搜索仍然需要大量试错。

本工作希望引入 **Structural Counterfactual Reasoning** 来解决这一问题。

---

## 核心 Idea

对于一个已经生成并完成评价的因子 ff，我们不只记录其整体 reward，而是进一步对其结构进行**最小反事实干预**：

fcf=T\(f,a\),f^\{cf\}=T\(f,a\),

其中 aa 表示一次结构编辑操作，例如：

- Feature：Return →\\rightarrow Volume；

- Operator：TSMean →\\rightarrow TSMax；

- Window：20 →\\rightarrow 10；

- Structure：Div →\\rightarrow Sub；

- 删除或替换某个 subtree。

然后比较原因子和反事实因子的表现：

Δcf\(f,a\)=R\(T\(f,a\)\)−R\(f\)\.\\Delta^\{cf\}\(f,a\) = R\(T\(f,a\)\)\-R\(f\)\.

其核心问题是：

> **如果当初采用另一种结构设计，这个因子的表现会发生什么变化？**
> 
> 

例如，一个因子的 RankIC 为 0\.05：

由此可以得到比整体 reward 更细粒度的信息：

- Return 可能是关键输入；

- TSMean 是较重要的 transformation；

- Rank 对当前因子影响较弱；

- 时间窗口仍存在优化空间。

因此，反事实干预能够将：

Factor\-level Reward\\text\{Factor\-level Reward\}

进一步转化为：

Structural Credit\.\\text\{Structural Credit\}\.

---

## 方法整体思路

整体方法可以形成如下闭环：

Generate→Counterfactual Intervention→Structural Credit→Value Prediction→Guided Search\\boxed\{ \\text\{Generate\} \\rightarrow \\text\{Counterfactual Intervention\} \\rightarrow \\text\{Structural Credit\} \\rightarrow \\text\{Value Prediction\} \\rightarrow \\text\{Guided Search\} \}

### Step 1：Factor Generation

首先通过现有搜索算法生成候选因子 ff，并通过真实市场数据获得：

R\(f\)\.R\(f\)\.

这一部分可以直接建立在 AlphaSAGE 等现有因子搜索框架上。

### Step 2：Structural Counterfactual Intervention

围绕较有潜力的因子构造若干局部反事实：

\{T\(f,a1\),T\(f,a2\),…,T\(f,aK\)\},\\\{T\(f,a\_1\),T\(f,a\_2\),\\ldots,T\(f,a\_K\)\\\},

并通过真实评价得到：

Δi=R\(T\(f,ai\)\)−R\(f\)\.\\Delta\_i = R\(T\(f,a\_i\)\)\-R\(f\)\.

关键是保持 **minimal intervention**：每次只改变一个主要结构决策，其余部分保持不变。

### Step 3：Structural Credit Assignment

利用不同 intervention 对 reward 的影响，分析：

- 哪些 feature 值得保留；

- 哪些 operator 是关键结构；

- 哪些时间尺度有效；

- 哪些 subtree 可能是冗余结构。

这样，每次因子评价不再只产生一个 scalar reward，而能够产生一组更加细粒度的结构反馈。

### Step 4：Counterfactual Value Estimation

如果所有可能的结构修改都实际进行 backtest，计算成本会很高。

因此进一步利用已经执行过的反事实样本：

\(f,a,Δcf\)\(f,a,\\Delta^\{cf\}\)

训练一个 counterfactual value model：

Qϕ\(f,a\)→Δcf\(f,a\),Q\_\\phi\(f,a\) \\rightarrow \\Delta^\{cf\}\(f,a\),

用于预测：

> 对当前因子执行某个尚未尝试的编辑动作，可能带来多大的 performance improvement。
> 
> 

模型真正需要学习的重点不一定是精确预测 Δ\\Delta，而可以是判断：

> **哪些编辑动作更值得优先尝试。**
> 
> 

### Step 5：Counterfactual\-Guided Search

对于当前因子的候选编辑动作，根据预测价值选择下一步搜索方向，例如：

a∗=arg⁡max⁡a\[Qϕ\(f,a\)\+βU\(f,a\)\+λD\(f,a\)\],a^\* = \\arg\\max\_a \\left\[ Q\_\\phi\(f,a\) \+ \\beta U\(f,a\) \+ \\lambda D\(f,a\) \\right\],

其中：

- QϕQ\_\\phi：预测的 counterfactual improvement；

- UU：预测不确定性，用于 exploration；

- DD：结构多样性。

这样，搜索从传统的：

> random / reward\-driven exploration
> 
> 

逐渐转变为：

> **counterfactual\-informed exploration**。
> 
> 

---

## LLM 在其中的作用

**LLM 负责提出“值得测试什么”，量化模型负责判断“什么真正有效”。**

具体来说，LLM 主要负责两个部分：

### Semantic Proposal

根据已有因子结构和金融含义，提出具有经济语义的修改方向。

例如原因子的假设是：

> “短期 momentum 在异常成交量确认后具有更强预测能力。”
> 
> 

LLM 可以提出：

- 去掉 volume confirmation；

- 将 momentum 改为 reversal；

- 保留原机制但改变时间尺度。

### Counterfactual Intervention Proposal

将这些语义假设进一步映射为具体的 expression edits。

而以下部分仍由算法和真实数据完成：

- factor encoding；

- backtest；

- counterfactual reward；

- value estimation；

- search policy；

- factor selection。

## Research Questions

目前可以将论文问题收敛为三个：

### RQ1：Structural Attribution

**Which structural components contribute to an alpha's predictive performance?**

### RQ2：Counterfactual Prediction

**Can observed structural interventions predict the value of unexplored modifications?**

### RQ3：Search Efficiency

**Can counterfactual feedback improve alpha discovery under a limited evaluation budget?**

## 初步实验方案

目前比较合适的方式是直接基于 **AlphaSAGE** 进行扩展。

主要 baseline 可以包括：

- Random / GP Search；

- RL\-based Alpha Mining；

- AlphaGen；

- AlphaSAGE；

- LLM\-based Alpha Search。

主要消融可以包括：

- w/o Counterfactual Intervention；

- w/o Counterfactual Value Model；

- Random Intervention；

- w/o LLM Semantic Intervention；

- w/o Uncertainty Exploration。

