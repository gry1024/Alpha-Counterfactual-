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
\mathrm{Mechanism\ Evidence}
\rightarrow
\mathrm{Mechanism\ Memory}
\rightarrow
\mathrm{Direct\ Offspring\ Generation}
\rightarrow
\mathrm{Parent\ Replacement}
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

机制可以对应局部 subtree，也可以覆盖整个因子。诊断不做编辑类型枚举：删除算子、调整窗口、替换输入与经济意义重写都属于同一类结构干预。例如把 `Mul(1.0, $close)` 整式替换为 `$close`，本身就是一次有意义的冗余诊断。

对 mechanism $m_i$ 构造反事实编辑：

$$
f_i'=T(f,m_i)
$$

其中每个编辑独立作用于原 parent，并尽量保持其余结构不变。重新评价得到：

$$
\Delta_i^{\mathrm{cf}}
=
|R(f_i')|-|R(f)|
$$

其中：

- $\Delta_i^{\mathrm{cf}}<0$：编辑后个体预测能力下降，该机制当前具有正贡献；
- $\Delta_i^{\mathrm{cf}}\approx0$：该机制贡献有限；
- $\Delta_i^{\mathrm{cf}}>0$：编辑后表现改善，该结构可被进一步重构。

这些量只是真实数据上的结构证据，不附带预设处置标签，也不限定后续演化操作：

> **诊断只给实测证据，不给结论；机制应被保留、替换还是重新组合，交由 LLM 结合证据与机制记忆自主判断。**

---

# 基础定义

## Factor Reward

对于因子 $f$，在每个交易日计算横截面 RankIC（$h$ 为前向收益跨度，默认 20）：

$$
IC_t(f)
=
\operatorname{Spearman}
\left(
f_t,\,
r_{t\rightarrow t+h}
\right)
$$

定义有符号的日均 RankIC 作为基础因子表现：

$$
R(f)
=
\frac{1}{T}
\sum_{t}
IC_t(f)
$$

个体质量以绝对值衡量，正负预测方向都参与后续预测：

$$
|R(f)|
$$

$|R(f)|$ 用于种子选择、子代合格判定与 $\Delta^{\mathrm{cf}}$；组合层与池层面则保留符号，由正负共存形成 hedge。本工作不以 RankICIR 替代 $R(f)$。

初始化阶段不做符号翻转：$Sub(0.0,f)$ 视为结构外层包装，是否添加由 LLM 在演化阶段依据机制证据自主决定；之后的反事实评价、子代评价与最终评价均按公式原符号执行。

---

## Signal Normalization 与 Cost

每个因子的横截面输出先做平均并列秩归一化（average tied rank），映射到 $[-0.5,0.5]$：

$$
\tilde f_{i,t}
=
\frac{\operatorname{rank}_t(f_{i,t})}{N_t-1}-0.5
$$

缺失值取中性 $0$，使组合保持固定等权。

signal cost 使用相邻交易日信号排序的变化作为 turnover proxy：

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

复杂度定义为因子 AST 的节点数（含 feature 与常数）：

$$
C(f)=\operatorname{nodes}(f)
$$

其中单层最外部方向包装 $Sub(0.0,f)$ 不计入，窗口长度不作为独立节点计数。

---

## Pool Utility

对于因子池：

$$
P=\{f_1,f_2,\ldots,f_K\}
$$

组合信号通过对各因子的归一化信号做 in-sample OLS 回归得到（与 `run_adaptive_combination.py` 同一形式）。把每个因子在交易日 $t$ 的截面信号 $\tilde f_{i,t}$ 与目标 $r_{t\to t+h}$ 一起视作回归样本，求最小二乘权重 $\mathbf{w}\in\mathbb{R}^K$，组合信号即为

$$
F_{P,t}
=
\sum_{i=1}^{K} w_i\,\tilde f_{i,t}
$$

OLS 在 $(T\cdot S)$ 个样本上单次拟合，权重包含正负号（不做非负截断），与 `run_adaptive_combination.py` 的逐日滚动回归保持同一形式。

定义：

$$
U(P)
=
R(F_P)
=
\frac{1}{T}
\sum_t
IC_t(F_P)
$$

与个体质量不同，$U(P)$ 保留符号：组合的泛化稳定性来自正负预测因子的 hedge。

组合效应对候选因子的响应由以下唯一的**组合层贡献量**描述。对参考成员 $f$ 与候选 $e$：

$$
C_{\mathrm{pool}}(e\mid f,P)
=
U\left(P_{-f}\cup\{e\}\right)-U(P),
\qquad
P_{-f}=P\setminus\{f\}
$$

$C_{\mathrm{pool}}>0$ 表示用 $e$ 替换 $f$ 后组合效用提升，$C_{\mathrm{pool}}<0$ 表示组合效用下降，即越大越好。参考成员不参与移除（纯加入候选）时约定 $P_{-f}=P$，退化为 $C_{\mathrm{pool}}(e\mid P)=U(P\cup\{e\})-U(P)$。

该量只有这一个定义，后文机制证据与因子替换打分都直接引用它，不另立等价公式：

- 机制证据（Step 3）：$f$ 为被诊断 parent，$e$ 为反事实 $f'$，$P$ 为轮初池；
- 父因子替换与因子选择打分：$f$ 为被替换的 parent，$e$ 为候选（含 parent 自身），$P$ 为当前池；初始化贪心选择中候选 $f\notin P$，退化为 $P_{-f}=P$。

---

## Factor Selection

Pool selection 统一考虑单因子质量、组合贡献、多样性和 turnover cost。各项**不做 min-max 归一化**，按其自然尺度直接做加权和；权重从最近一次完整 run 的实测分布推出，使各项在"中等表现"时贡献同量级：

$$
S(f\mid P)
=
\alpha\,|R(f)|
+
\beta\,C_{\mathrm{pool}}(f\mid P)
+
\gamma\,D(f,P)
+
\lambda\,\bigl(1-C_{\mathrm{cost}}(f)\bigr)
$$

四项均越大越好：

- $|R(f)|$：单因子有符号日均 RankIC 的绝对值，典型值 ~0.04。
- $C_{\mathrm{pool}}(f\mid P)$：替换后的 OLS 组合效用变化，典型值 ~0.002，可正可负。
- $D(f,P)$：与当前池的最大 |Spearman| 相关性之补，典型值 ~0.3。
- $1-C_{\mathrm{cost}}(f)$：相邻日信号排序稳定性，典型值 ~0.98（取值过于集中，几乎不区分因子，**实际默认权重为 0**）。

默认取

$$
\alpha=25,\quad
\beta=500,\quad
\gamma=3,\quad
\lambda=0
$$

使各项"中等表现"时贡献约为 1.0。超参数不再要求和为 1；其语义是"每个维度上每单位的物理贡献"。

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

上述 $C_{\mathrm{pool}}$ 与 $D$ 均在 $P_{-f}=P\setminus\{f\}$ 上计算，即始终**不包含候选自身**；候选 $f\notin P$ 时取 $P_{-f}=P$（初始化贪心选择），替换场景取被替换 parent 作为参考成员 $f$。

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
|\mathcal C_0|=150
$$

个候选种子，来自现有搜索算法与因子库的混合：Alpha158 30 + AlphaSAGE 50 + AlphaPROBE 50 + AlphaGen 20。

通过统一的 pool-aware selection 构建工作池：

$$
P_0
=
\operatorname{Select}_{50}(\mathcal C_0)
$$

初始化是**唯一**采用贪心 pool-aware selection 的环节，使用与父因子替换同一组 $(\alpha,\beta,\gamma,\lambda)$。之后整个 evolutionary process 始终维护：

$$
|P_t|=50
$$

落选种子不再参与后续搜索。

---

## Step 2：Counterfactual Diagnosis

每轮从当前工作池 $P_t$ 中选择固定数量的互异 parent：

$$
E_t\subset P_t,\qquad |E_t|=10
$$

直接对当前池做**无放回均匀随机抽样**，不依赖 RankIC 或访问次数；每个 parent 每被选中一次，访问数加一（用于诊断与去重，不参与选择）。

对于每个 $f\in E_t$，LLM 依据因子复杂度自主决定分解出多少个有意义的 mechanism：

$$
M(f)=\{m_1,m_2,\ldots,m_K\},\qquad K\le 5
$$

机制数量不是命令行参数，也不强制凑数；$K=0$ 表示该 parent 没有值得诊断的结构。

每个机制返回：

| 字段 | 含义 |
|---|---|
| `path` | mechanism 对应的 subtree 路径，`[]` 表示整式 |
| `description` | 机制的金融或数学语义 |
| `replacement` | 对该机制的反事实编辑 |
| `reason` | 提出该编辑的依据 |

不设置编辑类型枚举。修改窗口时，替换包含该窗口的 rolling subtree。程序负责检查路径、公式合法性、历史范围与可评价性。

---

## Step 3：Mechanism Evidence

对每个成功的编辑 $f'=T(f,m)$，在真实 Train 数据上保存三项实测证据。

编辑对单因子预测能力的作用：

$$
\Delta^{\mathrm{cf}}
=
|R(f')|-|R(f)|
$$

编辑对整体 pool utility 的作用直接引用基础定义中唯一的组合层贡献量：

$$
C_{\mathrm{pool}}(f'\mid f,P),\qquad P_{-f}=P\setminus\{f\}
$$

即用反事实 $f'$ 替换 parent $f$ 引起的组合效用变化，不在本节重复定义。

编辑前后信号排序的差异：

$$
\operatorname{dist}(f,f')
=
1-
\frac{1}{T}
\sum_t
\operatorname{Spearman}(f_t,f'_t)
$$

signal distance 的相关性**不取绝对值**，取值范围 $[0,2]$：接近 $0$ 提示排序信号等价，便于发现恒等算子；接近 $2$ 表示方向相反。它只在共同有效、非恒定的横截面上计算，缺失日期不参与日均。它不是形式上的结构等价证明，也不能保证样本外等价。

$\Delta^{\mathrm{cf}}<0$ 表示此次编辑使个体预测能力下降，$C_{\mathrm{pool}}>0$ 表示此次编辑提升组合质量；两者可能冲突。诊断不根据阈值生成决定性处理建议，由 LLM 结合证据自主判断。

正常诊断三项均为有限浮点数；被消融禁用的指标记为 `null`，不能当作 $0$。无法测量的编辑记入失败日志，不作为已测证据。

---

## Step 4：Mechanism Memory

Counterfactual diagnosis 的结果保存为结构化 Mechanism Memory：

| Factor | Path | Mechanism | CF Expression | $\Delta^{\mathrm{cf}}$ | $C_{\mathrm{pool}}$ | dist |
|---|---|---|---:|---:|---:|
| $f_1$ | `[1,1]` | short-term reversal | `-Delta(close,5)` | -0.021 | 0.014 | 0.08 |
| $f_1$ | `[1]` | volume confirmation | `volume/Mean(volume,20)` | -0.006 | 0.001 | 0.12 |
| $f_2$ | `[0]` | volatility normalization | `x/Std(ret,20)` | 0.004 | -0.001 | 0.35 |

Memory 记录 parent、对应 subtree、编辑内容与理由、完整反事实公式以及三项数值。

只有 $\Delta^{\mathrm{cf}}$ 测量成功的记录进入历史 memory。后续演化请求只为对应 parent 传入其自身的过往记录，不跨因子共享。机制证据依赖当时的 parent、干预与池，不能直接保证迁移后的效果。

---

## Step 5：Direct Offspring Generation

LLM 根据每个 parent 的 **mechanism 数量、规模与 factor 复杂度**，自主决定直接生成完整子代的数量：

$$
O(f),\qquad |O(f)|\in\{0,1,\ldots,5\}
$$

不设固定数量参数、不要求凑数；没有值得尝试的改进时返回空列表，由 parent 保留原槽位。默认每轮 10 个 parent，因此一轮生成 $0\sim50$ 个新假设：

$$
O_t=\bigcup_{f\in E_t}O(f)
$$

LLM 结合当前机制证据与历史 memory，自主提出去冗余、新颖且有效的改进：

$$
f'
=
\operatorname{Generate}
\left(
M(f),\,\operatorname{Memory}
\right)
$$

不限定操作类别、比例或必须保留的结构，也不另设细化或参数枚举阶段。

返回格式为：

```json
{"offspring": [{"expression": "完整公式", "description": "简短的证据与修改说明"}]}
```

每个子代的 description 用 1–2 句话说明依据哪些 mechanism 的实测证据（$\Delta^{\mathrm{cf}}$、$C_{\mathrm{pool}}$ 或 $\operatorname{dist}$），以及做了什么修改或探索、为什么。缺少证据时明确标为探索假设，不编造测量结果。

程序负责解析、验证、精确去重，并在 Train 上逐个评价；无效或重复子代直接跳过。

---

## Step 6：Parent Replacement

### 情形 1：等价去冗余

对每个子代 $f'$ 计算与 parent 的 signal distance：

$$
\operatorname{dist}(f,f')
=
1-\frac{1}{T}\sum_t
\operatorname{Spearman}(f_t,f'_t)
$$

当 signal distance 接近 $0$（程序用 $\le 1\times10^{-6}$ 判定），即排序信号等价，且节点数严格更少：

$$
C(f')<C(f)
$$

则直接替换。这覆盖删除无意义算子（如 $\mathrm{Mul}(1.0,f)$、$\mathrm{Add}(0.0,f)$）这类等价化简。

### 情形 2：组合得分提升

其余子代与 parent 一起放入候选集，按 $S(f\mid P)$ 打分。$P$ 为当前池，参考成员取被替换的 parent，因此参与比较的每个候选（含 parent 自身）都在同一个 $P_{-f}=P\setminus\{f\}$ 上计算 $C_{\mathrm{pool}}$ 与 $D$：

$$
S(f'\mid P)
>
S(f\mid P)
$$

得分严格高于 parent 的子代中，$S$ 最高者替换 parent：

$$
P
\leftarrow
\left(
P\setminus\{f\}
\right)
\cup
\{f^{*}\}
$$

单成员池时 $P_{-f}$ 为空，取 $D=1$。

没有满足任一情形的子代（包括 LLM 主动生成 $0$ 个、全部无效或重复）则保留 parent：

$$
P_t
\rightarrow
P_t\setminus\{f\}
\rightarrow
P_{t+1}
$$

同轮诊断基于轮初池；随后按 parent 顺序生成、评价、即时替换。情形 2 的 $S$ 使用当时实际的池（已包含本轮已接受的子代），避免同轮新子代互相高度重复。池容量天然保持为 50。

### 演化路径与可追溯性

给初始化成员与每个入池子代分配唯一 factor ID，并记录：

- 轮次、parent ID / 表达式、child ID / 表达式；
- 候选指标（reward、complexity、max correlation、eligible）及简短描述；
- 替换或保留结果。

入池子代另以 `child_description` 保存其证据与修改说明。`lineage.json` 记录完整演化路径，`pool_<round>.json` 中的 factor IDs 对应当前成员。未入池假设与模型输入输出仍保存在轮次日志中；表达式再次出现时分配新 ID，避免不同演化实例混淆。

---

# LLM 在其中的作用

LLM 主要负责：

### Mechanism Decomposition

将 factor 分解为少量具有明确金融或数学语义、并能映射到 expression subtree 的 mechanism。

### Counterfactual Proposal

为 mechanism 构造对应的结构干预：删除、替换、窗口调整或经济意义重写。

### Offspring Proposal

结合当前 factor 的机制证据与 Mechanism Memory，直接提出完整的改进子代。

以下部分由真实数据和程序完成：

- factor execution；
- reward calculation；
- counterfactual evaluation；
- pool utility；
- offspring validation / deduplication；
- factor selection 与 parent replacement。

基本原则为：

> **LLM 提出结构假设，真实市场数据提供结构证据。**

---

# Data Protocol

数据划分为：

$$
D_{\mathrm{train}}=2015\text{-}2021
$$

$$
D_{\mathrm{valid}}=2022
$$

$$
D_{\mathrm{test}}=2023\text{-}2026.04
$$

Factor search、counterfactual diagnosis 与所有入池判断只使用 Train；Train 内部删除跨边界 horizon 标签，禁止未来引用。

Validation / Test 切分由 `run_adaptive_combination.py` 按 `--train_end_year 2021` 内部处理，AlphaCF 不单独维护 valid 集合。

搜索完成后，最后一轮工作池直接交给原 `run_adaptive_combination.py`：

$$
P_G
\rightarrow
\mathrm{Adaptive\ Combination\ Backtest}
$$

不额外执行 final selection，也不在 Validation / Test 上做任何选择或调参。

---

# Research Questions

## RQ1：Mechanism Attribution

**Can structural counterfactual interventions identify predictive and transferable mechanisms inside alpha factors?**

## RQ2：Search Efficiency

**Can counterfactual mechanism evidence improve evolutionary alpha search under a fixed number of evolutionary rounds?**

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
- w/o Mechanism Memory；
- 初始化仅按 $|R|$ 选择（w/o Pool-Aware Selection）。

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