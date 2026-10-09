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

定义有符号的日均 RankIC 作为基础因子表现。$T$ 为标签有至少两个有限值且截面非恒定的 Train 日期数；在这些日期上，因子恒定、缺失或无法计算的 IC 按 0 计入，不能只平均有变化的日期。完全没有有效 IC 的因子仍拒绝：

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

$|R(f)|$ 用于种子选择、子代合格判定与 $\Delta^{\mathrm{cf}}$；组合层与池层面则保留符号，通过带符号的 OLS 系数利用不同预测方向。正负 IC 共存本身不保证 hedge 或泛化稳定性。本工作不以 RankICIR 替代 $R(f)$。

初始化阶段不做符号翻转：$Sub(0.0,f)$ 视为结构外层包装，是否添加由 LLM 在演化阶段依据机制证据自主决定；之后的反事实评价、子代评价与最终评价均按公式原符号执行。

---

## Signal Normalization 与 Cost

每个因子的横截面输出先做平均并列秩归一化（average tied rank），映射到 $[-0.5,0.5]$：

$$
\tilde f_{i,t}
=
\frac{\operatorname{rank}_t(f_{i,t})}{N_t-1}-0.5
$$

缺失值取中性 $0$，保留成员在 OLS 组合中的中性曝光，不逐股票重新分配权重。

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

全 Train 日志与快照的组合信号通过对各因子的归一化信号做 in-sample OLS 回归得到。把每个因子在交易日 $t$ 的截面信号 $\tilde f_{i,t}$ 与目标 $r_{t\to t+h}$ 一起视作回归样本，求最小二乘权重 $\mathbf{w}\in\mathbb{R}^K$，组合信号即为

$$
F_{P,t}
=
\sum_{i=1}^{K} w_i\,\tilde f_{i,t}
$$

OLS 在 $(T\cdot S)$ 个样本上单次拟合，权重包含正负号（不做非负截断），单成员池也通过 OLS 拟合方向。Train 上以 rank 信号组合，最终回测（`run_adaptive_combination.py`）使用 z-score 信号与逐日历史窗口拟合，两者均采用带符号的 OLS，但不视为完全相同的评价。

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

与个体质量不同，$U(P)$ 保留符号，越大表示组合预测与目标的排序关系越好；Train 组合改善不自动保证样本外改善。

pool_credit 与日志、快照共用完整 Train（2015–2021）上的 $U(P)$：原池与替换池分别拟合带符号 OLS，并在同一完整 Train 上评价，不做内部时间切分。

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
- 父因子替换打分：$f$ 为被替换的 parent，$e$ 为候选（含 parent 自身），$P$ 为当前池。

---

## Factor Selection

父因子替换统一考虑单因子质量、组合贡献、多样性和 turnover cost。各项**不做 min-max 归一化**，按其自然尺度直接做加权和；默认系数固定，不逐轮归一化或重新标定：

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

- $|R(f)|$：单因子有符号日均 RankIC 的绝对值。
- $C_{\mathrm{pool}}(f\mid P)$：替换后的 OLS 组合效用变化，可正可负。
- $D(f,P)$：与当前池的最大 |Spearman| 相关性之补。
- $1-C_{\mathrm{cost}}(f)$：相邻日信号排序稳定性，越大表示换手 proxy 越低。

原始尺度系数不要求和为 1；默认值固定，不逐轮更新，也不使用 Validation/Test 调参。每个候选的实际分量占比取决于其指标值，不宣称固定为历史重要性占比。

多样性定义为：

$$
\rho_{\mathrm{abs}}(f,g)
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
1-\max_{g\in P}\rho_{\mathrm{abs}}(f,g)
$$

上述 $C_{\mathrm{pool}}$ 与 $D$ 均在 $P_{-f}=P\setminus\{f\}$ 上计算，即始终**不包含候选自身**；取被替换 parent 作为参考成员 $f$。

---

# 方法整体设计

## Step 1：Initial Pool Construction

初始直接提供经过离线比较的 50 个精华种子：

$$
|P_0|=50
$$

种子仍来自 Alpha158、AlphaSAGE、AlphaPROBE、AlphaGen 的混合，各来源数量均低于原方案。`plot_profit_curve_start_pool.py` 离线比较多种配额和去相关强度：先覆盖不同的 OHLCV 经济结构，排除只改常量或窗口的同构表达式，以 Train 上的日均绝对 Spearman 限制信号冗余，再在合格方案中按 2022 年验证收益曲线择优。2021 年曲线只作诊断，2023 年以后的 Test 不参与选池。方案、配额和实测指标见 [初始池比较](plans/Start_Pool_Comparison.md)。

`start_pool.json` 即 $P_0$，训练仅解析、校验与评价全部成员，保持原顺序与符号，不再执行初始重选；数量不匹配、重复或不可用因子直接报错。之后整个 evolutionary process 始终维护：

$$
|P_t|=50
$$

离线研究中的未选因子不进入训练。

---

## Step 2：Counterfactual Diagnosis

每轮从当前工作池 $P_t$ 中选择固定数量的互异 parent：

$$
E_t\subset P_t,\qquad |E_t|=10
$$

轮初计算每个成员的删除贡献 $L_i=U(P_t)-U(P_t\setminus\{f_i\})$（在完整 Train 上重新拟合 OLS），按 $L_i$ 从小到大排序；parent 从低贡献集合与未选成员中混合抽样，不依赖单因子 RankIC 或访问次数。具体抽样规则、并发与日志见 plan 文件。

对于每个 $f\in E_t$，LLM 依据因子复杂度自主决定分解出多少个有意义的 mechanism：

$$
M(f)=\{m_1,m_2,\ldots,m_K\},\qquad K\le 5
$$

机制数量不是命令行参数，也不强制凑数；$K=0$ 表示该 parent 没有值得诊断的结构。

诊断先用 `understanding`（最多三句）写出完整公式的数学行为、待验证经济解释及边界条件；上一轮理解连同其对应公式传入，若为祖先则重新审视。每个机制的 `reason` 合并要检验的假设与定性预测，不另加预测字段。选择能区分不同解释的干预，预测不是测量结果。

每个机制返回：

| 字段 | 含义 |
|---|---|
| `path` | mechanism 对应的 subtree 路径，`[]` 表示整式 |
| `description` | 机制的金融或数学语义 |
| `replacement` | 对该机制的反事实编辑 |
| `reason` | 要检验的假设及对排序、覆盖或预测能力的定性预测 |

不设置编辑类型枚举。修改窗口时，替换包含该窗口的 rolling subtree。程序负责检查路径、公式合法性、历史范围与可评价性。诊断同时考虑冗余与质量改进；识别到冗余不阻止探索改善。默认最多 5 个 LLM 诊断并发，反事实测量与后续演化仍在主线程按父因子顺序执行，每次替换或保留决定后追加链历史。LLM 调用默认首次等待 600 秒、后续尝试等待 1200 秒，最多 5 次请求并使用指数退避；临时网络失败耗尽后保留该父因子并继续其他父因子和后续轮。网络重试、超时与日志策略见 plan 文件。

---

## Step 3：Mechanism Evidence

对每个成功的编辑 $f'=T(f,m)$，按原提案顺序编号 `m1`、`m2` 等（失败项不重编号），在真实 Train 数据上保存三项核心实测证据。

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
\operatorname{dist}(f,f')=1-q\,|\rho_{\mathrm{rank}}(f,f')|,
\qquad q=\frac{|M_f\cap M_{f'}|}{|M_f\cup M_{f'}|}
$$

$M$ 是完整 Train 有效观测掩码；$\rho_{\mathrm{rank}}$ 是完整归一化排名信号展平、缺失取中性 0 后的 cosine 相关度。距离范围 $[0,1]$，0 表示等价排序（含统一的整体反向），1 表示最不相干或没有有效重叠；双方全零时 cosine 取 1，仅一方全零时取 0，无有效并集时为 `null`。全局方向一致，不把逐日任意反向当作等价。

另保存两项解释性统计（不参与评分）：`coverage_change` 为有效掩码异或数量 / 并集数量；`common_rank_correlation` 为每日共同有效样本上重新排名后的有符号 Spearman 的有效日均值，无法计算时为 `null`。共同样本排序不变但覆盖改变，不等于完整信号等价。例如删除分母极小保护项可将零振幅样本的 `0` 变为 `NaN`。

真正合并等价公式时，还要求完整有效掩码相同，归一化排名以绝对容差 $10^{-6}$ 一致或整体反向一致。距离很小本身不授权删除分母保护项；它不是符号代数证明，也不能保证样本外等价。

$\Delta^{\mathrm{cf}}<0$ 表示此次编辑使个体预测能力下降，$C_{\mathrm{pool}}>0$ 表示此次编辑提升组合质量；两者可能冲突。诊断不根据阈值生成决定性处理建议，由 LLM 结合证据自主判断。

正常诊断的可测指标为有限浮点数；被消融禁用或相关性无法计算的指标记为 `null`，不能当作 $0$。无法评价的编辑记入失败日志。

---

## Step 4：Mechanism Memory

Counterfactual diagnosis 的结果保存为结构化 Mechanism Memory：

| Factor | Path | Mechanism | CF Expression | $\Delta^{\mathrm{cf}}$ | $C_{\mathrm{pool}}$ | dist |
|---|---|---|---:|---:|---:|
| $f_1$ | `[1,1]` | short-term reversal | `-Delta(close,5)` | -0.021 | 0.014 | 0.08 |
| $f_1$ | `[1]` | volume confirmation | `volume/Mean(volume,20)` | -0.006 | 0.001 | 0.12 |
| $f_2$ | `[0]` | volatility normalization | `x/Std(ret,20)` | 0.004 | -0.001 | 0.35 |

每个初始因子是一条独立演化链，以初始 factor ID 作为固定 `chain_id`；子代继承该 ID，公式变化不重置历史，不跨链共享。

Memory 按 `chain_id` 保存该链全部已完成演化，按发生顺序记录 parent 的 `understanding`、机制假设与证据、`updated_understanding`、所有子代提案（含无效或重复提案）、有效候选评价和最终替换/保留决定。没有机制、没有子代或保留 parent 的轮次也写入历史。

每次演化直接传入该链全部此前历史，不按当前公式过滤、不截断；本轮证据只放在当前 parent 上，决定完成后才追加历史。因此正常模式只有该链首次演化时历史为空；下一次诊断同时读取上一条历史的公式与 `updated_understanding`。`--no-memory` 消融显式传空历史及空的上一轮理解，但仍保存记录。机制证据依赖当时的 parent、干预与池，不能直接保证迁移后的效果。

Memory 只作为 LLM 提出进化子代的 context，不自动将反事实公式加入替换候选。正向与负向实测证据均保留；LLM 自主决定如何利用证据，包括是否复现某个反事实编辑。只有 LLM 返回并通过验证、去重的 offspring 参与既定父因子替换。

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

根据机制证据同时提出完整排名等价的简化子代与质量改善子代，发现冗余不阻止探索。等价组保留最简版本后统一竞争，不另设细化或参数枚举阶段。

在同一次演化调用中先返回 `updated_understanding`（最多三句）：对照诊断假设与实测证据，引用当前机制编号说明支持或推翻了什么、哪些结构承担什么作用、什么仍不确定。未测量时不能声称得到验证，实测信用不证明金融因果关系或迁移后有效。

返回格式为：

```json
{"updated_understanding": "基于证据修正后的理解", "offspring": [{"expression": "完整公式", "evidence_refs": ["m1"], "description": "使用的理解、修改及待验证假设"}]}
```

每个子代的 `description` 用 1–2 句话把理解连接到修改与新假设，`evidence_refs` 只能引用本次成功诊断的机制编号；负向证据也可引用。没有直接当前机制依据时用 `[]` 并明确是探索假设，历史证据可在描述中说明。没有子代也必须返回更新理解。程序校验字段与引用，理解、预测和引用均写入终端及既有日志/memory，不增加模型调用或评分项。

程序负责解析、验证、精确去重，并在 Train 上逐个评价；无效或重复子代直接跳过。

---

## Step 6：Parent Replacement

### 简化与改善统一竞争

子代须通过两个门槛：与 peers 的最大日均绝对 Spearman 相关性受限（peers 排除原 parent；相关性无法计算时按 1），且 $C_{\mathrm{pool}}>\varepsilon$；等价简化无例外；parent 自身不受门槛过滤。通过门槛的有效候选与 parent 一同竞争：按完整有效掩码和归一化排名一致性分组，每组保留 AST 节点最少的公式（复杂度相同则保持原顺序，parent 优先）。等价包括整体反向，不包括有效样本变化或逐日不同的方向翻转。

所有等价组代表统一计算 $S(f\mid P)$。参考成员仍是原 parent，所有候选在同一个 $P_{-f}=P\setminus\{f\}$ 上计算完整 Train 的 $C_{\mathrm{pool}}$ 与多样性 $D$。不同信号组只有得分严格高于 parent 所在组代表时才允许替换，取最高分者；得分相同时优先节点更少者、再按原顺序选择。

若没有不同信号组改善，parent 所在组保留最简代表；若它是更简单的子代则记录 equivalent_simplification，否则保留 parent。简化不再提前截断其他改善候选的竞争。每个 parent 最多替换一个原槽位，池容量不变；单成员池取 $D=1$。

同轮诊断基于轮初池；随后按 parent 顺序生成、评价、即时替换，评分使用已包含本轮先前子代的实际池。无有效子代时保留 parent。

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
D_{\mathrm{test}}=2023.05.01\text{-}2026.04.30
$$

使用原有 cn_data_rolling / us_data_qlib_latest，Test 为 2023-05-01–2026-04-30；保留 30 日未来缓冲以计算默认 20 日标签，缓冲不计入评价区间。

演化中的 factor search、counterfactual diagnosis 与所有入池判断只使用 Train；Train 内部删除跨边界 horizon 标签，禁止未来引用；pool_credit 在完整 Train 上拟合与评价，不做内部时间切分。初始池另在离线脚本中比较方案：Train 决定候选质量、结构和相关性，2022 Validation 仅用于合格方案的收益择优，因此 2022 不再是未见样本。所有演化消融共用这一冻结的初始池。

Validation / Test 切分由 `run_adaptive_combination.py` 按 `--train_end_year 2021` 内部处理，AlphaCME 不单独维护 valid 集合。

搜索完成后，最后一轮工作池直接交给原 `run_adaptive_combination.py`：

$$
P_G
\rightarrow
\mathrm{Adaptive\ Combination\ Backtest}
$$

搜索完成后不额外执行 final selection，也不在 Validation / Test 上重选最终池；Test 始终不参与初始池或演化调参。

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
- 初始池离线比较不同配额与去相关强度；所有演化消融使用同一冻结的 50 因子池。

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
