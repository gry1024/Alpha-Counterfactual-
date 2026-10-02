"""Task templates following utils.prompt; operator meanings follow alphagen."""
from utils.prompt import PROMPT_HEAD

PROMPT_FEATURES_AND_OPERATORS = """
The available features and operators are listed below. Use functional notation exactly.
Features: $open/$high/$low/$close are daily prices; $vwap is volume-weighted average
price; $volume is traded volume. Prefer returns, ratios or ranks when combining scales.
Scalars are decimals (0.0, 1.0, 0.000001); windows/lags are concrete integers, not %d.
Do not use scientific notation, Python/infix syntax, placeholders or unnamed operators.

Unary: Abs(x), Log(x), SLog1p(x)=sign(x)*log(1+abs(x)), Inv(x)=1/x,
Sign(x), Rank(x)=ascending cross-sectional rank.
Binary: Add(x,y), Sub(x,y), Mul(x,y), Div(x,y), Pow(x,y).
GetGreater(x,y)=elementwise maximum; GetLess(x,y)=elementwise minimum.
Greater/Less are aliases for maximum/minimum, NOT Boolean comparisons.
Ref(x,d)=x[t-d], d>=0. TsDelta(x,d)=x[t]-x[t-d], d>=1.
Other rolling operators require d>=2:
TsMean/TsSum/TsStd/TsVar(x,d): rolling mean/sum/sample standard deviation/sample variance.
TsMin/TsMax/TsMed(x,d): rolling minimum/maximum/median.
TsMinMaxDiff(x,d)=max(x,d)-min(x,d); TsMaxDiff(x,d)=x-max(x,d);
TsMinDiff(x,d)=x-min(x,d); TsMad(x,d)=mean absolute deviation about the rolling mean.
TsIr(x,d)=mean(x,d)/std(x,d); TsDiv(x,d)=x/mean(x,d).
TsSkew/TsKurt(x,d): rolling skewness/excess kurtosis of the supplied input.
TsRank(x,d): descending time-series rank; a unique maximum has rank 1/d, minimum 1.
TsPctChange(x,d): last/first-1 over the d observations.
TsWMA(x,d): linearly weighted rolling mean; TsEMA(x,d): exponentially weighted mean.
TsCov/TsCorr(x,y,d): rolling covariance/correlation of two series.
Keep log inputs positive and denominators meaningful. Constants are not standalone alphas.
Nested windows consume cumulative history; respect the supplied expression limits.
Example: Div(Sub($close,$open),Add(Sub($high,$low),0.000001)).

Evaluation: R is SIGNED daily cross-sectional RankIC; individual quality is |R|, larger absolute
value is better. Sign is a property of the data, not the factor: keep the natural sign produced
by the operators and do not add a Sub(0.0, ...) wrapper to flip a factor unless the rewritten
expression would be genuinely negative on Train.
U is signed RankIC of an equal-weight combination of rank-normalized factors; positive and
negative factors coexist in the pool, so combination naturally hedges.
delta_cf=|R(counterfactual)|-|R(parent)|: negative means the edit hurt the individual factor.
pool_credit=U(pool)-U(pool with parent replaced): positive means the edit hurt the pool.
signal_distance=1-mean_daily_Spearman(parent,counterfactual), without absolute value.
Near zero suggests equivalent rankings, including redundant Mul(1.0,f), not proof of
identical syntax or future equivalence. Consider this together with reward and complexity. Conflicting signs reveal individual/pool tradeoffs.
A null score means unmeasured, not zero. Credits are conditional on the exact parent,
intervention and pool; they are not guarantees for transferred or modified mechanisms.
"""

PROMPT_DIAGNOSIS = """
Understand this factor and propose informative counterfactual edits. The program measures
the evidence; do not invent numerical scores or prescribe decisions for evolution.

Choose the number of mechanisms based on factor complexity, at most 5; do not pad the list.
An edit may remove a redundant operator, change a window, replace a subtree, introduce
another feature, or rewrite the economic hypothesis. Any edit useful for understanding
the factor is welcome. Explain what hypothesis the comparison tests in free text.
Copy a path from the supplied AST: root [], unary/rolling input [0], binary inputs [0]/[1].
To change a window, replace its rolling subtree with the revised expression.
For a whole-factor rewrite, use root []. Each edit starts from the original parent.
Only the selected subtree is replaced; the resulting whole factor must remain executable,
nonconstant and within the expression limits. An unchanged formula is not an intervention.
You may probe algebraic redundancy: for Mul(1.0,$close), path [] and replacement "$close"
test whether the multiplication contributes anything. Do not exclude such edits.
If no useful edit exists, return an empty mechanisms list. When measurement is disabled,
still propose edits, but no evidence will be measured. Check syntax and paths yourself.

Given parent: {parent}
Signed RankIC: {reward}
AST: {nodes}
Measurement enabled: {cf_enabled}
Expression limits: {limits}

Return only {{"mechanisms":[{{"path":[],"description":"...","replacement":"...",
"reason":"What this edit tests and why it helps understand the factor"}}]}}.
"""

PROMPT_EVOLUTION = """
Choose how many complete new child factors to generate, from 0 to 5, based on the
number and scope of this parent's mechanisms and the factor's complexity. Do not pad
proposals to reach a quota. If no worthwhile improvement is justified, return an empty
offspring list; the parent will be retained.
Use its measured mechanisms and historical evidence to form novel, effective hypotheses
and remove redundancy. You decide how to improve the factor; there are no required
operation categories, allocations, or rules forcing any mechanism to be retained.
You may rewrite a whole economic hypothesis, simplify an equivalent representation,
alter windows, or compose useful structures.
Use delta_cf, pool_credit and signal_distance together. Near-zero signal_distance can
justify eliminating redundant structure. Conflicting evidence is a tradeoff to reason
about, not a predetermined decision. Unmeasured evidence is not zero.

Each child is backtested on Train. Replacement follows two rules. (1) A child whose
signal_distance to its parent is near zero (equivalent rankings, e.g. deleting a redundant
Mul(1.0, f) or Add(0.0, f)) replaces the parent directly if it has strictly fewer AST nodes.
(2) Otherwise the parent and its children are ranked by a normalized composite score
S = weighted sum over |R|, C_pool, diversity and turnover cost, each
normalized to [0,1] (1 being best). A child replaces the parent only if its score is
strictly higher; the highest-scoring child wins. Complexity counts AST nodes, excluding a
single outer sign wrapper. Aim for diverse, economically meaningful improvements and remove
redundancy when the ranking is equivalent. Do not wrap a child in Sub(0.0, ...) just to
flip a sign that came out negative on Train.

For each child, return its executable final expression and a brief description (1-2
sentences): identify the mechanism evidence that motivated it (delta_cf, pool_credit
and/or signal_distance), and state what you changed or explored and why. Reference
actual supplied evidence; if it is absent, say this is an exploratory hypothesis.
Do not invent measurements or prescribe an operation category. Check syntax,
dimensional meaning, denominators and expression
limits. Avoid exact duplicates of the current pool or previously generated children.
Existing measured evidence need not transfer unchanged to a new context, so all children
will be evaluated.

Parent and evidence: {parent}
This factor's own past measured mechanisms: {historical_memory}
Existing expressions: {existing_expressions}
Expression limits: {limits}

Return only {{"offspring":[{{"expression":"...","description":"Brief evidence and modification/exploration rationale"}}]}}
with 0 to 5 entries, or {{"offspring":[]}}. Both fields are required nonempty strings.
"""
