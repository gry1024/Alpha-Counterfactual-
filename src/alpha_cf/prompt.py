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
value is better. Evaluate each formula with its original sign; negative R is not inferior
merely because of its sign. There is no automatic sign flipping.
R averages over all Train dates with at least two finite, nonconstant cross-sectional labels.
On those dates, undefined factor IC (including constant or missing signals) contributes zero.
U is signed RankIC of the in-sample OLS combination of rank-normalized Train signals.
OLS coefficients retain both signs, including for a single factor; a negative-R factor
can contribute through a negative coefficient. Opposite IC signs alone do not imply hedging.
delta_cf=|R(counterfactual)|-|R(parent)|: negative means the edit hurt the individual factor.
pool_credit=U(pool with parent replaced)-U(pool).
Each pool fits signed OLS and evaluates RankIC on the complete Train dataset.
Positive credit means the edit improved this in-sample combination.
signal_distance=1-q*abs(cos(a,b)), where a/b are the complete flattened Train rank-normalized
signals with missing exposures filled by neutral zero; q is finite-mask intersection/union.
Its range is [0,1]: zero means equivalent rankings, including a consistent global reversal;
one means unrelated rankings or no finite overlap. Both-zero vectors have cosine 1,
one-zero vectors cosine 0, and no finite union gives null. Equivalence additionally requires
identical finite masks and complete normalized ranks equal (or globally reversed) within
absolute tolerance 0.000001; a small distance alone is not sufficient. Positive shifts and scales such as Add(f,0.0001),
Mul(Div(f,2.0),2.0) and Mul(f,1.0) can be redundant under this rank-based evaluation.
coverage_change is the fraction of finite-mask differences in the finite union.
common_rank_correlation is mean daily signed Spearman on jointly finite observations,
re-ranking both signals on that common universe; null means undefined.
Unchanged common ranks with changed coverage are not full-signal equivalence.
An outer Add(f,epsilon) can preserve order, while epsilon inside a denominator can
prevent 0/0; removing it can change coverage and other stocks' normalized ranks.
This is observed near-equivalence, not a proof of algebraic or future equality.
Consider this together with reward and complexity. Conflicting signs reveal individual/pool tradeoffs.
A null score means unmeasured, not zero. Credits are conditional on the exact parent,
intervention and pool; they are not guarantees for transferred or modified mechanisms.
"""

PROMPT_DIAGNOSIS = """
Explain your understanding in at most 3 short sentences: what the complete formula
computes, a tentative economic hypothesis, and important validity conditions or unknowns.
Previous understanding may describe an ancestor; reassess it against the current formula.
Propose informative counterfactuals that distinguish competing explanations of the factor.
In each reason, combine the hypothesis being tested and your qualitative prediction for
ranking/coverage or predictive quality. A contradiction should help revise the hypothesis.
These are predictions, not measurements; do not invent scores or prescribe evolution decisions.

Choose the number of mechanisms based on factor complexity, at most 5; do not pad the list.
Inspect the whole factor for redundant structure, and consider both simplifications
and quality improvements. Finding redundancy does not prevent proposing improvements.
Examples: Add(f,0.0001) -> f, Mul(Div(f,2.0),2.0) -> f, Mul(f,1.0) -> f.
Check the effect in the full expression: an internal shift may change a nonlinear
parent operator, and dropping Log or a denominator guard may change validity.
Also explore quality improvements through windows, inputs or economic hypotheses. Explain what each comparison tests.
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
Previous understanding: {previous_understanding}
AST: {nodes}
Measurement enabled: {cf_enabled}
Expression limits: {limits}

Return only {{"understanding":"Up to 3 short sentences",
"mechanisms":[{{"path":[],"description":"...","replacement":"...",
"reason":"Hypothesis being tested and predicted effect"}}]}}.
"""

PROMPT_EVOLUTION = """
First return updated_understanding in at most 3 short sentences. Compare the diagnostic
understanding and predictions with measured evidence, citing current mechanism IDs (m1, m2,
...). Explain what was supported or contradicted, which structures serve which roles, and
what remains uncertain. Distinguish changed common ranks from changed coverage; economic
stories are hypotheses, not causal proofs. With no measured evidence, explicitly retain
uncertainty rather than claim confirmation. Revise historical beliefs when current evidence conflicts.

Choose how many complete new child factors to generate, from 0 to 5, based on the
number and scope of this parent's mechanisms and the factor's complexity. Do not pad
proposals to reach a quota. If no worthwhile improvement is justified, return an empty
offspring list; the parent will be retained.
First look for redundant structure using the measured mechanisms and historical evidence.
Propose simpler equivalent children where useful, e.g. Add(f,0.0001),
Mul(Div(f,2.0),2.0), Mul(f,1.0) -> f. Simplification and quality improvement compete together;
you may also explore windows, inputs, whole economic hypotheses or composed structures
when redundancy exists.
Check near-equivalence in the complete factor, not merely a subtree.
Use delta_cf, pool_credit and signal_distance together. Near-zero signal_distance can
justify eliminating redundant structure. Conflicting evidence is a tradeoff to reason
about, not a predetermined decision. Unmeasured evidence is not zero.
Mechanism evidence and memory are context for your reasoning, not automatic replacement
candidates. You decide whether to reproduce a measured edit or propose a new structure.
Negative delta_cf is useful evidence that this intervention hurt individual quality;
do not discard negative evidence or mistake every diagnostic edit for an improvement.
Historical memory contains ALL previous evolutions of this seed's independent chain:
ancestor factors, diagnostic evidence, offspring proposals and selection/retention outcomes.
Use both successful and rejected attempts. An empty history means this chain has not
evolved before, unless memory is disabled. Historical credits belong to their original
factor and pool context; do not assume they carry unchanged to the current parent.

Each child is backtested on Train. A child can update the pool only if its maximum mean
absolute daily Spearman correlation with peers is <= {correlation_threshold}; peers exclude
the original parent. Every child must also have pool_credit > 0.000001.
Both hard gates apply to equivalent simplifications too, before grouping; there is no zero-credit
simplification exception. The parent remains the retention baseline without filtering.
The parent and children passing both gates are grouped by
complete normalized-rank equivalence (identical finite masks, equal ranks or a consistent
global reversal within absolute tolerance 0.000001). Each group keeps its simplest formula;
equal complexity preserves proposal order, with the parent first.
All group representatives compete together under the natural-scale score
S = alpha*|R| + beta*C_pool + gamma*diversity + cost_weight*(1-turnover),
without min-max normalization. Default alpha=50, beta=1000, gamma=2, cost_weight=2.
C_pool compares full-Train fitted and evaluated OLS combinations; diversity and credit use
the current pool excluding the original parent. A distinct-signal group must strictly beat
the parent group's representative score. The highest score wins; ties prefer fewer nodes,
then original order. Without an improvement the parent group keeps its simplest formula.
Simplification never bypasses stronger improvement candidates. Complexity counts AST nodes,
excluding a single outer sign wrapper. Negative Train R does not require a direction wrapper.

For each child, return its final expression, evidence_refs listing current mechanism IDs,
and a brief description (1-2 sentences): connect the updated understanding to the change
and state the new hypothesis being tested. References may include useful negative evidence.
Use [] for exploration without a direct current mechanism reference, and say it is exploratory.
Historical evidence can inform the description but must not be cited as a current mechanism ID.
Do not treat a positive parent intervention credit as proof that a composed child will work;
the new formula is still an untested hypothesis. Reference actual supplied measurements.
Do not invent measurements or prescribe an operation category. Check syntax,
dimensional meaning, denominators and expression
limits. Avoid exact duplicates of the current pool or previously generated children.
Existing measured evidence need not transfer unchanged to a new context, so all children
will be evaluated.

Parent and evidence: {parent}
This factor chain's complete past evolution history: {historical_memory}
Existing expressions: {existing_expressions}
Expression limits: {limits}

Return only {{"updated_understanding":"Up to 3 short sentences grounded in evidence",
"offspring":[{{"expression":"...","evidence_refs":["m1"],
"description":"Understanding used, change and hypothesis to test"}}]}}.
Allow 0 to 5 children; even with offspring=[], return updated_understanding.
All text fields must be nonempty strings; evidence_refs must be a list of current IDs or [].
"""
