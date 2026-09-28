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

Evaluation: R is SIGNED daily cross-sectional RankICIR; larger is better.
U is RankICIR of an equal-weight combination of rank-normalized factors.
delta_cf=R(ablated)-R(parent): negative means removal hurt the individual factor.
pool_credit=U(pool)-U(pool with parent replaced): positive means removal hurt the pool.
Near-zero effects are weak evidence. Conflicting signs reveal individual/pool tradeoffs.
A null score means unmeasured, not zero. Credits are conditional on the exact parent,
intervention and pool; they are not guarantees for transferred or modified mechanisms.
"""

PROMPT_DIAGNOSIS = """
Your task is to identify the useful structural units in this factor and propose independent
counterfactual ablations. The program will measure them; do not predict numerical credit.

1. Read the whole formula and its economic hypothesis. Identify up to {mechanism_count}
   coherent mechanisms: an interaction, a return signal, a volume transformation,
   smoothing, normalization, or another substantive unit supported by the actual formula.
   Do not fill the count with scalar leaves, sign scaffolding, epsilon guards or
   algebraically invariant edits such as removing centering inside a correlation.
2. Copy each mechanism's path from the supplied AST. Root is []; unary/rolling input is
   child 0; binary/pair-rolling inputs are children 0 and 1. Windows are not children.
   A mechanism can be nested or the whole factor. Avoid redundant overlapping diagnoses.
3. Propose one baseline at that path. "remove" keeps an exact proper descendant of the
   selected subtree. "neutralize" replaces the effect with an interpretable baseline:
   e.g. 0.0 for an additive contribution, 1.0 for a multiplicative contribution.
   Do not add new input features or replace the mechanism with a new alpha hypothesis.
4. Read the resulting WHOLE expression: it must still vary across stocks. For a root
   transform, keeping one of its inputs is often more informative than a constant.
   Each ablation starts from the original parent; other paths stay unchanged. Explain
   which behavior is removed and which is retained, including limitations of the comparison.
5. If no defensible nonconstant intervention exists, keep the mechanism with null mode
   and replacement. If cf_enabled is false, return semantic decomposition only, with
   all modes and replacements null. Check paths and syntax before returning the JSON.

Worked example (format and local editing, not a required factor):
Parent: Mul(Sub(Div(Ref($close,5),$close),1.0),Div($volume,TsMean($volume,20)))
{{"mechanisms":[{{"path":[1],"description":"Abnormal-volume scaling of a reversal signal",
"mode":"neutralize","replacement":"1.0",
"reason":"Remove volume-dependent amplification while retaining the five-day reversal."}}]}}
This changes only the right multiplicative input. The reversal is still a complete signal.

Given parent: {parent}
Parent R: {reward}
Available subtrees: {nodes}
cf_enabled: {cf_enabled}
Return only {{"mechanisms":[{{"path":[],"description":"...","mode":"remove|neutralize|null",
"replacement":"exact baseline expression or null","reason":"brief intervention rationale"}}]}}.
Use actual JSON null where appropriate, not the string "null".
"""

PROMPT_EVOLUTION = """
Your task is to generate {offspring_count} new factors from the parent, donor and their
counterfactual evidence. The purpose is to improve signed predictive quality and pool
complementarity through structural changes. You propose hypotheses; market data decides.

1. Read each parent mechanism with its baseline, delta_cf and pool_credit. Preserve useful
   behavior, simplify unsupported components and redesign harmful ones. If the signs
   conflict, explain the individual/pool tradeoff. With no measurements, state a hypothesis
   rather than claiming empirical support. Historical memory is evidence, not a recipe.
2. Use varied modification strategies across this batch, without forcing weak proposals:
   - mutation: keep a meaningful mechanism intact and redesign the surrounding structure;
   - replacement: replace one complete diagnosed mechanism with an economically motivated
     alternative, keeping the rest of that parent unchanged;
   - crossover: compose intact mechanisms from BOTH the current parent and supplied donor.
   If they are the same factor, do not claim crossover. In random-crossover mode, the donor
   is assigned randomly; use that donor rather than selecting another from memory.
3. Go beyond window changes, scalar tuning, sign flips and rank-equivalent wrappers.
   Try genuinely different representations of the hypothesis: interaction versus additive
   confirmation, price versus return behavior, conditional amplification versus risk
   normalization. A simpler formula may be a better hypothesis than a more complex one.
   These are examples, not a menu of required operators. Avoid arbitrary operator novelty.
4. Propose distinct full expressions with concrete default parameters. Prefer at most two
   different tunable windows; a one-day return lag may remain fixed. The program refines
   windows later. Do not copy a full parent, donor, existing pool factor or earlier child.
   Do not regenerate old complete factors merely because they appear in memory.
5. For each child give one concise explanation naming the retained/changed mechanisms,
   the supplied evidence motivating the change and the expected behavior to test. Do not
   claim higher measured R, lower correlation or profitability before evaluation.
6. Before responding, correct operator names, arity, scalar/window types, paths used in
   your own proposal, numerical domains and expression limits. Return only the corrected
   expressions, with matching operations and explanations in the same order.

Given parent and its evidence: {parent}
Given donor and its evidence: {donor}
Historical measured mechanisms: {historical_memory}
Existing pool and earlier children: {existing_expressions}
random_crossover: {random_crossover}
Expression limits: {limits}

Output format (all three arrays have {offspring_count} entries):
{{"expressions":["complete executable expression"],
"operations":["mutation|replacement|crossover"],
"explanations":["brief structural hypothesis grounded in the supplied evidence"]}}
Return one JSON object, without Markdown or any other text.
"""
