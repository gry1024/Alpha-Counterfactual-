"""Model instructions: free structural hypotheses, measured evidence, verifiable edits."""
from alphagen.data.expression import Operators, GetGreater, GetLess

SYSTEM = """You are researching structural mechanisms inside quantitative factor expressions.
Return ONE JSON object without Markdown fences or surrounding prose. Give concise, substantive
structural explanations, not private reasoning traces. Do not invent measurements or promise returns.

EXPRESSION LANGUAGE
Use only $open, $close, $high, $low, $volume, $vwap and these operators (name/argument count):
""" + ", ".join(f"{op.__name__}/{op.n_args()}" for op in dict.fromkeys([*Operators, GetGreater, GetLess])) + """
Use functional notation, not infix, Python, Qlib syntax, or named placeholders.
Example: Mul(Sub(0.0,TsDelta($close,5)),Div($volume,TsMean($volume,20))).
Scalar literals contain a decimal point (1.0); rolling windows/lags are integers (20).
Greater/GetGreater mean elementwise maximum; Less/GetLess mean minimum, not Boolean tests.
Ref(x,d) reads d days into the PAST; d>=0. TsDelta(x,d)=x[t]-x[t-d], d>=1.
Other rolling windows are >=2. TsDiv(x,w)=x[t]/mean(x,w).
TsPctChange compares the last and first observations within w days.
Respect the supplied node, depth and total lookback limits, including nested rolling operators.
Log, division and fractional powers may be numerically undefined. Prefer sensible domains;
never hide a new financial mechanism inside a purported numerical safeguard.

AST REFERENCES
Root path is []; unary/rolling operand is child 0; binary/pair-rolling inputs are children 0 and 1.
Windows are parameters, NOT child nodes. A path identifies one occurrence, even if text repeats.
Copy source subtrees and parent expressions exactly. Mechanism names are free descriptions,
not categories: use a complete economic or mathematical interpretation grounded in the actual AST.
A description is a hypothesis, not evidence that the factor works.

MEASUREMENTS
R is SIGNED daily cross-sectional RankICIR, not absolute IC. Higher R is better.
delta_cf=R(intervened factor)-R(parent): negative supports the removed structure in that context.
pool_credit=U(original pool)-U(pool with parent replaced): positive supports its pool value.
U ranks each factor, combines with equal weights, and evaluates the resulting signal.
Small effects, conflicting signs and coverage changes deserve caution; no sign is a universal rule.
Attribution is conditional on the exact parent, intervention baseline and historical pool.
Transferred or reparameterized mechanisms do NOT inherit measured credit.
"""


DIAGNOSE = """TASK: identify the main mechanisms of ONE parent and propose independent counterfactuals.

The context provides a parent expression, its AST, measurements, limits, a mechanism count budget,
and cf_enabled. Identify UP TO that many meaningful mechanisms; fewer is fine. There is no list of
allowed mechanism names. Avoid filling the budget with redundant raw features or arbitrary fragments.
Nested mechanisms may be meaningful, but their independently measured effects are not additive.
A root mechanism is allowed when it describes a complete transform or interaction.

Do NOT suppress a mechanism just because intervention is difficult. First identify it, then choose
one interpretable baseline or mark it unmeasurable. Do not design interventions to maximize credit.

If cf_enabled=true, for each mechanism provide ONE Remove/Neutralize proposal:
- remove: retain a featured proper descendant of this mechanism, removing a transform, wrapper
  or interaction. The replacement must be an exact existing descendant. Root unwrapping is allowed.
- neutralize: substitute a contextually justified baseline. Prefer an identity where available:
  additive/subtractive contribution -> 0.0; multiplicative contribution or denominator -> 1.0.
  These are examples, not an exhaustive list. A contextual baseline may use the mechanism's existing
  inputs, but must not add a new input feature or an unrelated predictive structure.
- Only the selected path will change; the rest of the parent stays identical. Each intervention
  starts from the original parent, never from another intervened version.
- Explain the specific effect being removed and the inputs/behavior retained. For example,
  TsCorr(x,y,w)->x compares the interaction with an x-only baseline; it does not isolate a unique,
  context-free "correlation contribution". Distinguish this from an algebraic identity.
- If every defensible baseline is degenerate or unidentifiable, return mode=null, replacement=null
  with a reason. A constant whole-factor baseline has undefined IC, not zero measured contribution.
- Changing reversal into a new momentum formula is evolution, not counterfactual ablation.

If cf_enabled=false, perform semantic decomposition only. Set mode and replacement to null.
Do not construct interventions or claim evidence. This is the blind-evolution ablation.

Return exactly this shape (repeat mechanism items as needed):
{"mechanisms":[
  {"path":[1],"description":"abnormal-volume confirmation",
   "subtree":"Div($volume,TsMean($volume,20))",
   "mode":"neutralize","replacement":"1.0",
   "reason":"Remove volume-dependent scaling while retaining the other multiplicative input."}
]}
The example is a format illustration, not a required mechanism or path. Use this parent's real AST.
Context:
"""


EVOLVE = """TASK: generate up to offspring_limit DISTINCT, substantive offspring as full expressions.

Use current_evidence and historical_memory to propose structural hypotheses. Read BOTH delta_cf and
pool_credit, their magnitudes, baseline and coverage. Action tags are suggestions, not permissions.
A mechanism may help its parent but hurt the pool, or vice versa. State the tradeoff behind your choice.
Unmeasured mechanisms are exploratory hypotheses, not supported discoveries.

CHOOSE THE SEARCH
Choose how many proposals use each operation; there are no per-operation quotas.
- mutation: select one current parent, keep one or more meaningful mechanisms intact, and rewrite
  the surrounding structure. Usually preserve well-supported mechanisms, but explain uncertainty.
- replacement: replace one complete diagnosed mechanism at path. All structure outside that path
  must remain identical. A strong mechanism can be replaced to test a justified alternative;
  its current benefit does not prove that no better alternative exists.
- crossover: choose TWO DIFFERENT current parents, select one or more mechanisms from EACH, and
  compose them into a full expression. Choose a fresh combination for each proposal where useful.
  The connecting operators and surrounding structure are yours to design; no fixed composition
  template is imposed. Prefer plausible complementarity over duplicating correlated transformations.

parents must reference the supplied current parents. keep must reference current_evidence exactly
by parent, path and subtree. Every declared kept subtree must literally occur in the child; do not
claim preservation after silently changing its window, sign or inputs. If a source occurs twice in
the keep list, it must appear twice in the child.
Historical memory supplies relevant successes, failures and conflicting examples. It can inspire
replacement or surrounding structure, but cannot revive an old complete factor or certify a child.
No measured numbers are requested in your output.

RANDOM-CROSSOVER ABLATION
If random_pairs is supplied, it assigns an independent random pair to each crossover proposal.
Consume these pairs IN ORDER of crossover appearances in children, starting with pair 0.
Use exactly that pair's parents and mechanisms for that crossover; you still design the composition.
Do not select pairs by their evidence or skip an inconvenient pair. If random_pairs is null,
choose combinations yourself using the dual evidence. An empty list means crossover is unavailable.

DEFAULT PARAMETERS AND REFINEMENT
First output a complete executable expression with concrete default parameters.
Window-only or scalar-only edits are refinement, not macro evolution.
Optionally provide refine, a list of at most two parameter positions IN THE NEW CHILD:
  window: path addresses a rolling operator; values must come from the supplied window_grid.
  constant: path addresses a scalar Constant; role is coefficient, exponent or threshold;
            values must come from the supplied constant_grid.
Omit values to use the configured grid. Program execution, not you, chooses the best values.
Select parameters that meaningfully change the signal. Do not tune identity/sign-construction
constants, tiny stability epsilons, outer additive offsets, or positive outer rescalings that
cannot change ranks. Parameters inside a preserved mechanism may be refined later, but that new
instance must not inherit the source mechanism's credit.
Omit refine or return [] to use the default of the first two window positions.
The program refines only the best default-parameter candidates within a small fixed trial budget.

OUTPUT
{"children":[
  {"operation":"mutation|replacement|crossover",
   "parents":["exact current parent expression"],
   "expression":"complete child expression",
   "keep":[{"parent":"exact current parent expression","path":[0],"subtree":"exact source subtree"}],
   "path":[1],
   "reason":"Specific structural hypothesis; cite the supplied evidence and any tradeoff.",
   "refine":[{"path":[0],"kind":"window","values":[5,10,20]},
             {"path":[1,1],"kind":"constant","role":"coefficient","values":[0.5,1.0,2.0]}]
}]}
path is required only for replacement; keep may be empty for replacement.
Use one parent for mutation/replacement and two for crossover. Use exact valid syntax, not the
schema placeholders. All accepted children will be evaluated; pool selection decides survival.
Context:
"""
