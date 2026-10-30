# Frozen AlphaAgent baseline evaluation

Export the three factors already frozen for the CSI300 AlphaAgent Test into
`data/logs/alphaagent/csi300/seed0/pool.json`, with their original expressions,
source manifest and exact factor outputs. Do not select new factors using Test.

`src/alpha_cf/alphaagent_expression.py` converts the original formulas into prefix
operator strings. Pools declare `expression_engine=alphaagent_pandas_v1`; the
existing `run_adaptive_combination.py` loader recognizes them directly. The
expression implementation reads current Qlib OHLCV and preserves the actual
AlphaAgent operator semantics, including LOG(x)=log(1+x), adjusted EMA, average-tie
percentile ranks, cross-sectional medians, and partial rolling windows. It does
not require cached factor values or another backtest entry point. The previous
cached signals and pool are retained as audit artifacts; their wrapper is removed.

Also export all 15 distinct candidates from the first five dependent evolutionary
rounds to `rounds_1_5/pool.json`, preserving each candidate's round and code
provenance. This is an additional AlphaAgent-candidates + CF-integrator experiment,
not the official AlphaAgent selection procedure. Keep the original three-factor
pool and result alongside it; do not select the pooled candidates using Test.

For CSI500, select the best of the five candidate rounds by the saved 2022
validation annualized excess return after costs: round 5. All five official
replacement decisions were false; this explicit candidate-round selection is
user requested and does not claim official SOTA acceptance. Export the selected
three-factor pool and the complete 15-candidate pool separately under
`data/logs/alphaagent/csi500/seed0/`. Preserve TS_SUM, per-instrument DELAY and
C-style conditional semantics, and compare converted outputs against original
mining-period results before Test. Use the same CF dynamic settings for both.

Use the existing dynamic branch (without `--use_weights`), default expanding
history, 20-day labels and 21-day observation gap. Validation is 2022; Test is
2023-05-01 through 2026-04-30. This is an additional baseline evaluation and does
not change AlphaCREAM's frozen Train weights or its search protocol in `idea.md`.
