# Alpha158 baseline

Run from the repository root in the existing WSL environment:

```bash
python src/alpha158/run.py
python plot_profit_curve.py
```

Uses all 158 features from Qlib's `Alpha158DL.get_feature_config()`. Features are
normalized across stocks each day, as in the other baselines. There is no CF/LLM
search or LightGBM model. The shared `run_adaptive_combination.py` selects up to
20 factors and fits expanding historical OLS using only matured 20-day labels.
Consequently this is an online OLS baseline, not a frozen train-only model.

Defaults: history starts in 2015, Validation is 2022, Test is
2023-05-01 through 2026-04-30. IC/RankIC use 20-day forward returns; the daily
return series uses the same top-20% long-only, 1-day gross return calculation
as the existing backtest, with no transaction costs.

Outputs in `data/alpha158_logs/csi300/`: `alpha158.json`, `metrics.json`
(Validation/Test IC, ICIR, RIC, RICIR, return, Sharpe and drawdown fields),
`ret_s.npy` and `ret_s_dates.npy`. The plot registry uses this default directory.
Use `--output-dir` for experiments; update the registry to plot another output.
