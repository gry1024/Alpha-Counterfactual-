train：2010-2021
valid：2022
test：2023.01.01-2026.04.30

result using fixed version (from pull request in AlphaSAGE)
'''
ret_s_nonoverlap = ret_s[::args.label_days]
ret_sharpe = batch_sharpe_ratio(ret_s_nonoverlap, risk_free_rate).item()
ret_mdd = batch_max_drawdown(ret_s_nonoverlap).item()
'''
--- Parseable Format ---
Dataset            IC   IC_STD     ICIR      RIC  RIC_STD    RICIR      RET  RET_STD    RETIR   RET_SR  RET_MDD
Validation     0.0858   0.1250   0.6868   0.0968   0.1338   0.7236   0.0178   0.0029   0.0754  -0.1140   0.0085
Test           0.0300   0.1491   0.2011   0.0404   0.1530   0.2641   0.1139   0.0024   0.1753   1.8142   0.0098

result using unfixed version (Original AlphaSAGE):
'''
ret_sharpe = batch_sharpe_ratio(ret_s, risk_free_rate).item()
ret_mdd = batch_max_drawdown(ret_s).item()
'''
--- Parseable Format ---
Dataset            IC   IC_STD     ICIR      RIC  RIC_STD    RICIR      RET  RET_STD    RETIR   RET_SR  RET_MDD
Validation     0.0858   0.1250   0.6868   0.0968   0.1338   0.7236   0.0178   0.0029   0.0754   1.1976   0.1488
Test           0.0300   0.1491   0.2011   0.0404   0.1530   0.2641   0.1139   0.0024   0.1753   2.7829   0.1619

### gfn seed 2
python train_gfn.py         --seed 2         --instrument csi300         --pool_capacity 50         --log_freq 500         --update_freq 64         --n_episodes 10000         --encoder_type gnn         --entropy_coef 0.01         --entropy_temperature 1.0         --mask_dropout_prob 1.0         --ssl_weight 1.0         --nov_weight 0.3         --weight_decay_type linear         --final_weight_ratio 0.0
'''
ret_sharpe = batch_sharpe_ratio(ret_s, risk_free_rate).item() / (args.label_days ** 0.5)
ret_mdd = batch_max_drawdown(ret_s).item()
'''
初始pool：
--- Parseable Format ---
Dataset            IC   IC_STD     ICIR      RIC  RIC_STD    RICIR      RET  RET_STD    RETIR   RET_SR  RET_MDD
Validation     0.0858   0.1250   0.6868   0.0968   0.1338   0.7236   0.0178   0.0029   0.0754   0.2678   0.1488
Test           0.0300   0.1491   0.2011   0.0404   0.1530   0.2641   0.1139   0.0024   0.1753   0.6223   0.1619

cf
After edited:
--- Parseable Format ---
Dataset            IC   IC_STD     ICIR      RIC  RIC_STD    RICIR      RET  RET_STD    RETIR   RET_SR  RET_MDD
Validation     0.0897   0.1240   0.7234   0.0978   0.1292   0.7573   0.0037   0.0026   0.0172   0.0612   0.1593
Test           0.0289   0.1877   0.1539   0.0505   0.1920   0.2631   0.1020   0.0022   0.1742   0.6182   0.1226

### ppo
python train_ppo.py     --instruments csi300     --pool 20     --seed 0
----------------------------------------
| pool/                   |            |
|    best_ic_ret          | 0.1        |
|    eval_cnt             | 30448      |
|    significant          | 20         |
|    size                 | 20         |
| rollout/                |            |
|    ep_len_mean          | 4.3        |
|    ep_rew_mean          | 0.0635     |
| test/                   |            |
|    ic                   | 0.0329     |
|    rank_ic              | 0.0361     |
| time/                   |            |
|    fps                  | 12         |
|    iterations           | 98         |
|    time_elapsed         | 16494      |
|    total_timesteps      | 200704     |
| train/                  |            |
|    approx_kl            | 0.04883343 |
|    clip_fraction        | 0.379      |
|    clip_range           | 0.2        |
|    entropy_loss         | -1.85      |
|    explained_variance   | 0.0143     |
|    learning_rate        | 0.0003     |
|    loss                 | -0.0844    |
|    n_updates            | 970        |
|    policy_gradient_loss | -0.0436    |
|    value_loss           | 0.00442    |
----------------------------------------
python run_adaptive_combination.py \
    --expressions_file data/ppo_logs/pool_20/ppo_csi300_20_0-20260905132532/ppo_csi300_20_0_20260905132532/200704_steps_pool.json \
    --instruments csi300 \
    --cuda 0 \
    --train_end_year 2021 \
    --seed 0 \
    --use_weights True
--- Parseable Format ---
Dataset            IC   IC_STD     ICIR      RIC  RIC_STD    RICIR      RET  RET_STD    RETIR   RET_SR  RET_MDD
Test           0.0329   0.1454   0.2263   0.0361   0.1655   0.2181   0.1301   0.0027   0.1805   0.6406   0.1918