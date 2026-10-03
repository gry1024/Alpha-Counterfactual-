import json
from pathlib import Path
import numpy as np

ROOT = Path("/home/groy/cf")
NEW = ROOT / "data/cf_logs/20261003_011950_475394_csi300_0"

RUNS = {
    "AlphaCF_new": NEW / "ret_s.npy",
    "AlphaCF_old": ROOT / "data/cf_logs/20261002_162225_644686_csi300_0/ret_s.npy",
    "AlphaSAGE": ROOT / "data/gfn_logs/pool_50/gfn_gnn_csi300_50_2-0.01-1.0-1.0-1.0-0.3-linear-0.0/ret_s.npy",
    "AlphaPROBE": ROOT / "data/knowledge_logs/pool_50/kg_dag_and_bayesian_icir_and_mutl_new_no_decay_MiniMax-M3_5_csi300_0.5_7_50_0.9_50_20_0.006_True_True_False_True_0.7_0.1_0.05/ret_s.npy",
    "AlphaGen": ROOT / "data/ppo_logs/pool_20/ppo_csi300_20_0-20260905132532/ppo_csi300_20_0_20260905132532/ret_s.npy",
}

def stats(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    cum = np.cumsum(r)
    peak = np.maximum.accumulate(cum)
    mdd = float((peak - cum).max()) if n else float("nan")
    return dict(n=n, mean_bp=r.mean()*1e4, std_bp=r.std()*1e4,
                ann=float(r.mean()*252), sharpe=float(r.mean()/r.std()*np.sqrt(252)) if r.std() else float("nan"),
                final=float(cum[-1]), mdd=mdd)

print("=== backtest (test period) ===")
for name, p in RUNS.items():
    if not p.exists():
        print(f"{name:<12} MISSING {p}")
        continue
    s = stats(np.load(p))
    print(f"{name:<12} N={s['n']:<5} ann={s['ann']*100:>7.2f}%  sharpe={s['sharpe']:>6.2f}  final={s['final']*100:>7.2f}%  mdd={s['mdd']*100:>6.2f}%  mean={s['mean_bp']:>7.3f}bp")

print("\n=== pool evolution (new run) ===")
for i in range(11):
    f = NEW / f"pool_{i}.json"
    if not f.exists():
        continue
    d = json.loads(f.read_text())
    rs = [abs(m["reward"]) for m in d["metrics"]]
    raw = [m["reward"] for m in d["metrics"]]
    cs = [m["cost"] for m in d["metrics"]]
    print(f"round {i:>2}: utility={d['utility']:+.5f}  mean|R|={np.mean(rs):.5f}  mean_signedR={np.mean(raw):+.5f}  frac_pos={np.mean([x>0 for x in raw]):.2f}  mean|R|top10={np.mean(sorted(rs, reverse=True)[:10]):.5f}  mean_cost={np.mean(cs):.4f}")

# replacement details
print("\n=== replacement detail ===")
import collections
for step in range(1, 11):
    f = NEW / f"round_{step}.jsonl"
    if not f.exists():
        continue
    rep = 0; ret = 0; child_rewards = []; parent_rewards = []; n_off = 0
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("event") == "replacement":
            if e["outcome"] == "replaced":
                rep += 1
                child_rewards.append(abs(next(c["reward"] for c in e["candidates"] if c["expression"] == e["child"])))
                parent_rewards.append(abs(e["parent_reward"]))
            else:
                ret += 1
        elif e.get("event") == "proposal":
            n_off += 1
    if child_rewards:
        print(f"round {step:>2}: replaced={rep} retained={ret} proposals={n_off} "
              f"mean|R|(parent)={np.mean(parent_rewards):.4f} mean|R|(child)={np.mean(child_rewards):.4f} "
              f"delta={np.mean(np.array(child_rewards)-np.array(parent_rewards)):+.4f}")
    else:
        print(f"round {step:>2}: replaced={rep} retained={ret} proposals={n_off}")

# memory
mem = json.loads((NEW / "memory.json").read_text())
print(f"\n=== memory: {len(mem)} records ===")
dcf = [m["delta_cf"] for m in mem if m.get("delta_cf") is not None]
pcr = [m["pool_credit"] for m in mem if m.get("pool_credit") is not None]
sd = [m["signal_distance"] for m in mem if m.get("signal_distance") is not None]
if dcf:
    print(f"delta_cf: pos={np.mean([x>0 for x in dcf]):.2f} mean={np.mean(dcf):+.5f} | dist~0(<=1e-3)={np.mean([x<=1e-3 for x in sd]):.2f}")
if pcr:
    print(f"pool_credit: pos={np.mean([x>0 for x in pcr]):.2f} mean={np.mean(pcr):+.5f}")
if sd:
    print(f"signal_distance: mean={np.mean(sd):.3f} min={np.min(sd):.3f} max={np.max(sd):.3f}")
