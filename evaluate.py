"""
Evaluate a trained PPO checkpoint against the classical baselines.

Reproduces the main thesis result (default, phi=4000.0, Table 4.1) or the
retuned-reward ablation (--phi 1500 --ckpt ppo_nigeria_v2_ckpt.zip --tag PPOv2,
Table 4.4), from a single script instead of two near-duplicate ones.

Baselines (EOQ, (s,S), MCDM) are computed once against phi=4000.0's environment
(reward weights don't affect these heuristics' cost/fill outcomes, since they
don't use the reward signal at all -- only PPO's training does) and cached to
results.json; subsequent runs against a different PPO checkpoint reuse the
cached baseline numbers rather than recomputing them, exactly as the original
evaluate_v2.py did.
"""
import json
import time
import argparse
import numpy as np
from scipy.stats import wilcoxon, ttest_rel
from stable_baselines3 import PPO

from nigeria_env import NigeriaSupplyChainEnv
from baselines import run_episodes, eoq_policy_factory, sS_policy_factory, mcdm_policy_factory, grid_search_sS

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default="ppo_nigeria_ckpt.zip")
parser.add_argument("--tag", type=str, default="PPO", help="Label for this run in the saved results, e.g. PPOv2")
parser.add_argument("--out", type=str, default=None, help="Output JSON path (default: results.json for tag=PPO, results_{tag}.json otherwise)")
parser.add_argument("--n-eval", type=int, default=100)
args = parser.parse_args()

N_EVAL = args.n_eval
OUT_PATH = args.out or ("results.json" if args.tag == "PPO" else f"results_{args.tag.lower()}.json")
BASELINE_CACHE = "results.json"

model = PPO.load(args.ckpt)

def ppo_policy(env):
    obs = env._obs()
    action, _ = model.predict(obs, deterministic=True)
    return action

try:
    cached = json.load(open(BASELINE_CACHE))
    res_eoq = {"cost": np.array(cached["raw"]["EOQ_cost"]), "fill": np.array(cached["raw"]["EOQ_fill"])}
    res_sS = {"cost": np.array(cached["raw"]["sS_cost"]), "fill": np.array(cached["raw"]["sS_fill"])}
    res_mcdm = {"cost": np.array(cached["raw"]["MCDM_cost"]), "fill": np.array(cached["raw"]["MCDM_fill"])}
    best_s, best_S = cached["sS_params"]["s"], cached["sS_params"]["S"]
    print(f"Reusing cached baseline results from {BASELINE_CACHE} (s,S)=({best_s},{best_S})")
except FileNotFoundError:
    print("No cached baselines found -- computing fresh (EOQ, (s,S) grid search, MCDM)...")
    best_s, best_S = grid_search_sS(n_eval_episodes=10)
    res_eoq = run_episodes(eoq_policy_factory(), n_episodes=N_EVAL, seed_offset=5000)
    res_sS = run_episodes(sS_policy_factory(best_s, best_S), n_episodes=N_EVAL, seed_offset=5000)
    res_mcdm = run_episodes(mcdm_policy_factory(), n_episodes=N_EVAL, seed_offset=5000)

print(f"Evaluating {args.tag} ({N_EVAL} episodes)...")
t0 = time.time()
res_ppo = run_episodes(ppo_policy, n_episodes=N_EVAL, seed_offset=5000)
print(f"  done in {time.time()-t0:.1f}s | mean cost={res_ppo['cost'].mean():.0f} fill={res_ppo['fill'].mean()*100:.1f}%")

if len(res_ppo["cost"]) != len(res_eoq["cost"]):
    raise SystemExit(
        f"--n-eval {N_EVAL} does not match the cached baseline episode count "
        f"({len(res_eoq['cost'])}). The paired significance tests below require equal-length, "
        f"same-seed samples. Either use --n-eval {len(res_eoq['cost'])} to match the cache, "
        f"or delete results.json to force baselines to recompute at your requested --n-eval."
    )


def summarize(name, res):
    d = {
        "name": name,
        "mean_cost": float(np.mean(res["cost"])), "std_cost": float(np.std(res["cost"])),
        "mean_fill": float(np.mean(res["fill"])) * 100, "std_fill": float(np.std(res["fill"])) * 100,
    }
    for k in ("hc", "tc", "sp", "wc"):
        if k in res:
            d[f"mean_{k}"] = float(np.mean(res[k]))
    return d


summary = {
    "EOQ": summarize("EOQ", res_eoq),
    "sS": summarize("(s,S)", res_sS),
    "MCDM": summarize("Static AHP-TOPSIS MCDM", res_mcdm),
    args.tag: summarize(f"Deep RL ({args.tag})", res_ppo),
}
eoq_cost = summary["EOQ"]["mean_cost"]
for k in summary:
    summary[k]["cost_vs_eoq_pct"] = 100 * (eoq_cost - summary[k]["mean_cost"]) / eoq_cost

stats_results = {}
for name, res in [("EOQ", res_eoq), ("sS", res_sS), ("MCDM", res_mcdm)]:
    try:
        w_stat, w_p = wilcoxon(res_ppo["cost"], res["cost"])
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    t_stat, t_p = ttest_rel(res_ppo["cost"], res["cost"])
    diff = res_ppo["cost"] - res["cost"]
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    stats_results[f"{args.tag}_vs_{name}"] = {
        "wilcoxon_p": float(w_p), "ttest_p": float(t_p), "cohens_d": float(cohens_d),
    }

RESULTS = {
    "summary": summary,
    "cost_stats": stats_results,
    "sS_params": {"s": best_s, "S": best_S},
    "n_eval_episodes": N_EVAL,
    "raw": {
        "EOQ_cost": res_eoq["cost"].tolist(), "EOQ_fill": res_eoq["fill"].tolist(),
        "sS_cost": res_sS["cost"].tolist(), "sS_fill": res_sS["fill"].tolist(),
        "MCDM_cost": res_mcdm["cost"].tolist(), "MCDM_fill": res_mcdm["fill"].tolist(),
        f"{args.tag}_cost": res_ppo["cost"].tolist(), f"{args.tag}_fill": res_ppo["fill"].tolist(),
    }
}
with open(OUT_PATH, "w") as f:
    json.dump(RESULTS, f, indent=2)

print(f"\n=== SUMMARY ({args.tag}) ===")
for k, v in summary.items():
    print(f"{v['name']:30s} cost=₦{v['mean_cost']:>10,.0f} ± {v['std_cost']:>8,.0f}   fill={v['mean_fill']:5.1f}% ± {v['std_fill']:4.1f}   Δcost vs EOQ={v['cost_vs_eoq_pct']:+.1f}%")
print(f"\n=== SIGNIFICANCE (cost, {args.tag} vs baseline) ===")
for k, v in stats_results.items():
    print(f"{k}: Wilcoxon p={v['wilcoxon_p']:.2e}  t-test p={v['ttest_p']:.2e}  Cohen's d={v['cohens_d']:.2f}")
print(f"\nSaved {OUT_PATH}")
