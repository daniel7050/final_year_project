import json
import time
import numpy as np
import torch
from scipy.stats import wilcoxon, ttest_rel

from nigeria_env import NigeriaSupplyChainEnv
from baselines import run_episodes, eoq_policy_factory, sS_policy_factory, mcdm_policy_factory
from bdq_network import BranchingQNetwork

N_EVAL = 100

q_net = BranchingQNetwork()
ckpt = torch.load("bdq_nigeria_ckpt.pt", map_location="cpu")
q_net.load_state_dict(ckpt["q_net"])
q_net.eval()


def bdq_policy(env):
    obs = env._obs()
    return q_net.act(obs, epsilon=0.0, device="cpu")


print("Evaluating BDQ (100 episodes)...")
t0 = time.time()
res_bdq = run_episodes(bdq_policy, n_episodes=N_EVAL, seed_offset=5000)
print(f"  done in {time.time()-t0:.1f}s | mean cost={res_bdq['cost'].mean():.0f} fill={res_bdq['fill'].mean()*100:.1f}%")

# Reuse existing baseline + PPO results (same seeds, already computed)
with open("results.json") as f:
    orig = json.load(f)

res_eoq = {"cost": np.array(orig["raw"]["EOQ_cost"]), "fill": np.array(orig["raw"]["EOQ_fill"])}
res_sS = {"cost": np.array(orig["raw"]["sS_cost"]), "fill": np.array(orig["raw"]["sS_fill"])}
res_mcdm = {"cost": np.array(orig["raw"]["MCDM_cost"]), "fill": np.array(orig["raw"]["MCDM_fill"])}
res_ppo = {"cost": np.array(orig["raw"]["PPO_cost"]), "fill": np.array(orig["raw"]["PPO_fill"])}


def summarize(name, res):
    d = {
        "name": name,
        "mean_cost": float(np.mean(res["cost"])), "std_cost": float(np.std(res["cost"])),
        "mean_fill": float(np.mean(res["fill"])) * 100, "std_fill": float(np.std(res["fill"])) * 100,
    }
    if "hc" in res:
        d["mean_hc"] = float(np.mean(res["hc"]))
        d["mean_tc"] = float(np.mean(res["tc"]))
        d["mean_sp"] = float(np.mean(res["sp"]))
        d["mean_wc"] = float(np.mean(res.get("wc", np.zeros_like(res["hc"]))))
    return d


summary = {
    "EOQ": summarize("EOQ", res_eoq),
    "sS": summarize("(s,S)", res_sS),
    "MCDM": summarize("Static AHP-TOPSIS MCDM", res_mcdm),
    "PPO": summarize("Deep RL (PPO)", res_ppo),
    "BDQ": summarize("Deep RL (Branching DQN)", res_bdq),
}
eoq_cost = summary["EOQ"]["mean_cost"]
for k in summary:
    summary[k]["cost_vs_eoq_pct"] = 100 * (eoq_cost - summary[k]["mean_cost"]) / eoq_cost

stats_results = {}
for name, res in [("EOQ", res_eoq), ("sS", res_sS), ("MCDM", res_mcdm), ("PPO", res_ppo)]:
    try:
        w_stat, w_p = wilcoxon(res_bdq["cost"], res["cost"])
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    t_stat, t_p = ttest_rel(res_bdq["cost"], res["cost"])
    diff = res_bdq["cost"] - res["cost"]
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    stats_results[f"BDQ_vs_{name}"] = {
        "wilcoxon_p": float(w_p), "ttest_p": float(t_p), "cohens_d": float(cohens_d),
    }

RESULTS = {
    "summary": summary,
    "cost_stats": stats_results,
    "n_eval_episodes": N_EVAL,
    "raw": {
        "BDQ_cost": res_bdq["cost"].tolist(), "BDQ_fill": res_bdq["fill"].tolist(),
        "BDQ_hc": res_bdq["hc"].tolist(), "BDQ_tc": res_bdq["tc"].tolist(),
        "BDQ_sp": res_bdq["sp"].tolist(), "BDQ_wc": res_bdq["wc"].tolist(),
    }
}
with open("results_bdq.json", "w") as f:
    json.dump(RESULTS, f, indent=2)

print("\n=== SUMMARY (BDQ vs all) ===")
for k, v in summary.items():
    print(f"{v['name']:30s} cost=N{v['mean_cost']:>14,.0f} sd=N{v['std_cost']:>12,.0f}  fill={v['mean_fill']:5.1f}%  vs_EOQ={v['cost_vs_eoq_pct']:+.1f}%")
print("\n=== SIGNIFICANCE (BDQ vs others) ===")
for k, v in stats_results.items():
    print(f"{k}: Wilcoxon p={v['wilcoxon_p']:.2e}  t-test p={v['ttest_p']:.2e}  Cohen's d={v['cohens_d']:.2f}")
print("\nSaved results_bdq.json")
