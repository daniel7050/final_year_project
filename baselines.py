"""
Baseline heuristic policies matching thesis Section 3.3.3.
These act directly on the *unnormalised* internal state of NigeriaSupplyChainEnv
(read via env attributes after reset/step) so that they mimic what a real
warehouse manager would see, then translate their decision into the
discretised MultiDiscrete action bins used by the environment.
"""

import numpy as np
from nigeria_env import NigeriaSupplyChainEnv


def _nearest_bin(value, levels):
    return int(np.argmin(np.abs(levels - value)))


def run_episodes(policy_fn, n_episodes=100, seed_offset=0):
    """Runs `policy_fn(env)-> action` for n_episodes and returns per-episode
    total_cost, fill_rate, stockout_units, holding, transport, stockout, waste costs."""
    costs, fills, hc_list, tc_list, sp_list, wc_list = [], [], [], [], [], []
    for ep in range(n_episodes):
        env = NigeriaSupplyChainEnv(seed=seed_offset + ep)
        obs, _ = env.reset(seed=seed_offset + ep)
        done = False
        ep_cost, ep_hc, ep_tc, ep_sp, ep_wc = 0.0, 0.0, 0.0, 0.0, 0.0
        served_total, demand_total = 0.0, 0.0
        discount = 1.0
        while not done:
            action = policy_fn(env)
            obs, reward, term, trunc, info = env.step(action)
            ep_cost += (env.gamma ** env.t) * info["total_cost"]
            ep_hc += info["holding_cost"]
            ep_tc += info["transport_cost"]
            ep_sp += info["stockout_cost"]
            ep_wc += info.get("waste_cost", 0.0)
            served_total += info["served"]
            demand_total += info["demand"]
            done = term or trunc
        costs.append(ep_cost)
        fills.append(served_total / demand_total if demand_total > 0 else 1.0)
        hc_list.append(ep_hc)
        tc_list.append(ep_tc)
        sp_list.append(ep_sp)
        wc_list.append(ep_wc)
    return {
        "cost": np.array(costs), "fill": np.array(fills),
        "hc": np.array(hc_list), "tc": np.array(tc_list), "sp": np.array(sp_list), "wc": np.array(wc_list),
    }


# ---------------------------------------------------------------------
# Baseline 2: EOQ with safety stock (Section 3.3.3, Baseline 2)
# Q* = sqrt(2DS/H); SS = z*sigma_d*sqrt(L); order up when inv < SS, order Q*
# ---------------------------------------------------------------------
def eoq_policy_factory():
    S_cost = 18000.0  # N/order fixed ordering cost (partial informal-levy + admin estimate)
    z = 1.65

    def policy(env):
        order_action = np.zeros(env.n_dcs, dtype=int)
        for i in range(env.n_dcs):
            D_annual = env.lambda_base[i] * 365
            H = env.h * 365
            Q_star = np.sqrt(2 * D_annual * S_cost / H)
            sigma_d = env.lambda_base[i] * np.mean(env.cv_target)
            L_mean = 3.0
            safety_stock = z * sigma_d * np.sqrt(L_mean)
            if env.dc_inventory[i] < safety_stock:
                order_action[i] = _nearest_bin(Q_star, env.order_levels)
        xdock_action = np.zeros(len(env.LANES), dtype=int)  # EOQ baseline: no cross-docking
        return np.concatenate([order_action, xdock_action])
    return policy


# ---------------------------------------------------------------------
# Baseline 1: Optimal (s,S) policy, grid-searched per DC (Section 3.3.3, Baseline 1)
# ---------------------------------------------------------------------
def sS_policy_factory(s_val=60, S_val=200):
    def policy(env):
        order_action = np.zeros(env.n_dcs, dtype=int)
        for i in range(env.n_dcs):
            if env.dc_inventory[i] < s_val:
                target_order = S_val - env.dc_inventory[i]
                order_action[i] = _nearest_bin(target_order, env.order_levels)
        xdock_action = np.zeros(len(env.LANES), dtype=int)
        return np.concatenate([order_action, xdock_action])
    return policy


def grid_search_sS(n_eval_episodes=15):
    """Grid search s in {20,40,60,80,100}, S in {100,150,200,250,300} (Section 3.3.3)."""
    best_cost, best_params = np.inf, (60, 200)
    for s in [20, 40, 60, 80, 100]:
        for S in [100, 150, 200, 250, 300]:
            if S <= s:
                continue
            pol = sS_policy_factory(s, S)
            res = run_episodes(pol, n_episodes=n_eval_episodes, seed_offset=9000)
            mean_cost = res["cost"].mean()
            if mean_cost < best_cost:
                best_cost, best_params = mean_cost, (s, S)
    return best_params


# ---------------------------------------------------------------------
# Baseline 3: Static MCDM -- AHP-TOPSIS weighted heuristic (Section 3.3.3, Baseline 3)
# AHP weights: cost=0.5, service=0.35, resilience=0.15
# ---------------------------------------------------------------------
def mcdm_policy_factory():
    w_cost, w_service, w_resilience = 0.5, 0.35, 0.15

    def policy(env):
        order_action = np.zeros(env.n_dcs, dtype=int)
        for i in range(env.n_dcs):
            inv_ratio = env.dc_inventory[i] / env.max_dc_inv
            cap_ratio = env.dc_capacity[i] / env.max_cap
            disruption = 1.0 if (env.port_active or env.fuel_active) else 0.0
            # TOPSIS-like score: low inventory + high disruption risk -> higher order priority
            score = (w_service * (1 - inv_ratio) + w_resilience * disruption
                     - w_cost * (1 - cap_ratio))
            if score > 0.55:
                order_action[i] = 5  # order 125 units
            elif score > 0.30:
                order_action[i] = 2  # order 50 units
        # limited cross-docking: rebalance if a DC is critically low and neighbour is high
        xdock_action = np.zeros(len(env.LANES), dtype=int)
        for lane_idx, (src, dst) in enumerate(env.LANES):
            if env.dc_inventory[dst] < 0.2 * env.max_dc_inv and env.dc_inventory[src] > 0.6 * env.max_dc_inv:
                xdock_action[lane_idx] = 2  # move 40 units
        return np.concatenate([order_action, xdock_action])
    return policy
