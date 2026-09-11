"""
NigeriaSupplyChainEnv
======================
Custom Gymnasium environment implementing the MDP formulated in Chapter 3
(Section 3.1.2) of the thesis:
 "Reinforcement Learning for Multi-Criteria Decision Making in a Supply
 Management" (Omole Daniel Oluwatosin, UNILAG Systems Engineering).

PRODUCT & CALIBRATION NOTE (added during post-submission review):
The modeled product is a standard 25kg packaged tomato crate, the unit
used in Nigeria's fresh-tomato trade (Mile 12 Market, Lagos, and
equivalent northern-production-zone markets such as Kano and Jos).
Every cost parameter below is derived from real, cited Nigerian data
rather than an arbitrary placeholder:

  - Unit value: N25,000/crate (Mile 12 Market survey baseline, 2026;
    Legit.ng market report).
  - Cost of capital: CBN Monetary Policy Rate = 26.5% p.a. (July 2026).
  - Storage cost: N20,000/sq.m/month for cold/air-conditioned dedicated
    space (SARA Lagos warehousing rates, 2025) -- cold storage is used
    rather than dry storage because tomatoes are perishable.
  - Freight: ~N10/tonne-km (informal Nigerian haulage-industry data
    point), applied to a 25kg crate over representative 300-800km
    inter-regional lanes, plus an allowance for informal checkpoint
    levies commonly reported on Nigerian interstate routes.
  - Spoilage: this is the parameter that most needed real grounding.
    A 2025 IITA/Michigan State University study (Nature Scientific
    Reports) found 33-50% of Nigerian tomato production fails to reach
    market. A UK FCDO postharvest-loss report found that reusable
    plastic crates cut Kano-to-south transit losses to ~10%, versus
    ~40% with traditional handling. This environment models spoilage
    as a genuine daily decay applied to on-hand DC inventory (product
    is physically removed from stock, not just cost-penalised), at a
    baseline rate calibrated to the ~10-15% good-conditions figure and
    an elevated rate during active port-congestion/capacity-reduction
    disruptions calibrated toward the ~40% poor-conditions figure.
  - Stockout penalty: a standard 1.4x-of-unit-value multiplier. This
    component is NOT independently Nigeria-sourced (no public Nigerian
    stockout-cost study was found); it follows conventional inventory-
    theory practice of penalising lost sales above their face value to
    reflect goodwill loss. This limitation is disclosed rather than
    presented as more precise than it is.

State space (29-dim, matches Table 3.2) -- unchanged in structure from
the original formulation:
  - DC inventory (5)      [0, 500]
  - CW inventory (1)      [0, 2000]
  - Demand forecast (7)   [0, 300]   rolling 7-day forecast
  - Transport costs (10)  per unit on the 10 cross-dock lanes
  - Remaining capacity (5)[0, 500]
  - Disruption flag (1)   {0,1}
  Total = 5+1+7+10+5+1 = 29

Action space (matches Section 3.1.2 "Action Space"):
  - o = [q1..q5], each in {0,25,50,75,100,125,150,175,200}  (9 levels, CW->DC)
  - c = [x_ij], 10 directed cross-dock lanes, each in {0,20,40,60,80,100} (6 levels)
  => MultiDiscrete([9]*5 + [6]*10)

Reward (extends Table 3.3 with a new spoilage/waste term):
  R_t = -(alpha*HC_t + beta*TC_t + theta*SP_t + omega*WC_t) + delta*SL_t + phi*XD_t
  alpha=1.0, beta=1.2, theta=2.5, omega=2.0 (dimensionless weights)
  delta=N8,000, phi=N4,000 (currency-bearing bonus terms)
  h=N150/unit/day (holding), p=N35,000/unit (stockout penalty)
  WC_t = value of crates lost to spoilage that day (unit value x spoiled units)

Disruptions (matches Table 3.8):
  Port congestion    p=0.10/day, lead time +2..+7 days, duration 3-14 days
  Fuel price shock   p=0.08/day, transport cost +30%..+60%, duration 5-10 days
  Capacity reduction p=0.05/day, DC capacity -20%..-50%, duration 1-5 days

Demand (matches Section 3.3.1):
  lambda_i^t = lambda_i^b * [1 + 0.4*sin(2*pi*t/365)] + eps_t
  lambda_i^b ~ U(50,200); eps_t = AR(1)-type noise calibrated to CV in [0.5,0.8]
  D_i^t ~ Poisson(max(lambda_i^t,1))
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

UNIT_VALUE = 25000.0        # N/crate (25kg tomato crate), Mile 12 Market baseline 2026
MPR = 0.265                 # CBN Monetary Policy Rate, July 2026
FREIGHT_NORM = 1200.0       # normalisation ceiling for freight observation (covers fuel-shock spikes)


class NigeriaSupplyChainEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, seed=None, phi=4000.0, n_dcs=5, demand_cv_range=(0.5, 0.8),
                 port_prob=0.10, fuel_prob=0.08, cap_prob=0.05, theta=2.5, gamma=0.99,
                 transport_cost_multiplier=1.0):
        super().__init__()
        self.n_dcs = n_dcs
        # Circulant-graph cross-dock lanes: each DC connects to its next two neighbours
        # (mod n_dcs). Reduces to the thesis's fixed 10-lane, 5-DC topology when n_dcs=5:
        # [(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,4),(3,0),(4,0),(4,1)].
        self.LANES = [(i, (i + k) % n_dcs) for i in range(n_dcs) for k in (1, 2)]
        self.horizon = 365
        self.rng = np.random.default_rng(seed)

        self.max_dc_inv = 500.0
        self.max_cw_inv = 2000.0
        self.max_cap = 500.0

        self.demand_cv_range = demand_cv_range
        self.port_prob = port_prob
        self.fuel_prob = fuel_prob
        self.cap_prob = cap_prob
        self.transport_cost_multiplier = transport_cost_multiplier

        self.order_levels = np.array([0, 25, 50, 75, 100, 125, 150, 175, 200], dtype=np.float64)
        self.xdock_levels = np.array([0, 20, 40, 60, 80, 100], dtype=np.float64)
        self.action_space = spaces.MultiDiscrete([9] * self.n_dcs + [6] * len(self.LANES))

        obs_dim = self.n_dcs + 1 + 7 + len(self.LANES) + self.n_dcs + 1
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        # --- reward weights (Table 3.3, tomato-crate NGN-calibrated) ---
        self.alpha, self.beta, self.theta, self.omega = 1.0, 1.2, theta, 2.0   # dimensionless weights
        self.delta, self.phi = 8000.0, phi             # currency-bearing bonus terms
        self.h = 150.0          # holding cost N/unit/day (26.5% MPR capital + real cold-storage rate)
        self.p = 35000.0        # stockout penalty N/unit (1.4x unit value, standard multiplier)
        self.unit_value = UNIT_VALUE
        self.gamma = gamma

        # --- spoilage parameters (real-data-calibrated decay) ---
        self.spoil_rate_normal = 0.023      # ~15% cumulative loss over 7 days, good handling (FCDO)
        self.spoil_rate_disrupted = 0.06    # ~40% cumulative loss over comparable period, poor handling (FCDO)

        self.reset(seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.lambda_base = self.rng.uniform(50, 200, size=self.n_dcs)
        self.cv_target = self.rng.uniform(self.demand_cv_range[0], self.demand_cv_range[1], size=self.n_dcs)
        self.ar_state = np.zeros(self.n_dcs)
        self.ar_phi = 0.6

        self.dc_inventory = self.rng.uniform(80, 200, size=self.n_dcs)
        self.cw_inventory = 1200.0
        self.dc_capacity = np.full(self.n_dcs, self.max_cap)

        self.cw_daily_replenish = float(np.sum(self.lambda_base) * 1.30)

        # Base transport cost per lane, NGN/unit, derived from ~N10/tonne-km x 25kg crate
        # over representative 300-800km lanes, plus informal checkpoint-levy allowance.
        self.base_freight = self.rng.uniform(325.0, 700.0, size=len(self.LANES)) * self.transport_cost_multiplier
        self.current_freight = self.base_freight.copy()

        self.port_active, self.port_days_left, self.extra_lead = False, 0, 0
        self.fuel_active, self.fuel_days_left, self.fuel_mult = False, 0, 1.0
        self.cap_active, self.cap_days_left, self.cap_mult = False, 0, 1.0

        self.pipeline = []
        self.demand_history = [self._expected_demand(0)]

        self._last_fill_rate = 1.0
        self._last_total_cost = 0.0

        return self._obs(), {}

    def _expected_demand(self, t):
        seasonal = 1 + 0.4 * np.sin(2 * np.pi * t / 365.0)
        return self.lambda_base * seasonal

    def _forecast(self):
        f = []
        for k in range(7):
            f.append(np.mean(self._expected_demand(self.t + k)))
        return np.array(f)

    def _obs(self):
        forecast = self._forecast()
        obs = np.concatenate([
            self.dc_inventory / self.max_dc_inv,
            [self.cw_inventory / self.max_cw_inv],
            forecast / 300.0,
            self.current_freight / FREIGHT_NORM,
            self.dc_capacity / self.max_cap,
            [1.0 if (self.port_active or self.fuel_active or self.cap_active) else 0.0],
        ]).astype(np.float32)
        return np.clip(obs, 0.0, 1.0)

    def _update_disruptions(self):
        if self.port_active:
            self.port_days_left -= 1
            if self.port_days_left <= 0:
                self.port_active = False
                self.extra_lead = 0
        elif self.rng.random() < self.port_prob:
            self.port_active = True
            self.port_days_left = self.rng.integers(3, 15)
            self.extra_lead = self.rng.integers(2, 8)

        if self.fuel_active:
            self.fuel_days_left -= 1
            if self.fuel_days_left <= 0:
                self.fuel_active = False
                self.fuel_mult = 1.0
                self.current_freight = self.base_freight.copy()
        elif self.rng.random() < self.fuel_prob:
            self.fuel_active = True
            self.fuel_days_left = self.rng.integers(5, 11)
            self.fuel_mult = self.rng.uniform(1.30, 1.60)
            self.current_freight = self.base_freight * self.fuel_mult

        if self.cap_active:
            self.cap_days_left -= 1
            if self.cap_days_left <= 0:
                self.cap_active = False
                self.cap_mult = 1.0
                self.dc_capacity = np.full(self.n_dcs, self.max_cap)
        elif self.rng.random() < self.cap_prob:
            self.cap_active = True
            self.cap_days_left = self.rng.integers(1, 6)
            self.cap_mult = self.rng.uniform(0.50, 0.80)
            self.dc_capacity = np.full(self.n_dcs, self.max_cap) * self.cap_mult

    def step(self, action):
        action = np.asarray(action)
        order_action = action[:self.n_dcs]
        xdock_action = action[self.n_dcs:]

        order_qty = self.order_levels[order_action]
        xdock_qty = self.xdock_levels[xdock_action]

        self._update_disruptions()

        self.cw_inventory = min(self.max_cw_inv, self.cw_inventory + self.cw_daily_replenish)

        total_requested = order_qty.sum()
        if total_requested > self.cw_inventory and total_requested > 0:
            scale = self.cw_inventory / total_requested
            allocated = order_qty * scale
        else:
            allocated = order_qty.copy()
        self.cw_inventory = max(0.0, self.cw_inventory - allocated.sum())

        base_lead = self.rng.integers(1, 6, size=self.n_dcs)
        lead_times = base_lead + (self.extra_lead if self.port_active else 0)
        for i in range(self.n_dcs):
            if allocated[i] > 0:
                self.pipeline.append({"dc": i, "units": allocated[i], "eta": int(lead_times[i])})

        xdock_bonus_triggered = np.zeros(self.n_dcs, dtype=bool)
        total_xdock_cost = 0.0
        for lane_idx, (src, dst) in enumerate(self.LANES):
            qty = xdock_qty[lane_idx]
            if qty <= 0:
                continue
            shippable = min(qty, self.dc_inventory[src])
            room = max(0.0, self.dc_capacity[dst] - self.dc_inventory[dst])
            shippable = min(shippable, room)
            if shippable > 0:
                self.dc_inventory[src] -= shippable
                self.dc_inventory[dst] += shippable
                total_xdock_cost += shippable * self.current_freight[lane_idx] * 0.5
                xdock_bonus_triggered[dst] = True

        arrived = np.zeros(self.n_dcs)
        still_pending = []
        for d in self.pipeline:
            d["eta"] -= 1
            if d["eta"] <= 0:
                arrived[d["dc"]] += d["units"]
            else:
                still_pending.append(d)
        self.pipeline = still_pending
        self.dc_inventory = np.clip(self.dc_inventory + arrived, 0, self.dc_capacity)

        # ---- Spoilage: real physical decay of on-hand DC inventory ----
        # Elevated during port congestion / capacity-reduction disruptions,
        # which represent the delayed-handling / poor-storage conditions
        # the FCDO report associates with higher tomato loss rates.
        spoil_rate = self.spoil_rate_disrupted if (self.port_active or self.cap_active) else self.spoil_rate_normal
        spoiled_units = self.dc_inventory * spoil_rate
        self.dc_inventory -= spoiled_units
        WC = np.sum(spoiled_units) * self.unit_value

        base_rate = self._expected_demand(self.t)
        self.ar_state = self.ar_phi * self.ar_state + self.rng.normal(0, 1, self.n_dcs)
        noisy_rate = base_rate * (1 + self.cv_target * 0.3 * self.ar_state)
        noisy_rate = np.clip(noisy_rate, 1, None)
        demand = self.rng.poisson(noisy_rate).astype(np.float64)

        served = np.minimum(demand, self.dc_inventory)
        stockout_units = np.maximum(0.0, demand - self.dc_inventory)
        self.dc_inventory -= served

        HC = np.sum(self.h * self.dc_inventory)
        TC = float(allocated.sum() and np.sum(allocated * np.mean(self.current_freight))) + total_xdock_cost
        SP = np.sum(self.p * stockout_units)
        fill_rate = served.sum() / demand.sum() if demand.sum() > 0 else 1.0
        SL = fill_rate
        XD = 1.0 if xdock_bonus_triggered.any() and stockout_units.sum() == 0 and total_xdock_cost > 0 else 0.0

        raw_reward = -(self.alpha * HC + self.beta * TC + self.theta * SP + self.omega * WC) + self.delta * SL + self.phi * XD
        total_cost_today = HC + TC + SP + WC  # for reporting (Chapter 4 tables), real NGN units
        reward = raw_reward / 100000.0  # scaled for PPO value-function stability

        self._last_fill_rate = fill_rate
        self._last_total_cost = total_cost_today

        self.t += 1
        terminated = self.t >= self.horizon
        truncated = False

        info = {
            "total_cost": total_cost_today,
            "fill_rate": fill_rate,
            "holding_cost": HC,
            "transport_cost": TC,
            "stockout_cost": SP,
            "waste_cost": WC,
            "spoiled_units": float(np.sum(spoiled_units)),
            "demand": demand.sum(),
            "served": served.sum(),
            "disruption_active": bool(self.port_active or self.fuel_active or self.cap_active),
        }
        return self._obs(), float(reward), terminated, truncated, info
