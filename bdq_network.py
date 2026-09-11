"""
Branching Dueling Q-Network (BDQ), following Tavakoli, Fatemi & Kormushev (2018),
"Action Branching Architectures for Deep Reinforcement Learning".

Standard DQN (including Stable-Baselines3's implementation) requires a single flat
Discrete action space. This environment's action space is MultiDiscrete([9]*5 + [6]*10)
-- 15 independent discrete action dimensions -- which a flat DQN cannot represent
without flattening to 9^5 * 6^10 ~= 3.57e12 joint actions (computationally impossible
for a Q-network output layer).

BDQ solves this by decomposing the joint action into independent per-dimension
"branches" sharing a common state-representation trunk. Each branch d has its own
advantage head A_d(s, a_d) of size n_d (the number of discrete levels for that
dimension), combined with a single shared state-value head V(s) via the dueling
aggregation:

    Q_d(s, a_d) = V(s) + A_d(s, a_d) - mean_{a'_d}( A_d(s, a'_d) )

Actions are selected independently per branch: a_d* = argmax_{a_d} Q_d(s, a_d).
This reduces the effective output size from a product (3.57e12) to a sum
(5*9 + 10*6 = 105 total logits), making the problem tractable while still allowing
every action dimension to be chosen conditionally on the full shared state.
"""

import numpy as np
import torch
import torch.nn as nn
import random
from collections import deque

BRANCH_SIZES = [9, 9, 9, 9, 9, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]  # 5 order dims (9 levels) + 10 xdock dims (6 levels)
N_BRANCHES = len(BRANCH_SIZES)
STATE_DIM = 29


class BranchingQNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, branch_sizes=BRANCH_SIZES, trunk=(256, 256)):
        super().__init__()
        layers = []
        last = state_dim
        for h in trunk:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        self.trunk = nn.Sequential(*layers)

        self.value_head = nn.Linear(last, 1)
        self.advantage_heads = nn.ModuleList([nn.Linear(last, n) for n in branch_sizes])

    def forward(self, x):
        feat = self.trunk(x)
        V = self.value_head(feat)  # (B, 1)
        Qs = []
        for head in self.advantage_heads:
            A = head(feat)  # (B, n_d)
            Q = V + A - A.mean(dim=1, keepdim=True)
            Qs.append(Q)
        return Qs  # list of (B, n_d) tensors, one per branch

    def act(self, obs, epsilon, device):
        if random.random() < epsilon:
            return np.array([random.randrange(n) for n in BRANCH_SIZES], dtype=np.int64)
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            Qs = self.forward(x)
            action = np.array([q.argmax(dim=1).item() for q in Qs], dtype=np.int64)
        return action


class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(s2, dtype=np.float32),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)
