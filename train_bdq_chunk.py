import os
import json
import time
import sys
import numpy as np
import torch
import torch.nn.functional as F

from nigeria_env import NigeriaSupplyChainEnv
from bdq_network import BranchingQNetwork, ReplayBuffer, N_BRANCHES, BRANCH_SIZES

SEED = 42
CHUNK_STEPS = 40_000
TARGET_STEPS = 320_000
CKPT_PATH = "bdq_nigeria_ckpt.pt"
BUFFER_PATH = "bdq_replay_buffer.pkl"
STATE_PATH = "bdq_train_state.json"

LR = 1e-4
BATCH_SIZE = 64
GAMMA = 0.99
TARGET_UPDATE_EVERY = 1000   # steps
TRAIN_EVERY = 4              # steps
BUFFER_CAPACITY = 100_000
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 200_000
LEARNING_STARTS = 2_000

device = torch.device("cpu")

env = NigeriaSupplyChainEnv(seed=SEED)

if os.path.exists(STATE_PATH):
    with open(STATE_PATH) as f:
        state = json.load(f)
else:
    state = {"steps_done": 0, "obs": None, "episode_step": 0}

q_net = BranchingQNetwork().to(device)
target_net = BranchingQNetwork().to(device)
optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)
buffer = ReplayBuffer(capacity=BUFFER_CAPACITY)

if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=device)
    q_net.load_state_dict(ckpt["q_net"])
    target_net.load_state_dict(ckpt["target_net"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"Resumed from checkpoint at {state['steps_done']} steps")
else:
    target_net.load_state_dict(q_net.state_dict())
    print("Starting fresh BDQ model")

if os.path.exists(BUFFER_PATH):
    import pickle
    with open(BUFFER_PATH, "rb") as f:
        buffer.buffer = pickle.load(f)
    print(f"Resumed replay buffer with {len(buffer)} transitions")

if state.get("obs") is not None:
    obs = np.array(state["obs"], dtype=np.float32)
else:
    obs, _ = env.reset(seed=SEED)

remaining = TARGET_STEPS - state["steps_done"]
this_chunk = min(CHUNK_STEPS, remaining)

if this_chunk <= 0:
    print("Training already complete.")
    sys.exit(0)

t0 = time.time()
losses = []
for i in range(this_chunk):
    global_step = state["steps_done"] + i
    eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS))

    action = q_net.act(obs, eps, device)
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    buffer.push(obs, action, reward, next_obs, float(done))
    obs = next_obs
    if done:
        obs, _ = env.reset()

    if len(buffer) >= LEARNING_STARTS and global_step % TRAIN_EVERY == 0:
        s, a, r, s2, d = buffer.sample(BATCH_SIZE)
        s_t = torch.as_tensor(s, device=device)
        a_t = torch.as_tensor(a, device=device)
        r_t = torch.as_tensor(r, device=device)
        s2_t = torch.as_tensor(s2, device=device)
        d_t = torch.as_tensor(d, device=device)

        Qs = q_net(s_t)
        with torch.no_grad():
            Qs_next_online = q_net(s2_t)
            Qs_next_target = target_net(s2_t)

        branch_losses = []
        for b in range(N_BRANCHES):
            q_taken = Qs[b].gather(1, a_t[:, b:b+1]).squeeze(1)
            next_action = Qs_next_online[b].argmax(dim=1, keepdim=True)  # Double DQN
            q_next = Qs_next_target[b].gather(1, next_action).squeeze(1)
            target = r_t + GAMMA * (1 - d_t) * q_next
            branch_losses.append(F.smooth_l1_loss(q_taken, target))
        loss = torch.stack(branch_losses).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
        optimizer.step()
        losses.append(loss.item())

    if global_step % TARGET_UPDATE_EVERY == 0:
        target_net.load_state_dict(q_net.state_dict())

elapsed = time.time() - t0

torch.save({
    "q_net": q_net.state_dict(),
    "target_net": target_net.state_dict(),
    "optimizer": optimizer.state_dict(),
}, CKPT_PATH)

import pickle
with open(BUFFER_PATH, "wb") as f:
    pickle.dump(buffer.buffer, f)

state["steps_done"] += this_chunk
state["obs"] = obs.tolist()
state["last_chunk_time"] = elapsed
state["last_mean_loss"] = float(np.mean(losses)) if losses else None
with open(STATE_PATH, "w") as f:
    json.dump(state, f)

print(f"Chunk done: {this_chunk} steps in {elapsed:.1f}s ({this_chunk/elapsed:.1f} sps), "
      f"mean_loss={state['last_mean_loss']}, eps={eps:.3f}")
print(f"TOTAL PROGRESS: {state['steps_done']}/{TARGET_STEPS}")
