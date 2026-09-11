import os
import json
import time
import sys
import argparse
import torch
from stable_baselines3 import PPO
from nigeria_env import NigeriaSupplyChainEnv

SEED = 42
CHUNK_STEPS = 35_000
TARGET_STEPS = 320_000

parser = argparse.ArgumentParser(description="Train PPO on NigeriaSupplyChainEnv (chunked/resumable).")
parser.add_argument("--phi", type=float, default=4000.0,
                     help="Cross-docking bonus reward weight. Default 4000.0 reproduces the thesis's "
                          "main-results run; 1500.0 reproduces the retuned-reward ablation (Table 4.4).")
parser.add_argument("--ckpt", type=str, default=None,
                     help="Checkpoint filename. Defaults to ppo_nigeria_ckpt.zip for phi=4000.0 "
                          "(the main result) and ppo_nigeria_v2_ckpt.zip for phi=1500.0 (the ablation), "
                          "matching the filenames already used elsewhere in this thesis; any other phi "
                          "value must specify --ckpt explicitly.")
parser.add_argument("--target-steps", type=int, default=TARGET_STEPS)
args = parser.parse_args()

if args.ckpt:
    CKPT_PATH = args.ckpt
elif args.phi == 4000.0:
    CKPT_PATH = "ppo_nigeria_ckpt.zip"
elif args.phi == 1500.0:
    CKPT_PATH = "ppo_nigeria_v2_ckpt.zip"
else:
    parser.error(f"--ckpt is required for phi={args.phi} (no default filename for this value)")

STATE_PATH = CKPT_PATH.replace("_ckpt.zip", "_train_state.json")
TARGET_STEPS = args.target_steps

env = NigeriaSupplyChainEnv(seed=SEED, phi=args.phi)

if os.path.exists(STATE_PATH):
    with open(STATE_PATH) as f:
        state = json.load(f)
else:
    state = {"steps_done": 0}

policy_kwargs = dict(
    activation_fn=torch.nn.Tanh,
    net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
)

if os.path.exists(CKPT_PATH):
    model = PPO.load(CKPT_PATH, env=env)
    print(f"Resumed from checkpoint at {state['steps_done']} steps (phi={args.phi})")
else:
    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=64, n_epochs=10,
        gamma=0.99, policy_kwargs=policy_kwargs, seed=SEED, verbose=1,
    )
    print(f"Starting fresh PPO model (phi={args.phi})")

remaining = TARGET_STEPS - state["steps_done"]
this_chunk = min(CHUNK_STEPS, remaining)

if this_chunk <= 0:
    print("Training already complete.")
    sys.exit(0)

t0 = time.time()
model.learn(total_timesteps=this_chunk, reset_num_timesteps=False)
elapsed = time.time() - t0

model.save(CKPT_PATH)
state["steps_done"] += this_chunk
state["last_chunk_time"] = elapsed
with open(STATE_PATH, "w") as f:
    json.dump(state, f)

print(f"Chunk done: {this_chunk} steps in {elapsed:.1f}s ({this_chunk/elapsed:.1f} sps)")
print(f"TOTAL PROGRESS: {state['steps_done']}/{TARGET_STEPS}")
