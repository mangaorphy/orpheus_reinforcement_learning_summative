"""
DQN Training Script — Savanna Acoustic Threat Detection
=======================================================
Implements Deep Q-Network (DQN) using Stable-Baselines3.
Runs 10 hyperparameter combinations and saves results for comparison.

DQN Key Hyperparameters tuned:
  - learning_rate       : step size for gradient updates
  - buffer_size         : replay buffer capacity
  - learning_starts     : steps before training begins
  - batch_size          : minibatch size for updates
  - tau                 : soft update coefficient for target network
  - gamma               : discount factor
  - train_freq          : update frequency
  - target_update_interval: hard update interval
  - exploration_fraction  : fraction of training spent exploring
  - exploration_final_eps : final epsilon for ε-greedy
  - net_arch            : neural network architecture

Usage:
    python training/dqn_training.py
    python training/dqn_training.py --run best  (run only best config)
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import SavannaAcousticEnv


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "models", "dqn")
LOG_DIR     = os.path.join(os.path.dirname(__file__), "..", "logs", "dqn")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(LOG_DIR,     exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 100_000   # per run (increase to 300k for final submission)


# --------------------------------------------------------------------------- #
# Reward tracking callback
# --------------------------------------------------------------------------- #

class RewardLoggerCallback(BaseCallback):
    """Logs episode rewards during training."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self._ep_reward = 0.0
        self._ep_len    = 0

    def _on_step(self) -> bool:
        self._ep_reward += self.locals["rewards"][0]
        self._ep_len    += 1
        if self.locals["dones"][0]:
            self.episode_rewards.append(self._ep_reward)
            self.episode_lengths.append(self._ep_len)
            self._ep_reward = 0.0
            self._ep_len    = 0
        return True


# --------------------------------------------------------------------------- #
# 10 Hyperparameter configurations
# --------------------------------------------------------------------------- #

DQN_CONFIGS = [
    # Run 1 — Baseline (SB3 defaults adapted)
    {
        "name": "baseline",
        "learning_rate":            1e-4,
        "buffer_size":              50_000,
        "learning_starts":          1_000,
        "batch_size":               64,
        "tau":                      1.0,
        "gamma":                    0.99,
        "train_freq":               4,
        "target_update_interval":   1_000,
        "exploration_fraction":     0.2,
        "exploration_final_eps":    0.05,
        "policy_kwargs":            {"net_arch": [128, 128]},
        "notes": "SB3 default starting point",
    },
    # Run 2 — Higher LR, faster learning
    {
        "name": "high_lr",
        "learning_rate":            5e-4,
        "buffer_size":              50_000,
        "learning_starts":          1_000,
        "batch_size":               64,
        "tau":                      1.0,
        "gamma":                    0.99,
        "train_freq":               4,
        "target_update_interval":   1_000,
        "exploration_fraction":     0.2,
        "exploration_final_eps":    0.05,
        "policy_kwargs":            {"net_arch": [128, 128]},
        "notes": "Higher LR — faster but potentially unstable",
    },
    # Run 3 — Deeper network
    {
        "name": "deep_net",
        "learning_rate":            1e-4,
        "buffer_size":              50_000,
        "learning_starts":          1_000,
        "batch_size":               64,
        "tau":                      1.0,
        "gamma":                    0.99,
        "train_freq":               4,
        "target_update_interval":   1_000,
        "exploration_fraction":     0.2,
        "exploration_final_eps":    0.05,
        "policy_kwargs":            {"net_arch": [256, 256, 128]},
        "notes": "Deeper 3-layer network for complex obs space",
    },
    # Run 4 — Large replay buffer
    {
        "name": "large_buffer",
        "learning_rate":            1e-4,
        "buffer_size":              200_000,
        "learning_starts":          5_000,
        "batch_size":               128,
        "tau":                      1.0,
        "gamma":                    0.99,
        "train_freq":               4,
        "target_update_interval":   2_000,
        "exploration_fraction":     0.3,
        "exploration_final_eps":    0.05,
        "policy_kwargs":            {"net_arch": [128, 128]},
        "notes": "Large buffer for better sample diversity",
    },
    # Run 5 — Low gamma (myopic)
    {
        "name": "low_gamma",
        "learning_rate":            1e-4,
        "buffer_size":              50_000,
        "learning_starts":          1_000,
        "batch_size":               64,
        "tau":                      1.0,
        "gamma":                    0.90,
        "train_freq":               4,
        "target_update_interval":   1_000,
        "exploration_fraction":     0.2,
        "exploration_final_eps":    0.05,
        "policy_kwargs":            {"net_arch": [128, 128]},
        "notes": "Low discount — short-sighted agent",
    },
    # Run 6 — Soft target update (Polyak)
    {
        "name": "soft_update",
        "learning_rate":            1e-4,
        "buffer_size":              50_000,
        "learning_starts":          1_000,
        "batch_size":               64,
        "tau":                      0.01,
        "gamma":                    0.99,
        "train_freq":               4,
        "target_update_interval":   100,
        "exploration_fraction":     0.2,
        "exploration_final_eps":    0.05,
        "policy_kwargs":            {"net_arch": [128, 128]},
        "notes": "Polyak soft update for stable target network",
    },
    # Run 7 — Long exploration
    {
        "name": "long_explore",
        "learning_rate":            1e-4,
        "buffer_size":              100_000,
        "learning_starts":          2_000,
        "batch_size":               64,
        "tau":                      1.0,
        "gamma":                    0.99,
        "train_freq":               4,
        "target_update_interval":   1_000,
        "exploration_fraction":     0.5,
        "exploration_final_eps":    0.1,
        "policy_kwargs":            {"net_arch": [128, 128]},
        "notes": "Extended exploration for sparse reward env",
    },
    # Run 8 — Large batch size
    {
        "name": "large_batch",
        "learning_rate":            2e-4,
        "buffer_size":              100_000,
        "learning_starts":          2_000,
        "batch_size":               256,
        "tau":                      1.0,
        "gamma":                    0.99,
        "train_freq":               4,
        "target_update_interval":   1_000,
        "exploration_fraction":     0.2,
        "exploration_final_eps":    0.05,
        "policy_kwargs":            {"net_arch": [256, 256]},
        "notes": "Large batch + wider network",
    },
    # Run 9 — Frequent target update
    {
        "name": "fast_target",
        "learning_rate":            1e-4,
        "buffer_size":              50_000,
        "learning_starts":          1_000,
        "batch_size":               64,
        "tau":                      1.0,
        "gamma":                    0.99,
        "train_freq":               1,
        "target_update_interval":   500,
        "exploration_fraction":     0.15,
        "exploration_final_eps":    0.02,
        "policy_kwargs":            {"net_arch": [128, 128]},
        "notes": "Frequent updates — more aggressive convergence",
    },
    # Run 10 — Tuned best (based on observed performance)
    {
        "name": "best_tuned",
        "learning_rate":            2e-4,
        "buffer_size":              100_000,
        "learning_starts":          2_000,
        "batch_size":               128,
        "tau":                      0.05,
        "gamma":                    0.995,
        "train_freq":               4,
        "target_update_interval":   500,
        "exploration_fraction":     0.3,
        "exploration_final_eps":    0.03,
        "policy_kwargs":            {"net_arch": [256, 128, 64]},
        "notes": "Tuned config — high gamma, soft update, tapered explore",
    },
]


# --------------------------------------------------------------------------- #
# Training function
# --------------------------------------------------------------------------- #

def train_dqn(config: dict, run_id: int, total_timesteps: int = TOTAL_TIMESTEPS):
    """Train a single DQN run with the given hyperparameter config."""
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"  DQN Run {run_id:>2}/10 — {name}")
    print(f"  {config['notes']}")
    print(f"{'='*60}")

    # Environment
    def make_env():
        env = SavannaAcousticEnv(render_mode=None, seed=run_id * 100)
        env = Monitor(env)
        return env

    train_env = make_vec_env(make_env, n_envs=1)

    # Eval environment (separate, fixed seed)
    eval_env = Monitor(SavannaAcousticEnv(render_mode=None, seed=999))

    # Callbacks
    reward_cb = RewardLoggerCallback()
    eval_cb   = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, name),
        log_path=os.path.join(LOG_DIR, name),
        eval_freq=max(2000, total_timesteps // 20),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(10_000, total_timesteps // 5),
        save_path=os.path.join(MODEL_DIR, name, "checkpoints"),
        name_prefix="dqn_savanna",
    )

    # Extract policy_kwargs separately
    policy_kwargs = config.get("policy_kwargs", {})
    hp_keys = [
        "learning_rate", "buffer_size", "learning_starts", "batch_size",
        "tau", "gamma", "train_freq", "target_update_interval",
        "exploration_fraction", "exploration_final_eps",
    ]
    hp = {k: config[k] for k in hp_keys}

    model = DQN(
        policy          = "MlpPolicy",
        env             = train_env,
        verbose         = 1,
        tensorboard_log = os.path.join(LOG_DIR, "tb"),
        policy_kwargs   = policy_kwargs,
        seed            = run_id * 7,
        **hp,
    )

    t_start = time.time()
    model.learn(
        total_timesteps = total_timesteps,
        callback        = [reward_cb, eval_cb, ckpt_cb],
        tb_log_name     = f"dqn_{name}",
        progress_bar    = True,
    )
    elapsed = time.time() - t_start

    # Final evaluation
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )

    # Save final model
    final_path = os.path.join(MODEL_DIR, name, "final_model")
    model.save(final_path)

    # Compile results
    results = {
        "algorithm":        "DQN",
        "run_id":           run_id,
        "config_name":      name,
        "notes":            config["notes"],
        "hyperparameters":  {**hp, "policy_kwargs": policy_kwargs},
        "mean_reward":      float(mean_r),
        "std_reward":       float(std_r),
        "training_time_s":  float(elapsed),
        "total_timesteps":  total_timesteps,
        "episode_rewards":  reward_cb.episode_rewards[-50:],  # last 50
        "model_path":       final_path,
    }

    out_path = os.path.join(RESULTS_DIR, f"dqn_{name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✓ Run {run_id} complete")
    print(f"    Mean reward : {mean_r:.2f} ± {std_r:.2f}")
    print(f"    Train time  : {elapsed:.0f}s")
    print(f"    Saved to    : {final_path}")

    train_env.close()
    eval_env.close()
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train DQN on Savanna env")
    parser.add_argument("--run",   type=str, default="all",
                        help="'all' or config name (e.g. 'best_tuned')")
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS,
                        help="Total training timesteps per run")
    args = parser.parse_args()

    all_results = []

    if args.run == "all":
        for i, cfg in enumerate(DQN_CONFIGS, start=1):
            res = train_dqn(cfg, i, total_timesteps=args.steps)
            all_results.append(res)
    elif args.run == "best":
        cfg = next(c for c in DQN_CONFIGS if c["name"] == "best_tuned")
        res = train_dqn(cfg, 10, total_timesteps=args.steps)
        all_results.append(res)
    else:
        cfg = next((c for c in DQN_CONFIGS if c["name"] == args.run), None)
        if cfg is None:
            print(f"Unknown config '{args.run}'. Available:")
            for c in DQN_CONFIGS:
                print(f"  {c['name']}")
            sys.exit(1)
        idx = DQN_CONFIGS.index(cfg) + 1
        res = train_dqn(cfg, idx, total_timesteps=args.steps)
        all_results.append(res)

    # Summary table
    print("\n" + "="*70)
    print("  DQN HYPERPARAMETER TUNING SUMMARY")
    print("="*70)
    print(f"  {'Config':<18} {'Mean Reward':>12} {'Std':>8} {'Time(s)':>10}")
    print("  " + "-"*58)
    all_results.sort(key=lambda r: r["mean_reward"], reverse=True)
    for r in all_results:
        print(f"  {r['config_name']:<18} {r['mean_reward']:>12.2f} "
              f"{r['std_reward']:>8.2f} {r['training_time_s']:>10.0f}")
    print("="*70)
    print(f"\n  Best config: {all_results[0]['config_name']} "
          f"(reward={all_results[0]['mean_reward']:.2f})")

    # Save combined results
    combined_path = os.path.join(RESULTS_DIR, "dqn_all_results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Combined results saved to: {combined_path}")


if __name__ == "__main__":
    main()
