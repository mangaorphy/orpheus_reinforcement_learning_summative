"""
Policy Gradient Training Script — Savanna Acoustic Threat Detection
===================================================================
Implements three policy-gradient algorithms using Stable-Baselines3:
  1. REINFORCE  (via SB3's custom REINFORCE wrapper — uses A2C with no critic)
  2. PPO        (Proximal Policy Optimization)
  3. A2C        (Advantage Actor-Critic)

Each algorithm runs 10 hyperparameter combinations.

Usage:
    python training/pg_training.py                    # train all three
    python training/pg_training.py --algo ppo         # PPO only
    python training/pg_training.py --algo a2c --run best
    python training/pg_training.py --algo reinforce
"""

import os
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import torch.nn as nn

from environment.custom_env import SavannaAcousticEnv


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR   = os.path.join(BASE_DIR, "models", "pg")
LOG_DIR     = os.path.join(BASE_DIR, "logs",   "pg")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
for d in [MODEL_DIR, LOG_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

TOTAL_TIMESTEPS = 100_000


# --------------------------------------------------------------------------- #
# Callback
# --------------------------------------------------------------------------- #

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True


# --------------------------------------------------------------------------- #
# REINFORCE — Custom implementation using SB3 architecture
# --------------------------------------------------------------------------- #

class REINFORCEPolicy(nn.Module):
    """Simple REINFORCE policy network (no value head)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.net    = nn.Sequential(*layers)
        self.action_dim = action_dim

    def forward(self, x):
        return self.net(x)

    def get_action(self, obs_tensor):
        logits = self.forward(obs_tensor)
        dist   = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, obs_tensor, actions):
        logits = self.forward(obs_tensor)
        dist   = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()


class REINFORCETrainer:
    """
    Pure REINFORCE (Monte-Carlo Policy Gradient) with optional baseline.

    Algorithm:
      For each episode:
        1. Collect full trajectory using current policy
        2. Compute discounted returns G_t for each step t
        3. Update policy: θ ← θ + α ∇_θ Σ_t log π_θ(a_t|s_t) G_t
    """

    def __init__(self, env, config: dict, seed: int = 0):
        self.env       = env
        self.config    = config
        self.seed      = seed
        self.device    = torch.device("cpu")

        obs_dim    = env.observation_space.shape[0]
        action_dim = env.action_space.n
        net_arch   = config.get("net_arch", (128, 128))

        self.policy    = REINFORCEPolicy(obs_dim, action_dim, net_arch).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.get("learning_rate", 1e-3)
        )
        self.gamma       = config.get("gamma",       0.99)
        self.use_baseline= config.get("use_baseline", True)
        self.entropy_coef= config.get("entropy_coef", 0.01)

        self.episode_rewards = []
        self.total_steps     = 0
        torch.manual_seed(seed)

    def collect_episode(self):
        obs, _ = self.env.reset()
        states, actions, rewards, log_probs = [], [], [], []
        done   = False

        while not done:
            obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, lp = self.policy.get_action(obs_t)
            obs, r, terminated, truncated, _ = self.env.step(action.item())
            done = terminated or truncated

            states.append(obs_t)
            actions.append(action)
            rewards.append(r)
            log_probs.append(lp)
            self.total_steps += 1

        return states, actions, rewards, log_probs

    def compute_returns(self, rewards):
        G, returns = 0.0, []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        if self.use_baseline:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, log_probs, returns, actions, states):
        log_probs_t = torch.stack(log_probs)
        _, entropy  = self.policy.evaluate(
            torch.cat(states), torch.stack(actions)
        )
        policy_loss = -(log_probs_t * returns).mean()
        entropy_loss = -entropy.mean()
        loss = policy_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()
        return loss.item()

    def train(self, total_timesteps: int):
        while self.total_steps < total_timesteps:
            states, actions, rewards, log_probs = self.collect_episode()
            ep_reward = sum(rewards)
            self.episode_rewards.append(ep_reward)
            returns = self.compute_returns(rewards)
            self.update(log_probs, returns, actions, states)

            if len(self.episode_rewards) % 20 == 0:
                avg = np.mean(self.episode_rewards[-20:])
                print(f"    Step {self.total_steps:>7} | "
                      f"Ep {len(self.episode_rewards):>4} | "
                      f"Avg reward (20ep): {avg:>8.2f}")

        return self.episode_rewards

    def evaluate(self, n_episodes=10):
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            ep_r = 0.0
            done = False
            while not done:
                obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _ = self.policy.get_action(obs_t)
                obs, r, term, trunc, _ = self.env.step(action.item())
                ep_r += r
                done = term or trunc
            rewards.append(ep_r)
        return float(np.mean(rewards)), float(np.std(rewards))

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state": self.policy.state_dict(),
            "config":       self.config,
        }, path + ".pt")
        return path + ".pt"


# --------------------------------------------------------------------------- #
# Hyperparameter configs — 10 per algorithm
# --------------------------------------------------------------------------- #

REINFORCE_CONFIGS = [
    {"name": "rf_baseline",     "learning_rate": 1e-3,  "gamma": 0.99,  "use_baseline": True,  "net_arch": (128,128), "entropy_coef": 0.01,  "notes": "Standard REINFORCE with baseline"},
    {"name": "rf_no_baseline",  "learning_rate": 1e-3,  "gamma": 0.99,  "use_baseline": False, "net_arch": (128,128), "entropy_coef": 0.01,  "notes": "REINFORCE without baseline (high variance)"},
    {"name": "rf_high_lr",      "learning_rate": 5e-3,  "gamma": 0.99,  "use_baseline": True,  "net_arch": (128,128), "entropy_coef": 0.01,  "notes": "High LR — faster but noisy updates"},
    {"name": "rf_low_lr",       "learning_rate": 1e-4,  "gamma": 0.99,  "use_baseline": True,  "net_arch": (128,128), "entropy_coef": 0.01,  "notes": "Low LR — slow stable convergence"},
    {"name": "rf_low_gamma",    "learning_rate": 1e-3,  "gamma": 0.90,  "use_baseline": True,  "net_arch": (128,128), "entropy_coef": 0.01,  "notes": "Low gamma — short-horizon planning"},
    {"name": "rf_deep",         "learning_rate": 1e-3,  "gamma": 0.99,  "use_baseline": True,  "net_arch": (256,256,128), "entropy_coef": 0.01,  "notes": "Deeper network"},
    {"name": "rf_high_entropy", "learning_rate": 1e-3,  "gamma": 0.99,  "use_baseline": True,  "net_arch": (128,128), "entropy_coef": 0.05,  "notes": "High entropy bonus — more exploration"},
    {"name": "rf_no_entropy",   "learning_rate": 1e-3,  "gamma": 0.99,  "use_baseline": True,  "net_arch": (128,128), "entropy_coef": 0.0,   "notes": "No entropy regularization"},
    {"name": "rf_wide_net",     "learning_rate": 2e-3,  "gamma": 0.99,  "use_baseline": True,  "net_arch": (512,256), "entropy_coef": 0.01,  "notes": "Wide first layer for large obs space"},
    {"name": "rf_best",         "learning_rate": 2e-3,  "gamma": 0.995, "use_baseline": True,  "net_arch": (256,128), "entropy_coef": 0.02,  "notes": "Best tuned: high gamma, moderate entropy"},
]

PPO_CONFIGS = [
    {"name": "ppo_baseline",    "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0,  "vf_coef": 0.5,  "policy_kwargs": {"net_arch": [128,128]}, "notes": "SB3 PPO defaults"},
    {"name": "ppo_small_steps", "learning_rate": 3e-4, "n_steps": 512,  "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0,  "vf_coef": 0.5,  "policy_kwargs": {"net_arch": [128,128]}, "notes": "Small rollout — more frequent updates"},
    {"name": "ppo_large_steps", "learning_rate": 3e-4, "n_steps": 4096, "batch_size": 128, "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0,  "vf_coef": 0.5,  "policy_kwargs": {"net_arch": [256,256]}, "notes": "Large rollout buffer"},
    {"name": "ppo_high_clip",   "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.4, "ent_coef": 0.0,  "vf_coef": 0.5,  "policy_kwargs": {"net_arch": [128,128]}, "notes": "High clip range — less conservative"},
    {"name": "ppo_low_clip",    "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.1, "ent_coef": 0.0,  "vf_coef": 0.5,  "policy_kwargs": {"net_arch": [128,128]}, "notes": "Low clip — very conservative updates"},
    {"name": "ppo_entropy",     "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,  "policy_kwargs": {"net_arch": [128,128]}, "notes": "Entropy bonus for exploration"},
    {"name": "ppo_high_vf",     "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.0,  "vf_coef": 1.0,  "policy_kwargs": {"net_arch": [128,128]}, "notes": "Higher value function weight"},
    {"name": "ppo_low_gae",     "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,  "n_epochs": 10, "gamma": 0.99,  "gae_lambda": 0.80, "clip_range": 0.2, "ent_coef": 0.0,  "vf_coef": 0.5,  "policy_kwargs": {"net_arch": [128,128]}, "notes": "Low GAE lambda — less bias correction"},
    {"name": "ppo_more_epochs", "learning_rate": 2e-4, "n_steps": 2048, "batch_size": 64,  "n_epochs": 20, "gamma": 0.99,  "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,  "policy_kwargs": {"net_arch": [256,128]}, "notes": "More SGD epochs per rollout"},
    {"name": "ppo_best",        "learning_rate": 2e-4, "n_steps": 2048, "batch_size": 128, "n_epochs": 15, "gamma": 0.995, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,  "policy_kwargs": {"net_arch": [256,128,64]}, "notes": "Best tuned PPO"},
]

A2C_CONFIGS = [
    {"name": "a2c_baseline",    "learning_rate": 7e-4, "n_steps": 5,   "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.0,   "vf_coef": 0.5,  "max_grad_norm": 0.5, "policy_kwargs": {"net_arch": [64,64]},    "notes": "SB3 A2C defaults"},
    {"name": "a2c_more_steps",  "learning_rate": 7e-4, "n_steps": 20,  "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.0,   "vf_coef": 0.5,  "max_grad_norm": 0.5, "policy_kwargs": {"net_arch": [128,128]},  "notes": "Longer rollouts for A2C"},
    {"name": "a2c_high_lr",     "learning_rate": 1e-3, "n_steps": 5,   "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.0,   "vf_coef": 0.5,  "max_grad_norm": 0.5, "policy_kwargs": {"net_arch": [128,128]},  "notes": "Higher LR"},
    {"name": "a2c_gae",         "learning_rate": 7e-4, "n_steps": 10,  "gamma": 0.99,  "gae_lambda": 0.95, "ent_coef": 0.0,   "vf_coef": 0.5,  "max_grad_norm": 0.5, "policy_kwargs": {"net_arch": [128,128]},  "notes": "GAE advantage estimation"},
    {"name": "a2c_entropy",     "learning_rate": 7e-4, "n_steps": 5,   "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.01,  "vf_coef": 0.5,  "max_grad_norm": 0.5, "policy_kwargs": {"net_arch": [128,128]},  "notes": "Entropy bonus"},
    {"name": "a2c_high_vf",     "learning_rate": 7e-4, "n_steps": 5,   "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.0,   "vf_coef": 1.0,  "max_grad_norm": 0.5, "policy_kwargs": {"net_arch": [128,128]},  "notes": "High value function coeff"},
    {"name": "a2c_low_gamma",   "learning_rate": 7e-4, "n_steps": 5,   "gamma": 0.90,  "gae_lambda": 1.0,  "ent_coef": 0.0,   "vf_coef": 0.5,  "max_grad_norm": 0.5, "policy_kwargs": {"net_arch": [128,128]},  "notes": "Low gamma — short horizon"},
    {"name": "a2c_deep",        "learning_rate": 5e-4, "n_steps": 10,  "gamma": 0.99,  "gae_lambda": 0.95, "ent_coef": 0.01,  "vf_coef": 0.5,  "max_grad_norm": 0.5, "policy_kwargs": {"net_arch": [256,256,128]}, "notes": "Deeper network"},
    {"name": "a2c_grad_clip",   "learning_rate": 7e-4, "n_steps": 5,   "gamma": 0.99,  "gae_lambda": 1.0,  "ent_coef": 0.0,   "vf_coef": 0.5,  "max_grad_norm": 0.1, "policy_kwargs": {"net_arch": [128,128]},  "notes": "Aggressive gradient clipping"},
    {"name": "a2c_best",        "learning_rate": 5e-4, "n_steps": 20,  "gamma": 0.995, "gae_lambda": 0.95, "ent_coef": 0.01,  "vf_coef": 0.5,  "max_grad_norm": 0.5, "policy_kwargs": {"net_arch": [256,128]},  "notes": "Best tuned A2C"},
]


# --------------------------------------------------------------------------- #
# Training functions
# --------------------------------------------------------------------------- #

def train_reinforce(config: dict, run_id: int, total_timesteps: int = TOTAL_TIMESTEPS):
    """Train REINFORCE with a given config."""
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"  REINFORCE Run {run_id:>2}/10 — {name}")
    print(f"  {config['notes']}")
    print(f"{'='*60}")

    train_env = Monitor(SavannaAcousticEnv(render_mode=None, seed=run_id * 100))
    eval_env  = Monitor(SavannaAcousticEnv(render_mode=None, seed=999))

    trainer  = REINFORCETrainer(train_env, config, seed=run_id * 7)
    t_start  = time.time()
    trainer.train(total_timesteps)
    elapsed  = time.time() - t_start

    mean_r, std_r = trainer.evaluate(eval_env)

    model_path = os.path.join(MODEL_DIR, "reinforce", name, "final_model")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    saved_path = trainer.save(model_path)

    results = {
        "algorithm":       "REINFORCE",
        "run_id":          run_id,
        "config_name":     name,
        "notes":           config["notes"],
        "hyperparameters": config,
        "mean_reward":     float(mean_r),
        "std_reward":      float(std_r),
        "training_time_s": float(elapsed),
        "total_timesteps": total_timesteps,
        "n_episodes":      len(trainer.episode_rewards),
        "episode_rewards": trainer.episode_rewards[-50:],
        "model_path":      saved_path,
    }

    out_path = os.path.join(RESULTS_DIR, f"reinforce_{name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Run {run_id} complete | Reward: {mean_r:.2f}±{std_r:.2f} | "
          f"Time: {elapsed:.0f}s | Episodes: {len(trainer.episode_rewards)}")

    train_env.close()
    eval_env.close()
    return results


def train_sb3_algo(algo_cls, algo_name: str, config: dict,
                   run_id: int, total_timesteps: int = TOTAL_TIMESTEPS):
    """Generic SB3 trainer for PPO and A2C."""
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"  {algo_name} Run {run_id:>2}/10 — {name}")
    print(f"  {config['notes']}")
    print(f"{'='*60}")

    def make_env():
        env = SavannaAcousticEnv(render_mode=None, seed=run_id * 100)
        return Monitor(env)

    train_env = make_vec_env(make_env, n_envs=1)
    eval_env  = Monitor(SavannaAcousticEnv(render_mode=None, seed=999))

    reward_cb = RewardLoggerCallback()
    eval_cb   = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, algo_name.lower(), name),
        log_path=os.path.join(LOG_DIR, algo_name.lower(), name),
        eval_freq=max(2000, total_timesteps // 20),
        n_eval_episodes=5,
        deterministic=True,
        verbose=0,
    )

    # Extract policy kwargs
    policy_kwargs = config.get("policy_kwargs", {})
    hp_keys_ppo = ["learning_rate","n_steps","batch_size","n_epochs",
                   "gamma","gae_lambda","clip_range","ent_coef","vf_coef"]
    hp_keys_a2c = ["learning_rate","n_steps","gamma","gae_lambda",
                   "ent_coef","vf_coef","max_grad_norm"]
    hp_keys = hp_keys_ppo if algo_name == "PPO" else hp_keys_a2c
    hp = {k: config[k] for k in hp_keys if k in config}

    model = algo_cls(
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
        callback        = [reward_cb, eval_cb],
        tb_log_name     = f"{algo_name.lower()}_{name}",
        progress_bar    = True,
    )
    elapsed = time.time() - t_start

    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=10, deterministic=True
    )

    final_path = os.path.join(MODEL_DIR, algo_name.lower(), name, "final_model")
    model.save(final_path)

    results = {
        "algorithm":       algo_name,
        "run_id":          run_id,
        "config_name":     name,
        "notes":           config["notes"],
        "hyperparameters": {**hp, "policy_kwargs": policy_kwargs},
        "mean_reward":     float(mean_r),
        "std_reward":      float(std_r),
        "training_time_s": float(elapsed),
        "total_timesteps": total_timesteps,
        "episode_rewards": reward_cb.episode_rewards[-50:],
        "model_path":      final_path,
    }

    out_path = os.path.join(RESULTS_DIR, f"{algo_name.lower()}_{name}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  ✓ Run {run_id} complete | Reward: {mean_r:.2f}±{std_r:.2f} | "
          f"Time: {elapsed:.0f}s")

    train_env.close()
    eval_env.close()
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def print_summary(results: list, algo_name: str):
    """Print sorted results table."""
    print(f"\n{'='*70}")
    print(f"  {algo_name} HYPERPARAMETER TUNING SUMMARY")
    print("="*70)
    print(f"  {'Config':<22} {'Mean Reward':>12} {'Std':>8} {'Time(s)':>10}")
    print("  " + "-"*56)
    results.sort(key=lambda r: r["mean_reward"], reverse=True)
    for r in results:
        print(f"  {r['config_name']:<22} {r['mean_reward']:>12.2f} "
              f"{r['std_reward']:>8.2f} {r['training_time_s']:>10.0f}")
    print("="*70)
    if results:
        print(f"\n  Best config: {results[0]['config_name']} "
              f"(reward={results[0]['mean_reward']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Train PG algorithms on Savanna env")
    parser.add_argument("--algo",  type=str, default="all",
                        choices=["all", "reinforce", "ppo", "a2c"],
                        help="Algorithm to train")
    parser.add_argument("--run",   type=str, default="all",
                        help="'all' or config name")
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS,
                        help="Total training timesteps per run")
    args = parser.parse_args()

    def run_algo(configs, train_fn, algo_label, save_key):
        results = []
        if args.run == "all":
            for i, cfg in enumerate(configs, start=1):
                res = train_fn(cfg, i, total_timesteps=args.steps)
                results.append(res)
        elif args.run == "best":
            best_cfg = configs[-1]  # last config is "best" by convention
            res = train_fn(best_cfg, len(configs), total_timesteps=args.steps)
            results.append(res)
        else:
            cfg = next((c for c in configs if c["name"] == args.run), None)
            if cfg is None:
                print(f"Config '{args.run}' not found in {algo_label} configs.")
                return []
            idx = configs.index(cfg) + 1
            res = train_fn(cfg, idx, total_timesteps=args.steps)
            results.append(res)

        print_summary(results, algo_label)
        combined = os.path.join(RESULTS_DIR, f"{save_key}_all_results.json")
        with open(combined, "w") as f:
            json.dump(results, f, indent=2)
        return results

    all_results = {}

    if args.algo in ("all", "reinforce"):
        all_results["reinforce"] = run_algo(
            REINFORCE_CONFIGS,
            train_reinforce,
            "REINFORCE",
            "reinforce"
        )

    if args.algo in ("all", "ppo"):
        all_results["ppo"] = run_algo(
            PPO_CONFIGS,
            lambda cfg, run_id, total_timesteps: train_sb3_algo(
                PPO, "PPO", cfg, run_id, total_timesteps),
            "PPO",
            "ppo"
        )

    if args.algo in ("all", "a2c"):
        all_results["a2c"] = run_algo(
            A2C_CONFIGS,
            lambda cfg, run_id, total_timesteps: train_sb3_algo(
                A2C, "A2C", cfg, run_id, total_timesteps),
            "A2C",
            "a2c"
        )

    # Grand summary across all algos
    if args.algo == "all":
        print("\n" + "="*70)
        print("  CROSS-ALGORITHM COMPARISON")
        print("="*70)
        print(f"  {'Algorithm':<15} {'Best Config':<22} {'Best Reward':>12}")
        print("  " + "-"*52)
        for algo_key, results in all_results.items():
            if results:
                best = max(results, key=lambda r: r["mean_reward"])
                print(f"  {algo_key.upper():<15} {best['config_name']:<22} "
                      f"{best['mean_reward']:>12.2f}")
        print("="*70)


if __name__ == "__main__":
    main()
