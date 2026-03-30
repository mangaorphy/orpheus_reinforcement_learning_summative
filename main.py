"""
main.py — Savanna Acoustic Threat Detection RL System
======================================================
Entry point for running the best-performing trained agent
with full Pygame visualization and terminal verbose output.

Usage:
    python main.py                         # Run best model (auto-detect)
    python main.py --algo ppo              # Run best PPO model
    python main.py --algo dqn              # Run best DQN model
    python main.py --algo a2c              # Run best A2C model
    python main.py --algo reinforce        # Run best REINFORCE model
    python main.py --demo                  # Random agent demo (no model)
    python main.py --algo ppo --episodes 5 # Run 5 evaluation episodes
    python main.py --compare               # Compare all saved models
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import glob

# ── Make sure project root is on path ─────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from environment.custom_env import SavannaAcousticEnv

ACTION_NAMES = {
    0: "NORTH ↑",
    1: "SOUTH ↓",
    2: "EAST  →",
    3: "WEST  ←",
    4: "SCAN  🔊",
    5: "ALERT 🚨",
    6: "HOVER ⏸",
}


# --------------------------------------------------------------------------- #
# Model loader helpers
# --------------------------------------------------------------------------- #

def load_sb3_model(algo: str, config_name: str):
    """Load a Stable-Baselines3 model (DQN/PPO/A2C)."""
    from stable_baselines3 import DQN, PPO, A2C
    algo_map = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
    cls = algo_map[algo]

    # DQN lives at:  models/dqn/<config_name>/
    # PPO/A2C live at: models/pg/<algo>/<config_name>/
    if algo == "dqn":
        search_roots = [
            os.path.join(ROOT, "models", "dqn", config_name),
        ]
    else:
        search_roots = [
            os.path.join(ROOT, "models", "pg", algo, config_name),
        ]

    # Try best_model then final_model in the correct folder
    for root in search_roots:
        for fname in ["best_model", "final_model"]:
            path = os.path.join(root, fname)
            if os.path.exists(path + ".zip"):
                print(f"  Loading {algo.upper()} model: {path}.zip")
                return cls.load(path)

    # Fallback: search entire models/dqn or models/pg tree
    # but ONLY load files from the requested config folder
    if algo == "dqn":
        base = os.path.join(ROOT, "models", "dqn")
    else:
        base = os.path.join(ROOT, "models", "pg", algo)

    # Look for config_name folder specifically first
    zips = glob.glob(os.path.join(base, config_name, "*.zip"), recursive=False)
    if not zips:
        # Last resort — any zip but warn user
        zips = glob.glob(os.path.join(base, "**", "*.zip"), recursive=True)
        if zips:
            print(f"  WARNING: '{config_name}' not found, loading: {zips[0]}")

    if zips:
        print(f"  Loading model: {zips[0]}")
        return cls.load(zips[0])

    return None


def load_reinforce_model(config_name: str):
    """Load a REINFORCE policy (PyTorch)."""
    import torch
    from training.pg_training import REINFORCEPolicy

    base = os.path.join(ROOT, "models", "pg", "reinforce", config_name)
    pts  = glob.glob(os.path.join(base, "*.pt"))
    if not pts:
        # search wider
        pts = glob.glob(os.path.join(ROOT, "models", "pg", "reinforce", "**", "*.pt"),
                        recursive=True)
    if not pts:
        return None

    pt_path = pts[0]
    print(f"  Loading REINFORCE model: {pt_path}")
    ckpt    = torch.load(pt_path, map_location="cpu")
    config  = ckpt.get("config", {})
    net_arch = config.get("net_arch", (128, 128))

    dummy_env = SavannaAcousticEnv()
    obs_dim   = dummy_env.observation_space.shape[0]
    act_dim   = dummy_env.action_space.n
    dummy_env.close()

    policy = REINFORCEPolicy(obs_dim, act_dim, net_arch)
    policy.load_state_dict(ckpt["policy_state"])
    policy.eval()
    return policy


def find_best_config(algo: str) -> str:
    """Read results JSON to find the best-performing config name."""
    results_dir = os.path.join(ROOT, "results")
    pattern     = os.path.join(results_dir, f"{algo}_all_results.json")
    if os.path.exists(pattern):
        with open(pattern) as f:
            results = json.load(f)
        if results:
            best = max(results, key=lambda r: r.get("mean_reward", -9999))
            return best.get("config_name", f"{algo}_best")

    # Also try individual result files
    files = glob.glob(os.path.join(results_dir, f"{algo}_*.json"))
    if files:
        best_r, best_name = -9999, None
        for fp in files:
            with open(fp) as f:
                r = json.load(f)
            if isinstance(r, dict) and r.get("mean_reward", -9999) > best_r:
                best_r    = r["mean_reward"]
                best_name = r.get("config_name")
        if best_name:
            return best_name

    # Fall back to known "best" configs
    fallbacks = {
        "dqn":       "best_tuned",
        "ppo":       "ppo_best",
        "a2c":       "a2c_best",
        "reinforce": "rf_best",
    }
    return fallbacks.get(algo, f"{algo}_best")


# --------------------------------------------------------------------------- #
# Episode runner
# --------------------------------------------------------------------------- #

def predict_action(model, algo: str, obs: np.ndarray):
    """Get deterministic action from any model type."""
    if algo == "reinforce":
        import torch
        obs_t  = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, _ = model.get_action(obs_t)
        return action.item()
    else:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)


def run_episode(model, algo: str, env: SavannaAcousticEnv,
                episode_num: int, verbose: bool = True) -> dict:
    """Run a single evaluation episode with full verbose output."""
    obs, info = env.reset()
    ep_reward = 0.0
    step      = 0
    done      = False

    if verbose:
        print(f"\n{'─'*65}")
        print(f"  EPISODE {episode_num} START")
        print(f"  Threats spawned: {len(env.threats)}")
        for i, t in enumerate(env.threats):
            tname = ["Poaching", "Predator", "Habitat"][t["type"] - 1]
            sname = ["Elephant", "Lion", "Hyena", "Zebra"][t["species"]]
            print(f"    Threat {i+1}: {tname:<10} at {t['pos']} — {sname} "
                  f"(severity={t['severity']:.2f})")
        print(f"{'─'*65}")

    while not done:
        action = predict_action(model, algo, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        done       = terminated or truncated

        if verbose and (step % 25 == 0 or abs(reward) > 5):
            batt_bar = "█" * int(env.battery / 5) + "░" * (20 - int(env.battery / 5))
            print(
                f"  Step {step:>3} | {ACTION_NAMES[action]:<12} | "
                f"Reward: {reward:>+7.1f} | Total: {ep_reward:>8.1f} | "
                f"Batt: [{batt_bar}] {env.battery:.0f}%"
            )

        if env.render_mode:
            if env.renderer is None:
                from environment.rendering import SavannaRenderer
                env.renderer = SavannaRenderer(env.grid_size)
            env.renderer.render(env, env.render_mode,
                                last_reward=reward, last_action=action)

        step += 1

    resolved  = sum(1 for t in env.threats if t["alerted"])
    confirmed = sum(1 for t in env.threats if t["confirmed"])

    if verbose:
        print(f"\n  {'─'*60}")
        print(f"  EPISODE {episode_num} COMPLETE")
        print(f"  Steps          : {step}")
        print(f"  Total reward   : {ep_reward:.2f}")
        print(f"  Threats alerted: {resolved} / {len(env.threats)}")
        print(f"  Confirmed      : {confirmed} / {len(env.threats)}")
        print(f"  Cells visited  : {len(env.visited)} / 400 "
              f"({100*len(env.visited)/400:.1f}%)")
        print(f"  Battery left   : {env.battery:.1f}%")

        if all(t["alerted"] for t in env.threats):
            print("  🎉 MISSION COMPLETE — All threats neutralised!")
        elif env.battery <= 0:
            print("  ⚠️  Battery depleted — drone returned to base")
        else:
            print("  ⏱  Max steps reached")
        print(f"  {'─'*60}")

    return {
        "episode":         episode_num,
        "steps":           step,
        "total_reward":    ep_reward,
        "threats_alerted": resolved,
        "threats_total":   len(env.threats),
        "cells_visited":   len(env.visited),
        "battery_left":    env.battery,
    }


# --------------------------------------------------------------------------- #
# Compare all saved models
# --------------------------------------------------------------------------- #

def compare_models():
    """Load all saved result JSON files and print a comparison table."""
    results_dir = os.path.join(ROOT, "results")
    all_files   = glob.glob(os.path.join(results_dir, "*_all_results.json"))

    if not all_files:
        print("  No result files found. Train models first.")
        return

    print("\n" + "="*75)
    print("  CROSS-ALGORITHM PERFORMANCE COMPARISON")
    print("="*75)
    print(f"  {'Algorithm':<12} {'Best Config':<22} {'Mean Reward':>12} {'Std':>8} {'Time(s)':>10}")
    print("  " + "─"*66)

    all_bests = []
    for fp in sorted(all_files):
        with open(fp) as f:
            results = json.load(f)
        if not results:
            continue
        best = max(results, key=lambda r: r.get("mean_reward", -9999))
        all_bests.append(best)
        print(f"  {best['algorithm']:<12} {best['config_name']:<22} "
              f"{best['mean_reward']:>12.2f} {best['std_reward']:>8.2f} "
              f"{best['training_time_s']:>10.0f}")

    if all_bests:
        overall_best = max(all_bests, key=lambda r: r["mean_reward"])
        print("="*75)
        print(f"\n  🏆 Overall best: {overall_best['algorithm']} "
              f"— {overall_best['config_name']} "
              f"(reward={overall_best['mean_reward']:.2f})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Run best-performing RL agent on Savanna environment"
    )
    parser.add_argument("--algo",     type=str, default="ppo",
                        choices=["dqn", "ppo", "a2c", "reinforce"],
                        help="Algorithm to run")
    parser.add_argument("--config",   type=str, default=None,
                        help="Specific config name (default: auto-detect best)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of evaluation episodes")
    parser.add_argument("--no-render",action="store_true",
                        help="Disable Pygame rendering")
    parser.add_argument("--demo",     action="store_true",
                        help="Run random agent demo (no model needed)")
    parser.add_argument("--compare",  action="store_true",
                        help="Compare all saved models and exit")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    # Banner
    print("\n" + "="*65)
    print("  🦁  SAVANNA ACOUSTIC THREAT DETECTION")
    print("  African Wildlife Conservation RL System")
    print("="*65)

    # Compare mode
    if args.compare:
        compare_models()
        return

    # Random demo mode
    if args.demo:
        from environment.rendering import run_random_demo
        run_random_demo(num_steps=300, seed=args.seed)
        return

    # Auto-detect best config if not specified
    config_name = args.config or find_best_config(args.algo)
    render_mode = None if args.no_render else "human"

    print(f"\n  Algorithm   : {args.algo.upper()}")
    print(f"  Config      : {config_name}")
    print(f"  Episodes    : {args.episodes}")
    print(f"  Rendering   : {'off' if args.no_render else 'Pygame GUI'}")
    print(f"  Seed        : {args.seed}")

    # Load model
    print("\n  Loading model...")
    if args.algo == "reinforce":
        model = load_reinforce_model(config_name)
    else:
        model = load_sb3_model(args.algo, config_name)

    if model is None:
        print(f"\n  ⚠️  No trained model found for {args.algo}/{config_name}.")
        print(f"  Train first: python training/{'dqn' if args.algo=='dqn' else 'pg'}_training.py "
              f"--algo {args.algo} --run best")
        print(f"\n  Falling back to random agent demo...")
        from environment.rendering import run_random_demo
        run_random_demo(num_steps=300, seed=args.seed)
        return

    print(f"  ✓ Model loaded successfully")

    # Environment
    env = SavannaAcousticEnv(render_mode=render_mode, seed=args.seed)

    print(f"\n  Environment specs:")
    print(f"    Grid size    : {env.grid_size}×{env.grid_size}")
    print(f"    Obs shape    : {env.observation_space.shape}")
    print(f"    Action space : {env.action_space.n} discrete")
    print(f"    Max steps    : 500")
    print(f"    Threats/ep   : 5")

    # Pre-init renderer so window appears immediately
    if render_mode == "human":
        from environment.rendering import SavannaRenderer
        env.renderer = SavannaRenderer(env.grid_size)
        env.renderer._init()

    # Run evaluation episodes
    all_ep_results = []
    for ep in range(1, args.episodes + 1):
        # Clear trail between episodes
        if render_mode == "human" and env.renderer:
            env.renderer._trail.clear()
            env.renderer._cumulative_reward = 0.0
        result = run_episode(model, args.algo, env, ep, verbose=True)
        all_ep_results.append(result)
        if render_mode == "human":
            time.sleep(1.0)   # pause between episodes so you can see the result

    env.close()

    # Aggregate stats
    rewards  = [r["total_reward"]    for r in all_ep_results]
    resolved = [r["threats_alerted"] for r in all_ep_results]
    visited  = [r["cells_visited"]   for r in all_ep_results]

    print("\n" + "="*65)
    print("  EVALUATION SUMMARY")
    print("="*65)
    print(f"  Episodes evaluated : {args.episodes}")
    print(f"  Mean reward        : {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Best episode       : {max(rewards):.2f}")
    print(f"  Worst episode      : {min(rewards):.2f}")
    print(f"  Avg threats alerted: {np.mean(resolved):.1f} / 5")
    print(f"  Avg cells visited  : {np.mean(visited):.0f} / 400")
    print("="*65)
    print("\n  Run with --compare to compare all trained algorithms.")
    print("  Run with --demo for a random-agent visualization.")


if __name__ == "__main__":
    main()