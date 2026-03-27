"""
train_all.py — Memory-Managed Full Training Pipeline
=====================================================
Trains all 4 algorithms sequentially with aggressive memory
cleanup between runs. Use this for local Mac training.

Usage:
    python train_all.py                   # all algorithms, 100k steps
    python train_all.py --algo dqn        # DQN only
    python train_all.py --steps 50000     # faster (less learning)
    python train_all.py --best-only       # only best config per algo
"""

import os, sys, gc, time, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def clear_memory(label=""):
    """Aggressively free RAM between training runs."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"  [memory] {label} | RAM: {ram.percent:.0f}% used "
              f"({(ram.total-ram.available)/1e9:.1f}/{ram.total/1e9:.1f} GB)")
        if ram.percent > 85:
            print("  WARNING: RAM > 85% — consider closing other apps")
    except ImportError:
        print(f"  [memory cleared] {label}")

def run_training(script_args: list, label: str):
    """Run a training script as a subprocess for full memory isolation."""
    print(f"\n{'='*60}")
    print(f"  STARTING: {label}")
    print(f"{'='*60}")
    t0  = time.time()
    ret = subprocess.run([sys.executable] + script_args)
    elapsed = time.time() - t0
    print(f"\n  FINISHED: {label} in {elapsed/60:.1f} min")
    clear_memory(f"after {label}")
    return ret.returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",      default="all",
                        choices=["all","dqn","ppo","a2c","reinforce"])
    parser.add_argument("--steps",     type=int, default=100_000)
    parser.add_argument("--best-only", action="store_true",
                        help="Run only best config per algorithm (faster)")
    args = parser.parse_args()

    run_arg = "--run best" if args.best_only else ""

    print("\n" + "="*60)
    print("  SAVANNA RL — FULL TRAINING PIPELINE")
    print(f"  Timesteps per run : {args.steps:,}")
    print(f"  Mode              : {'best config only' if args.best_only else 'all 10 configs'}")
    print("="*60)
    clear_memory("startup")

    success = {}

    if args.algo in ("all", "dqn"):
        cmd = ["training/dqn_training.py", f"--steps={args.steps}"]
        if args.best_only:
            cmd += ["--run", "best"]
        success["DQN"] = run_training(cmd, f"DQN ({args.steps:,} steps)")

    if args.algo in ("all", "ppo"):
        cmd = ["training/pg_training.py", "--algo", "ppo", f"--steps={args.steps}"]
        if args.best_only:
            cmd += ["--run", "best"]
        success["PPO"] = run_training(cmd, f"PPO ({args.steps:,} steps)")

    if args.algo in ("all", "a2c"):
        cmd = ["training/pg_training.py", "--algo", "a2c", f"--steps={args.steps}"]
        if args.best_only:
            cmd += ["--run", "best"]
        success["A2C"] = run_training(cmd, f"A2C ({args.steps:,} steps)")

    if args.algo in ("all", "reinforce"):
        cmd = ["training/pg_training.py", "--algo", "reinforce", f"--steps={args.steps}"]
        if args.best_only:
            cmd += ["--run", "best"]
        success["REINFORCE"] = run_training(cmd, f"REINFORCE ({args.steps:,} steps)")

    print("\n" + "="*60)
    print("  TRAINING COMPLETE — SUMMARY")
    print("="*60)
    for algo, ok in success.items():
        status = "OK" if ok else "FAILED"
        print(f"  {algo:<12} : {status}")
    print("\n  Run comparison:")
    print("    python main.py --compare")
    print("  Run best agent:")
    print("    python main.py --algo ppo --episodes 3")
    print("="*60)

if __name__ == "__main__":
    main()
