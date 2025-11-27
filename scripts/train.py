#!/usr/bin/env python3
"""
Main training script for Humanoid-v5 algorithm comparison.

Usage:
    python scripts/train.py
    python scripts/train.py --algorithms PPO SAC --timesteps 1000000
    python scripts/train.py --config config.yaml
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    ExperimentConfig,
    AlgorithmType,
    TrainingConfig,
    EnvironmentConfig,
)
from src.training import ExperimentRunner


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL algorithms on Humanoid-v5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train PPO and A2C with default settings
  python scripts/train.py --algorithms PPO A2C
  
  # Quick test with 100k timesteps
  python scripts/train.py --algorithms PPO --timesteps 100000
  
  # Full benchmark
  python scripts/train.py --algorithms RANDOM PPO TD3 SAC A2C --timesteps 10000000
        """,
    )
    
    parser.add_argument(
        "--algorithms",
        nargs="+",
        type=str,
        default=["RANDOM", "PPO", "A2C"],
        choices=["RANDOM", "PPO", "TD3", "SAC", "A2C"],
        help="Algorithms to train (default: RANDOM PPO A2C)",
    )
    
    parser.add_argument(
        "--timesteps",
        type=int,
        default=int(1e7),
        help="Total timesteps per algorithm (default: 10M)",
    )
    
    parser.add_argument(
        "--n-envs",
        type=int,
        default=32,
        help="Number of parallel environments (default: 32)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("./logs"),
        help="Directory for logs (default: ./logs)",
    )
    
    parser.add_argument(
        "--figs-dir",
        type=Path,
        default=Path("./figs"),
        help="Directory for figures (default: ./figs)",
    )
    
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20)",
    )
    
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation after training",
    )
    
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Convert algorithm names to enum
    algorithms = [AlgorithmType[name] for name in args.algorithms]
    
    # Create configuration
    config = ExperimentConfig(
        algorithms=algorithms,
        environment=EnvironmentConfig(
            n_envs=args.n_envs,
        ),
        training=TrainingConfig(
            total_timesteps=args.timesteps,
            seed=args.seed,
            log_dir=args.log_dir,
        ),
        figs_dir=args.figs_dir,
        n_eval_episodes_final=args.eval_episodes,
    )
    
    # Create and run experiment
    runner = ExperimentRunner(config)
    
    try:
        if args.skip_eval and args.skip_plots:
            # Only train
            runner.train_all()
        elif args.skip_eval:
            # Train and plot training curves only
            training_results = runner.train_all()
            runner.plot_results(training_results=training_results, evaluation_results=[])
        elif args.skip_plots:
            # Train and evaluate, no plots
            runner.train_all()
            runner.evaluate_all()
            runner.save_results()
        else:
            # Full experiment
            runner.run()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        return 1
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    sys.exit(main())