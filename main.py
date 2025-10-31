# Poppy RL - Main entry point
# Usage: python main.py train | evaluate MODEL.zip | visualize MODEL.zip | visualize-best MODEL.zip | compare MODEL.zip

import argparse
import sys
import time

from config import Config
from utils import train, evaluate, compare_with_baseline, visualize, visualize_best_episodes


def print_banner():
    banner = """
    ================================================================
                POPPY RL - WALKING ROBOT PROJECT
           Reinforcement Learning for Humanoid Locomotion
    ================================================================
    """
    print(banner)


def cmd_train(args):
    print("\n" + "=" * 60)
    print("MODE: TRAINING")
    print("=" * 60 + "\n")

    config_path = args.config if args.config else "configs/ppo_humanoid.yaml"
    print(f"Config: {config_path}\n")

    cfg = Config(config_path)
    cfg.print_summary()

    train_kwargs = {
        'total_timesteps': args.steps,
        'n_envs': args.envs,
        'model_name': args.name if args.name else "ppo_humanoid_final",
        'resume_from': args.resume,
        'use_eval': True,
    }

    print("Starting training in 3 seconds...")
    time.sleep(3)

    model = train(cfg=cfg, **train_kwargs)

    print("\n[OK] Training complete!")
    print(f"Model saved: models/{train_kwargs['model_name']}.zip")
    print(f"\nView logs: tensorboard --logdir=tensorboard_logs")

    return model


def cmd_evaluate(args):
    if not args.model:
        print("[ERROR] --model required for evaluate")
        print("   Example: python main.py evaluate models/ppo_humanoid_final.zip")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("MODE: EVALUATION")
    print("=" * 60 + "\n")

    n_episodes = args.episodes if args.episodes else 20
    print(f"Model: {args.model}")
    print(f"Episodes: {n_episodes}\n")

    stats = evaluate(
        model_path=args.model,
        n_episodes=n_episodes,
        render=not args.no_render,
        deterministic=True
    )

    return stats


def cmd_visualize(args):
    if not args.model:
        print("[ERROR] --model required for visualize")
        print("   Example: python main.py visualize models/ppo_humanoid_final.zip")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("MODE: VISUALIZATION")
    print("=" * 60 + "\n")

    n_episodes = args.episodes if args.episodes else 3
    print(f"Model: {args.model}")
    print(f"Episodes: {n_episodes}")
    print(f"Video: {'YES' if args.video else 'NO'}\n")

    visualize(
        model_path=args.model,
        n_episodes=n_episodes,
        deterministic=True,
        save_video=args.video,
        video_folder="./videos"
    )


def cmd_visualize_best(args):
    if not args.model:
        print("[ERROR] --model required for visualize-best")
        print("   Example: python main.py visualize-best models/ppo_humanoid_final.zip")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("MODE: BEST EPISODES VISUALIZATION")
    print("=" * 60 + "\n")

    n_total = args.total if args.total else 100
    n_best = args.best if args.best else 10

    print(f"Model: {args.model}")
    print(f"Total episodes to evaluate: {n_total}")
    print(f"Best episodes to show: {n_best}")
    print(f"Video: {'YES' if args.video else 'NO'}\n")

    visualize_best_episodes(
        model_path=args.model,
        n_total_episodes=n_total,
        n_show_best=n_best,
        deterministic=True,
        save_video=args.video,
        video_folder="./videos/best"
    )


def cmd_compare(args):
    if not args.model:
        print("[ERROR] --model required for compare")
        print("   Example: python main.py compare models/ppo_humanoid_final.zip")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("MODE: COMPARISON WITH BASELINE")
    print("=" * 60 + "\n")

    n_episodes = args.episodes if args.episodes else 20
    print(f"Model: {args.model}")
    print(f"Episodes: {n_episodes}\n")

    compare_with_baseline(
        model_path=args.model,
        n_episodes=n_episodes
    )


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Poppy RL - Reinforcement Learning for Humanoid Locomotion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train
  python main.py train --config configs/ppo_humanoid_test.yaml
  python main.py train --steps 100000 --envs 4
  python main.py evaluate models/ppo_humanoid_final.zip
  python main.py visualize models/ppo_humanoid_final.zip --video
  python main.py visualize-best models/ppo_humanoid_final.zip --total 100 --best 10
  python main.py compare models/ppo_humanoid_final.zip
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Train
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--config', type=str, help='Config YAML file')
    train_parser.add_argument('--steps', type=int, help='Number of timesteps')
    train_parser.add_argument('--envs', type=int, help='Number of parallel environments')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    train_parser.add_argument('--name', type=str, help='Model name')

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a model')
    eval_parser.add_argument('model', type=str, nargs='?', help='Path to model .zip')
    eval_parser.add_argument('--model', type=str, dest='model_flag', help='Path to model .zip')
    eval_parser.add_argument('--episodes', type=int, help='Number of episodes')
    eval_parser.add_argument('--no-render', action='store_true', help='Disable visualization')

    # Visualize
    viz_parser = subparsers.add_parser('visualize', help='Visualize a model')
    viz_parser.add_argument('model', type=str, nargs='?', help='Path to model .zip')
    viz_parser.add_argument('--model', type=str, dest='model_flag', help='Path to model .zip')
    viz_parser.add_argument('--episodes', type=int, help='Number of episodes')
    viz_parser.add_argument('--video', action='store_true', help='Record video')
    viz_parser.add_argument('--no-render', action='store_true', help='Disable visualization')

    # Visualize Best
    viz_best_parser = subparsers.add_parser('visualize-best', help='Visualize best episodes')
    viz_best_parser.add_argument('model', type=str, nargs='?', help='Path to model .zip')
    viz_best_parser.add_argument('--model', type=str, dest='model_flag', help='Path to model .zip')
    viz_best_parser.add_argument('--total', type=int, help='Total episodes to evaluate (default: 100)')
    viz_best_parser.add_argument('--best', type=int, help='Number of best to show (default: 10)')
    viz_best_parser.add_argument('--video', action='store_true', help='Record video')

    # Compare
    comp_parser = subparsers.add_parser('compare', help='Compare with baseline')
    comp_parser.add_argument('model', type=str, nargs='?', help='Path to model .zip')
    comp_parser.add_argument('--model', type=str, dest='model_flag', help='Path to model .zip')
    comp_parser.add_argument('--episodes', type=int, help='Number of episodes')

    args = parser.parse_args()

    # Handle positional or flag model argument
    if hasattr(args, 'model') and hasattr(args, 'model_flag'):
        if args.model_flag:
            args.model = args.model_flag

    if not args.command:
        parser.print_help()
        print("\n[ERROR] No command specified")
        print("   Use: train, evaluate, visualize, visualize-best, or compare\n")
        sys.exit(1)

    try:
        if args.command == 'train':
            cmd_train(args)
        elif args.command == 'evaluate':
            cmd_evaluate(args)
        elif args.command == 'visualize':
            cmd_visualize(args)
        elif args.command == 'visualize-best':
            cmd_visualize_best(args)
        elif args.command == 'compare':
            cmd_compare(args)

        print("\n[OK] Operation complete!\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR]: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
