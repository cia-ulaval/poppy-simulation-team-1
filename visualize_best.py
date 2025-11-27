# Quick script to visualize best episodes of a trained model

import sys
from utils import visualize_best_episodes

if __name__ == "__main__":
    # Default model path
    model_path = "configs/models/ppo_humanoid_final.zip"

    # Allow command line argument
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    print("=" * 70)
    print("BEST EPISODES VISUALIZER")
    print("=" * 70 + "\n")

    print(f"Model: {model_path}\n")

    # Ask user for parameters
    try:
        n_total = int(input("Total episodes to evaluate (default 100): ") or "100")
        n_best = int(input("Number of best to show (default 10): ") or "10")
        save_video = input("Save videos? (y/n, default n): ").lower() == 'y'

        print("\nStarting...\n")

        visualize_best_episodes(
            model_path=model_path,
            n_total_episodes=n_total,
            n_show_best=n_best,
            save_video=save_video,
            video_folder="./videos/best"
        )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
