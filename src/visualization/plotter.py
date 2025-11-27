from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from src.core.interfaces import TrainingResult, EvaluationResult
from src.visualization.tensorboard_reader import TensorBoardReader


@dataclass
class PlotStyle:
    """Plot styling configuration."""
    figsize: tuple = (12, 6)
    dpi: int = 150
    title_fontsize: int = 14
    label_fontsize: int = 12
    legend_fontsize: int = 11
    grid_alpha: float = 0.3
    line_width: float = 2.0
    raw_alpha: float = 0.15


class AlgorithmColors:
    """Color scheme for different algorithms."""
    
    COLORS: Dict[str, str] = {
        'RANDOM': '#95a5a6',
        'PPO': '#3498db',
        'TD3': '#e74c3c',
        'SAC': '#2ecc71',
        'A2C': '#f39c12',
    }
    
    @classmethod
    def get(cls, algorithm: str) -> str:
        """Get color for algorithm, with fallback."""
        return cls.COLORS.get(algorithm.upper(), '#95a5a6')


class TrainingCurvePlotter:
    """
    Plots training reward curves over time.
    
    Reads TensorBoard logs and creates smoothed training curves
    for comparing algorithm learning progress.
    """
    
    def __init__(
        self,
        style: Optional[PlotStyle] = None,
        save_dir: Path = Path("./figs"),
    ):
        self.style = style or PlotStyle()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def _smooth_data(
        self,
        data: List[float],
        window: Optional[int] = None,
    ) -> np.ndarray:
        """Apply moving average smoothing."""
        if window is None:
            window = min(10, len(data) // 20 + 1)
        
        if len(data) <= window:
            return np.array(data)
        
        return np.convolve(data, np.ones(window) / window, mode='valid')
    
    def plot(
        self,
        training_results: List[TrainingResult],
        title: str = "Training Progress - Reward vs Timesteps",
        save_name: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Plot training curves for all algorithms.
        
        Args:
            training_results: List of training results with log directories
            title: Plot title
            save_name: Base name for saved file (auto-generated if None)
            
        Returns:
            Path to saved plot or None if no data
        """
        print(f"\n{'='*80}")
        print("📊 CREATING TRAINING CURVES")
        print(f"{'='*80}\n")
        
        fig, ax = plt.subplots(figsize=self.style.figsize)
        has_data = False
        
        for result in training_results:
            algo_name = result.algorithm_name
            reader = TensorBoardReader(result.log_dir)
            
            data = reader.read_rewards()
            
            if data is None:
                print(f"⚠️  No data for {algo_name}")
                continue
            
            timesteps, rewards = data
            
            # Smooth data
            rewards_smooth = self._smooth_data(rewards)
            timesteps_smooth = timesteps[:len(rewards_smooth)]
            
            # Get color and style
            color = AlgorithmColors.get(algo_name)
            linestyle = '--' if algo_name == 'RANDOM' else '-'
            linewidth = 1.5 if algo_name == 'RANDOM' else self.style.line_width
            
            # Plot smoothed line
            ax.plot(
                timesteps_smooth, rewards_smooth,
                label=algo_name,
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.8,
            )
            
            # Plot raw data with transparency
            ax.plot(
                timesteps, rewards,
                color=color,
                linewidth=0.5,
                alpha=self.style.raw_alpha,
            )
            
            has_data = True
            print(f"✅ Plotted {algo_name}: {len(timesteps)} data points")
        
        if not has_data:
            print("⚠️  No training data available to plot")
            plt.close()
            return None
        
        # Style the plot
        ax.set_xlabel('Timesteps', fontsize=self.style.label_fontsize, fontweight='bold')
        ax.set_ylabel('Episode Reward', fontsize=self.style.label_fontsize, fontweight='bold')
        ax.set_title(title, fontsize=self.style.title_fontsize, fontweight='bold')
        ax.legend(loc='best', fontsize=self.style.legend_fontsize)
        ax.grid(True, alpha=self.style.grid_alpha)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = save_name or "training_curves"
        
        save_path = self.save_dir / f"{base_name}_{timestamp}.png"
        plt.savefig(save_path, dpi=self.style.dpi, bbox_inches='tight')
        print(f"\n✅ Training curves saved: {save_path}")
        
        latest_path = self.save_dir / f"{base_name}_latest.png"
        plt.savefig(latest_path, dpi=self.style.dpi, bbox_inches='tight')
        print(f"✅ Latest curves saved: {latest_path}\n")
        
        plt.close()
        return save_path


class ComparisonPlotter:
    """
    Creates comparison plots for evaluation results.
    
    Generates bar charts and box plots comparing
    algorithm performance.
    """
    
    def __init__(
        self,
        style: Optional[PlotStyle] = None,
        save_dir: Path = Path("./figs"),
    ):
        self.style = style or PlotStyle()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot(
        self,
        results: List[EvaluationResult],
        save_name: Optional[str] = None,
    ) -> Path:
        """
        Create comparison plots.
        
        Args:
            results: List of evaluation results
            save_name: Base name for saved file
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        algos = [r.algorithm_name for r in results]
        means = [r.mean_reward for r in results]
        stds = [r.std_reward for r in results]
        rewards = [r.episode_rewards for r in results]
        
        colors = [AlgorithmColors.get(algo) for algo in algos]
        
        # Bar plot
        ax1 = axes[0]
        bars = ax1.bar(
            algos, means,
            yerr=stds,
            capsize=10,
            color=colors,
            alpha=0.7,
            edgecolor='black',
        )
        
        ax1.set_ylabel('Mean Reward', fontsize=self.style.label_fontsize, fontweight='bold')
        ax1.set_title('Algorithm Comparison - Mean Reward', fontsize=self.style.title_fontsize, fontweight='bold')
        ax1.grid(axis='y', alpha=self.style.grid_alpha)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2., height + std,
                f'{mean:.0f}',
                ha='center', va='bottom', fontweight='bold',
            )
        
        # Box plot
        ax2 = axes[1]
        bp = ax2.boxplot(rewards, labels=algos, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Reward Distribution', fontsize=self.style.label_fontsize, fontweight='bold')
        ax2.set_title('Algorithm Comparison - Reward Distribution', fontsize=self.style.title_fontsize, fontweight='bold')
        ax2.grid(axis='y', alpha=self.style.grid_alpha)
        
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_name = save_name or "algorithm_comparison"
        
        save_path = self.save_dir / f"{base_name}_{timestamp}.png"
        plt.savefig(save_path, dpi=self.style.dpi, bbox_inches='tight')
        print(f"\n✅ Comparison plot saved: {save_path}")
        
        latest_path = self.save_dir / f"{base_name}_latest.png"
        plt.savefig(latest_path, dpi=self.style.dpi, bbox_inches='tight')
        print(f"✅ Latest plot saved: {latest_path}\n")
        
        plt.close()
        return save_path


def print_comparison_table(results: List[EvaluationResult]) -> None:
    """Print formatted comparison table to console."""
    print(f"\n{'='*100}")
    print(f"{'🏆 FINAL COMPARISON - HUMANOID-V5':^100}")
    print(f"{'='*100}\n")
    
    # Sort by mean reward
    results_sorted = sorted(results, key=lambda x: x.mean_reward, reverse=True)
    
    # Print header
    print(f"{'Rank':<6} {'Algorithm':<12} {'Mean Reward':<20} {'Min':<12} {'Max':<12} {'Mean Length':<12}")
    print(f"{'-'*100}")
    
    # Print rows
    for i, r in enumerate(results_sorted, 1):
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        print(
            f"{rank_emoji:<6} {r.algorithm_name:<12} "
            f"{r.mean_reward:>8.2f} ± {r.std_reward:<6.2f}  "
            f"{r.min_reward:>8.2f}    {r.max_reward:>8.2f}    {r.mean_length:>8.1f}"
        )
    
    print(f"{'-'*100}\n")
    
    # Performance gap
    if len(results_sorted) > 1:
        best = results_sorted[0]
        worst = results_sorted[-1]
        
        if abs(worst.mean_reward) > 0:
            gain = ((best.mean_reward - worst.mean_reward) / abs(worst.mean_reward)) * 100
        else:
            gain = 0
        
        print(f"📊 Performance Gap:")
        print(f"   Best ({best.algorithm_name}): {best.mean_reward:.2f}")
        print(f"   Worst ({worst.algorithm_name}): {worst.mean_reward:.2f}")
        print(f"   Improvement: {gain:+.1f}%\n")
    
    print(f"{'='*100}\n")