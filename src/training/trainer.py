import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from src.config.settings import ExperimentConfig, AlgorithmType
from src.core.interfaces import TrainingResult, EvaluationResult
from src.algorithms import AlgorithmRegistry
from src.evaluation.evaluator import ModelEvaluator
from src.visualization.plotter import (
    TrainingCurvePlotter,
    ComparisonPlotter,
    print_comparison_table,
)


class ExperimentRunner:
    """
    Facade for running complete RL experiments.
    
    Simplifies the process of:
    - Training multiple algorithms
    - Evaluating trained models
    - Generating visualizations
    - Saving results
    
    Benefits:
    - Single entry point for experiments
    - Coordinates complex workflows
    - Handles errors gracefully
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._training_results: List[TrainingResult] = []
        self._evaluation_results: List[EvaluationResult] = []
    
    def _print_header(self) -> None:
        """Print experiment header."""
        algo_names = [a.name for a in self.config.algorithms]
        
        print("\n" + "=" * 100)
        print(" " * 30 + "🤖 HUMANOID-V5 ALGORITHM COMPARISON 🤖")
        print(" " * 35 + "💪 RL BENCHMARK 💪")
        print("=" * 100)
        print(f"\nAlgorithms to test: {', '.join(algo_names)}")
        print(f"Total timesteps per algorithm: {self.config.training.total_timesteps:,}")
        print(f"Evaluation episodes: {self.config.n_eval_episodes_final}")
        print(f"\n{'='*100}\n")
    
    def train_all(self) -> List[TrainingResult]:
        """
        Train all configured algorithms.
        
        Returns:
            List of training results
        """
        self._training_results = []
        
        for algo_type in self.config.algorithms:
            algorithm = AlgorithmRegistry.create(
                algorithm_type=algo_type,
                config=self.config,
            )
            
            # Train
            result = algorithm.train()
            self._training_results.append(result)
            
            print(f"\n⏸️  Waiting 5 seconds before next algorithm...\n")
            time.sleep(5)
        
        print(f"\n{'='*100}")
        print(" " * 35 + "✅ ALL TRAINING COMPLETE!")
        print(f"{'='*100}\n")
        
        return self._training_results
    
    def evaluate_all(
        self,
        training_results: Optional[List[TrainingResult]] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate all trained models.
        
        Args:
            training_results: Results to evaluate (uses stored results if None)
            
        Returns:
            List of evaluation results
        """
        results = training_results or self._training_results
        
        if not results:
            raise ValueError("No training results to evaluate. Run train_all() first.")
        
        print(f"\n{'='*100}")
        print(" " * 30 + "🔍 STARTING EVALUATION PHASE")
        print(f"{'='*100}\n")
        
        self._evaluation_results = []
        
        for train_result in results:
            algo_type = AlgorithmType[train_result.algorithm_name]
            
            # Create evaluator
            evaluator = ModelEvaluator(
                env_config=self.config.environment,
                algorithm_type=algo_type,
            )
            
            # Evaluate
            eval_result = evaluator.evaluate(
                model_path=train_result.model_path,
                vec_normalize_path=train_result.vec_normalize_path,
                n_episodes=self.config.n_eval_episodes_final,
                seed=self.config.training.seed,
            )
            
            self._evaluation_results.append(eval_result)
        
        return self._evaluation_results
    
    def plot_results(
        self,
        training_results: Optional[List[TrainingResult]] = None,
        evaluation_results: Optional[List[EvaluationResult]] = None,
    ) -> None:
        """Generate all plots."""
        train_results = training_results or self._training_results
        eval_results = evaluation_results or self._evaluation_results
        
        # Training curves
        if train_results:
            curve_plotter = TrainingCurvePlotter(save_dir=self.config.figs_dir)
            curve_plotter.plot(train_results)
        
        # Comparison plots
        if eval_results:
            comparison_plotter = ComparisonPlotter(save_dir=self.config.figs_dir)
            comparison_plotter.plot(eval_results)
            
            # Print table
            print_comparison_table(eval_results)
    
    def save_results(
        self,
        evaluation_results: Optional[List[EvaluationResult]] = None,
    ) -> Path:
        """Save results to file."""
        results = evaluation_results or self._evaluation_results
        
        results_file = self.config.training.log_dir / "comparison_results.txt"
        
        with open(results_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("HUMANOID-V5 ALGORITHM COMPARISON RESULTS\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total timesteps: {self.config.training.total_timesteps:,}\n")
            f.write(f"Evaluation episodes: {self.config.n_eval_episodes_final}\n\n")
            
            results_sorted = sorted(results, key=lambda x: x.mean_reward, reverse=True)
            
            for i, r in enumerate(results_sorted, 1):
                f.write(f"{i}. {r.algorithm_name}: {r.mean_reward:.2f} ± {r.std_reward:.2f}\n")
                f.write(f"   Min: {r.min_reward:.2f}, Max: {r.max_reward:.2f}\n")
                f.write(f"   Mean length: {r.mean_length:.1f} steps\n\n")
        
        print(f"📝 Results saved to: {results_file}")
        return results_file
    
    def run(self) -> tuple[List[TrainingResult], List[EvaluationResult]]:
        """
        Run complete experiment: train, evaluate, plot, save.
        
        Returns:
            Tuple of (training_results, evaluation_results)
        """
        self._print_header()
        
        training_results = self.train_all()
        
        evaluation_results = self.evaluate_all()
        
        self.plot_results()
        
        self.save_results()
        
        self._print_footer()
        
        return training_results, evaluation_results
    
    def _print_footer(self) -> None:
        """Print experiment footer."""
        print("\n" + "=" * 100)
        print(" " * 35 + "🎉 BENCHMARK COMPLETE! 🎉")
        print("=" * 100)
        print(f"\n📊 View training progress:")
        print(f"   tensorboard --logdir={self.config.training.log_dir}")
        print(f"\n📈 Plots generated:")
        print(f"   {self.config.figs_dir}/training_curves_latest.png")
        print(f"   {self.config.figs_dir}/algorithm_comparison_latest.png")
        print(f"\n📝 Detailed results:")
        print(f"   {self.config.training.log_dir}/comparison_results.txt")
        print("\n" + "=" * 100 + "\n")