"""
Example Usage:
    from src.config import ExperimentConfig, AlgorithmType
    from src.training import ExperimentRunner
    
    config = ExperimentConfig(
        algorithms=[AlgorithmType.PPO, AlgorithmType.SAC],
        training=TrainingConfig(total_timesteps=1_000_000),
    )
    
    runner = ExperimentRunner(config)
    training_results, eval_results = runner.run()
"""

from src.config import ExperimentConfig, AlgorithmType
from src.training import ExperimentRunner
from src.algorithms import AlgorithmRegistry

__all__ = [
    "ExperimentConfig",
    "AlgorithmType",
    "ExperimentRunner",
    "AlgorithmRegistry",
]