from typing import Dict, Type, Optional, List

from src.config.settings import AlgorithmType, ExperimentConfig
from src.core.base_algorithm import BaseAlgorithm
from src.core.interfaces import TrainingObserver

from src.algorithms.ppo_algorithm import PPOAlgorithm
from src.algorithms.td3_algorithm import TD3Algorithm
from src.algorithms.sac_algorithm import SACAlgorithm
from src.algorithms.a2c_algorithm import A2CAlgorithm
from src.algorithms.random_algorithm import RandomAlgorithm


class AlgorithmRegistry:
    """
    Registry Pattern for algorithm creation.
    
    Benefits:
    - Decouples algorithm creation from usage
    - Easy to add new algorithms
    - Provides a single point of access
    """
    
    _algorithms: Dict[AlgorithmType, Type] = {
        AlgorithmType.PPO: PPOAlgorithm,
        AlgorithmType.TD3: TD3Algorithm,
        AlgorithmType.SAC: SACAlgorithm,
        AlgorithmType.A2C: A2CAlgorithm,
        AlgorithmType.RANDOM: RandomAlgorithm,
    }
    
    @classmethod
    def create(
        cls,
        algorithm_type: AlgorithmType,
        config: ExperimentConfig,
        observers: Optional[List[TrainingObserver]] = None,
    ) -> BaseAlgorithm:
        """
        Factory method to create an algorithm instance.
        
        Args:
            algorithm_type: Type of algorithm to create
            config: Experiment configuration
            observers: Optional list of training observers
            
        Returns:
            Algorithm instance
            
        Raises:
            ValueError: If algorithm type is not registered
        """
        if algorithm_type not in cls._algorithms:
            raise ValueError(
                f"Unknown algorithm: {algorithm_type}. "
                f"Available: {list(cls._algorithms.keys())}"
            )
        
        algorithm_class = cls._algorithms[algorithm_type]
        return algorithm_class(config=config, observers=observers)
    
    @classmethod
    def register(cls, algorithm_type: AlgorithmType, algorithm_class: Type) -> None:
        """
        Register a new algorithm type.
        
        Allows extending with custom algorithms without modifying this code.
        """
        cls._algorithms[algorithm_type] = algorithm_class
    
    @classmethod
    def available_algorithms(cls) -> List[AlgorithmType]:
        """Return list of available algorithm types."""
        return list(cls._algorithms.keys())


__all__ = [
    "AlgorithmRegistry",
    "PPOAlgorithm",
    "TD3Algorithm",
    "SACAlgorithm",
    "A2CAlgorithm",
    "RandomAlgorithm",
]