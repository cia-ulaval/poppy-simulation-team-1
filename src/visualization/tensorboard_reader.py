from pathlib import Path
from typing import Optional, Tuple, List
from glob import glob

try:
    from tensorboard.backend.event_processing import event_accumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TensorBoardReader:
    """
    Reads TensorBoard event files to extract training metrics.
    
    Useful for:
    - Post-training analysis
    - Plotting training curves
    - Comparing algorithm performance over time
    """
    
    REWARD_TAGS = [
        'rollout/ep_rew_mean',
        'eval/mean_reward',
        'train/reward',
        'rollout/reward',
    ]
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        
        if not TENSORBOARD_AVAILABLE:
            raise ImportError(
                "TensorBoard is required for reading logs. "
                "Install with: pip install tensorboard"
            )
    
    def _find_event_files(self) -> List[Path]:
        """Find all TensorBoard event files in log directory."""
        pattern = str(self.log_dir / "**" / "events.out.tfevents.*")
        files = glob(pattern, recursive=True)
        return [Path(f) for f in files]
    
    def read_rewards(
        self,
        add_initial_point: bool = True,
    ) -> Optional[Tuple[List[int], List[float]]]:
        """
        Read reward values from TensorBoard logs.
        
        Args:
            add_initial_point: If True, add point at timestep 0 for plotting
            
        Returns:
            Tuple of (timesteps, rewards) or None if not available
        """
        event_files = self._find_event_files()
        
        if not event_files:
            print(f"⚠️  No TensorBoard event files found in {self.log_dir}")
            return None
        
        event_file = max(event_files, key=lambda f: f.stat().st_mtime)
        
        try:
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()
            
            scalar_tags = ea.Tags()['scalars']
            
            timesteps = []
            rewards = []
            
            for tag in self.REWARD_TAGS:
                if tag in scalar_tags:
                    events = ea.Scalars(tag)
                    timesteps = [e.step for e in events]
                    rewards = [e.value for e in events]
                    break
            
            if not timesteps:
                print(f"⚠️  No reward data found in logs")
                return None
            
            if add_initial_point and timesteps[0] > 0:
                timesteps = [0] + timesteps
                rewards = [rewards[0]] + rewards
            
            return timesteps, rewards
            
        except Exception as e:
            print(f"⚠️  Error reading TensorBoard logs: {e}")
            return None
    
    def read_scalar(
        self,
        tag: str,
    ) -> Optional[Tuple[List[int], List[float]]]:
        """Read any scalar value from logs."""
        event_files = self._find_event_files()
        
        if not event_files:
            return None
        
        event_file = max(event_files, key=lambda f: f.stat().st_mtime)
        
        try:
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()
            
            if tag not in ea.Tags()['scalars']:
                return None
            
            events = ea.Scalars(tag)
            timesteps = [e.step for e in events]
            values = [e.value for e in events]
            
            return timesteps, values
            
        except Exception as e:
            print(f"⚠️  Error reading scalar {tag}: {e}")
            return None
    
    @property
    def available_scalars(self) -> List[str]:
        """List all available scalar tags."""
        event_files = self._find_event_files()
        
        if not event_files:
            return []
        
        event_file = max(event_files, key=lambda f: f.stat().st_mtime)
        
        try:
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()
            return ea.Tags()['scalars']
        except Exception:
            return []