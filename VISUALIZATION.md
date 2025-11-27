# Visualization Guide

## Overview

The visualization system allows you to watch trained models perform in real-time and optionally record videos.

## Standard Visualization

Visualize a trained model for a few episodes:

```bash
# Visualize 3 episodes (default)
python main.py visualize configs/models/ppo_humanoid_final.zip

# Visualize 10 episodes
python main.py visualize configs/models/ppo_humanoid_final.zip --episodes 10

# Record videos
python main.py visualize configs/models/ppo_humanoid_final.zip --video
```

## Best Episodes Visualization

**NEW**: Run many episodes and only visualize the best ones!

### Method 1: Use the script

```bash
python visualize_best.py

# Or specify a model
python visualize_best.py configs/models/my_model.zip
```

This will:
1. Evaluate 100 episodes (configurable)
2. Rank them by reward
3. Show you the top 10 (configurable)
4. Optionally record videos of the best ones

### Method 2: Use Python directly

```python
from utils import visualize_best_episodes

visualize_best_episodes(
    model_path="configs/models/ppo_humanoid_final.zip",
    n_total_episodes=100,  # Evaluate 100 episodes
    n_show_best=10,        # Show top 10
    save_video=False       # Set True to record
)
```

## Features

### Standard Visualization
- ✅ Real-time rendering in MuJoCo window
- ✅ Shows progress every 100 steps
- ✅ Summary statistics at the end
- ✅ Optional video recording (MP4)

### Best Episodes Visualization
- ✅ **Phase 1**: Evaluate many episodes without rendering (fast)
- ✅ **Phase 2**: Visualize only the best ones
- ✅ Shows ranking of best episodes
- ✅ Can pause between episodes
- ✅ Optional video recording with descriptive names

## Tips

1. **For quick testing**: Use standard visualization with 3 episodes
   ```bash
   python main.py visualize models/my_model.zip
   ```

2. **For best performance showcase**: Use best episodes mode
   ```bash
   python visualize_best.py
   # Then enter: 100 total, 5 best
   ```

3. **For creating demos**: Record videos of best episodes
   ```bash
   python visualize_best.py
   # Then: 100 total, 10 best, save_video=yes
   ```

4. **Slow motion**: The visualization runs at ~100 FPS. MuJoCo actually runs at 500Hz, so you're seeing everything in detail!

## Video Files

Videos are saved in MP4 format:
- Standard mode: `videos/humanoid-episode-0.mp4`
- Best episodes mode: `videos/best/best_ep_01_seed042.mp4`

### Convert to GIF

```python
from utils.visualize import create_gif_from_video

create_gif_from_video("videos/humanoid-episode-0.mp4")
```

## Understanding the Output

### During visualization:
```
Running... 100.200.300.400.500. DONE
```
- Numbers show progress every 100 steps
- Episode continues until robot falls or reaches 1000 steps

### Episode summary:
```
Total reward: 523.45
Duration: 876 steps
Result: Fell
```
- **Success**: Reached 1000 steps without falling
- **Fell**: Terminated before 1000 steps

### Best episodes ranking:
```
Top 10 episodes by reward:
   1. Seed  42 | Reward:  1234.56 | Steps: 1000 | Success
   2. Seed  17 | Reward:  1198.23 | Steps: 1000 | Success
   ...
```

## Troubleshooting

### "No window appears"
- Make sure you didn't use `--no-render` flag
- Check that MuJoCo rendering is working: `python -c "import mujoco; print('OK')"`

### "Visualization is too slow"
- This is normal! We slow it down to 0.01s per step for viewing
- For faster evaluation, use `evaluate` command instead

### "Video recording fails"
- Make sure the `videos` folder exists or can be created
- Check disk space
- Try without `--video` first to ensure model works

## Examples

### Compare two models
```bash
# Model 1 - best episodes
python visualize_best.py configs/models/model_v1.zip

# Model 2 - best episodes
python visualize_best.py configs/models/model_v2.zip
```

### Create demo videos
```bash
# Get top 5 performances and save videos
python visualize_best.py
# Enter: 200 total, 5 best, save_video=yes
```

### Quick sanity check
```bash
# Just watch 1 episode
python main.py visualize models/checkpoint.zip --episodes 1
```
