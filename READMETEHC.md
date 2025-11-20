# Poppy Humanoid - Isaac Gym RL Training

Training Poppy Humanoid robot to walk using PPO and Isaac Gym's massively parallel GPU simulation.

## 📁 Project Structure

```
poppy-simulation-team-1/
├── assets/urdf/            # Robot URDF files
├── cfg/
│   ├── task/              # Environment configurations
│   └── train/             # Training hyperparameters
├── tasks/                 # Isaac Gym environments
├── utils/                 # Helper functions
├── train.py               # Main training script
├── eval.py                # Evaluation script
├── run_slurm.sh          # Supercomputer batch script
└── README.md             # This file
```

## 🚀 Quick Start

### 1. Installation (Local - WSL2)

```bash
# Navigate to project in WSL
cd /mnt/c/Users/Utilisateur/GIT/CIA/poppy-simulation-team-1

# Install dependencies
pip install -r requirements.txt

# Isaac Gym should already be installed from earlier
# If not: cd ~/isaacgym/python && pip install -e .
```

### 2. Place Poppy URDF

Copy your `Poppy_Humanoid.URDF` file to:
```bash
cp /path/to/Poppy_Humanoid.URDF assets/urdf/poppy_humanoid.urdf
```

### 3. Test Installation (Local)

```bash
# Quick test with few environments (1-2 minutes)
python train.py --task PoppyHumanoid --num_envs 64 --max_epochs 10
```

If this runs without errors, you're ready for full training!

## 🖥️ Training on Supercomputer

### Transfer Files

```bash
# From WSL, transfer entire project to supercomputer
scp -r /mnt/c/Users/Utilisateur/GIT/CIA/poppy-simulation-team-1 \
    username@supercomputer.address:/path/to/your/home/

# Or use rsync for faster transfer
rsync -avz --progress \
    /mnt/c/Users/Utilisateur/GIT/CIA/poppy-simulation-team-1 \
    username@supercomputer.address:/path/to/your/home/
```

### Setup on Supercomputer

```bash
# SSH into supercomputer
ssh username@supercomputer.address

# Navigate to project
cd poppy-simulation-team-1

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install Isaac Gym (upload the .tar.gz first)
tar -xzvf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
pip install -e .
cd ../..

# Install other dependencies
pip install -r requirements.txt
```

### Launch Training

**Option A: Interactive (for testing)**
```bash
# Request GPU node interactively
srun --gres=gpu:1 --mem=32G --time=2:00:00 --pty bash

# Once on GPU node, run training
python train.py --task PoppyHumanoid --num_envs 4096 --max_epochs 10000
```

**Option B: SLURM Batch Job (recommended)**
```bash
# Edit run_slurm.sh if needed (GPU type, time limit, etc.)
# Then submit job
sbatch run_slurm.sh

# Check job status
squeue -u $USER

# View output logs
tail -f logs/train_JOBID.out
```

## 📊 Monitor Training

### TensorBoard (Local or Supercomputer)

```bash
# Start TensorBoard
tensorboard --logdir=runs --port=6006

# If on supercomputer, forward port via SSH:
# On your local machine:
ssh -L 6006:localhost:6006 username@supercomputer.address

# Then open browser: http://localhost:6006
```

### Check Progress

Training metrics to monitor:
- **Mean reward**: Should increase over time (target: >1000 for walking)
- **Episode length**: Longer episodes = more stable walking
- **Value loss**: Should decrease and stabilize
- **Policy loss**: Should oscillate but trend downward

## 🎮 Evaluate Trained Model

```bash
# Run trained model with visualization (16 parallel robots)
python eval.py \
    --checkpoint runs/PoppyHumanoid_TIMESTAMP/checkpoints/best.pth \
    --num_envs 16 \
    --num_episodes 20

# For deterministic evaluation (no exploration)
python eval.py \
    --checkpoint runs/PoppyHumanoid_TIMESTAMP/checkpoints/best.pth \
    --deterministic
```

## ⚙️ Configuration

### Adjust Number of Environments

Based on GPU memory:
- **RTX 3050** (8GB): 1024-2048 envs
- **RTX 3080** (10GB): 2048-4096 envs
- **A100** (40GB): 4096-8192 envs

Edit `cfg/task/PoppyHumanoid.yaml`:
```yaml
env:
  numEnvs: 4096  # Adjust this
```

Or use command line:
```bash
python train.py --num_envs 2048
```

### Tune Reward Function

Edit `cfg/task/PoppyHumanoid.yaml`:
```yaml
learn:
  rewardScales:
    forwardVel: 2.0    # Increase to encourage faster walking
    upright: 0.5       # Increase for more stable posture
    height: 0.3        # Maintain standing height
    energy: 0.0002     # Increase to reduce torque usage
    actionRate: 0.01   # Increase for smoother motions
```

### Adjust PPO Hyperparameters

Edit `cfg/train/PoppyHumanoidPPO.yaml`:
```yaml
config:
  learning_rate: 3e-4   # Lower if training unstable
  horizon_length: 32    # Longer = more stable but slower
  minibatch_size: 16384 # Adjust based on GPU memory
  mini_epochs: 5        # More epochs = better convergence
```

## 📈 Expected Results

### Training Timeline
- **1-2 hours** (4096 envs on A100): Initial walking behavior emerges
- **4-6 hours**: Stable forward walking
- **8-12 hours**: Optimized gait with good stability

### Performance Metrics
- **Initial reward**: ~50-100 (random actions)
- **Basic walking**: ~500-1000 reward
- **Good walking**: >1500 reward
- **Episode length**: 500-1000 steps for stable walking

## 🐛 Troubleshooting

### Out of Memory Error
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce `--num_envs` or `minibatch_size` in config

### Robot Falls Immediately
**Symptoms**: Episode length < 100 steps, low rewards

**Solutions**:
1. Increase `upright` reward scale
2. Decrease `forwardVel` reward scale initially
3. Check URDF is correct and robot starts upright

### Training Not Improving
**Symptoms**: Reward plateaus, no progress

**Solutions**:
1. Adjust reward scales (may be conflicting objectives)
2. Increase `learning_rate` or use adaptive LR
3. Increase `horizon_length` for more stable gradients
4. Check TensorBoard for value/policy loss trends

### Simulation Unstable
**Symptoms**: Robot vibrates, unrealistic movements

**Solutions**:
1. Increase `actionRate` penalty for smoother actions
2. Decrease control frequency in task config
3. Adjust PD gains in URDF if available

## 📝 Tips for Presentation

### What to Show
1. **Training curves** (TensorBoard screenshots)
2. **Video of trained robot walking** (from eval.py)
3. **Comparison**: Random policy vs trained policy
4. **Metrics**: Reward improvement, episode lengths

### Quick Demo Video
```bash
# Generate evaluation video
python eval.py \
    --checkpoint runs/best_model/checkpoints/best.pth \
    --num_envs 4 \
    --num_episodes 5

# Record screen while running (use OBS, SimpleScreenRecorder, etc.)
```

## 🔧 Advanced: Domain Randomization (Post-Presentation)

For sim-to-real transfer, add to `cfg/task/PoppyHumanoid.yaml`:
```yaml
env:
  randomization:
    mass: [0.8, 1.2]      # ±20% mass variation
    friction: [0.5, 1.5]  # Friction randomization
    motor_strength: [0.9, 1.1]  # Motor variation
```

Implement in `tasks/poppy_humanoid.py` after basic training works.

## 📚 Resources

- **Isaac Gym Docs**: `isaacgym/docs/index.html`
- **RL Games**: https://github.com/Denys88/rl_games
- **PPO Paper**: https://arxiv.org/abs/1707.06347

## 🆘 Need Help?

1. Check logs: `logs/train_JOBID.out` and `logs/train_JOBID.err`
2. TensorBoard for training visualization
3. Test locally with small `--num_envs` before supercomputer
4. Verify URDF loads correctly: `python -c "from tasks.poppy_humanoid import PoppyHumanoid"`

---

**Good luck with your presentation! 🚀🤖**