#!/bin/bash
#SBATCH --job-name=poppy_train      # Job name
#SBATCH --output=logs/train_%j.out  # Output file (%j = job ID)
#SBATCH --error=logs/train_%j.err   # Error file
#SBATCH --time=24:00:00             # Time limit (24 hours)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks-per-node=1         # Tasks per node
#SBATCH --cpus-per-task=8           # CPUs per task
#SBATCH --gres=gpu:1                # Number of GPUs (adjust based on availability)
#SBATCH --mem=32G                   # Memory per node

# CUSTOMIZE THESE BASED ON YOUR SUPERCOMPUTER:
# - Replace "gpu:1" with specific GPU type if needed (e.g., "gpu:a100:1" or "gpu:v100:1")
# - Adjust --mem based on available memory
# - Modify --time based on queue limits
# - Add partition/account if required: #SBATCH --partition=gpu or #SBATCH --account=your_account

echo "========================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "========================================="

# Create logs directory
mkdir -p logs

# Load modules (CUSTOMIZE FOR YOUR SYSTEM)
# Common module loads for supercomputers:
# module load python/3.8
# module load cuda/11.8
# module load cudnn/8.6

# Activate virtual environment if using one
# source venv/bin/activate

# Or use conda
# conda activate poppy_env

# Print GPU info
nvidia-smi

echo ""
echo "Starting training..."
echo ""

# Run training with desired number of environments
# Adjust --num_envs based on GPU memory:
# - RTX 3080/3090: 2048-4096 envs
# - A100: 4096-8192 envs
# - V100: 2048-4096 envs

python train.py \
    --task PoppyHumanoid \
    --num_envs 1000 \
    --max_epochs 10000 \
    --experiment poppy_${SLURM_JOB_ID}

echo ""
echo "========================================="
echo "Job completed at $(date)"
echo "========================================="