#!/bin/bash

#SBATCH --job-name=breaking-eggs-preview
#SBATCH --output=logs/preview_%j.out
#SBATCH --error=logs/preview_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=15
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

mkdir -p logs
source /home/sharifm/teaching/mlsec-0368-3423/ofirblechman/miniconda3/etc/profile.d/conda.sh
conda activate /home/sharifm/teaching/mlsec-0368-3423/ofirblechman/conda_envs/breaking-eggs

echo "Running on host: $(hostname)"
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -u -m experiments.initialization_study.run.run_preview