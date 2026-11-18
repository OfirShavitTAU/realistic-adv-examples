#!/bin/bash

#SBATCH --job-name=breaking-eggs-e2e
#SBATCH --output=logs/e2e_%j.out
#SBATCH --error=logs/e2e_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=15
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=4

mkdir -p logs
mkdir -p /home/sharifm/teaching/mlsec-0368-3423/ofirblechman/.cache

export XDG_CACHE_HOME=/home/sharifm/teaching/mlsec-0368-3423/ofirblechman/.cache
export TORCH_HOME=/home/sharifm/teaching/mlsec-0368-3423/ofirblechman/.cache/torch
export MPLCONFIGDIR=/home/sharifm/teaching/mlsec-0368-3423/ofirblechman/.cache/matplotlib

source /home/sharifm/teaching/mlsec-0368-3423/ofirblechman/miniconda3/etc/profile.d/conda.sh
conda activate /home/sharifm/teaching/mlsec-0368-3423/ofirblechman/conda_envs/breaking-eggs

echo "Running on host: $(hostname)"
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -u -m experiments.initialization_study.tests.e2e.test_e2e