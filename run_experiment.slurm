#!/bin/bash

#SBATCH --job-name=breaking-eggs-experiment
#SBATCH --output=logs/exp_%j.out
#SBATCH --error=logs/exp_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=60
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

mkdir -p logs
mkdir -p /home/sharifm/teaching/mlsec-0368-3423/ofirblechman/.cache

export XDG_CACHE_HOME=/home/sharifm/teaching/mlsec-0368-3423/ofirblechman/.cache
export TORCH_HOME=/home/sharifm/teaching/mlsec-0368-3423/ofirblechman/.cache/torch
export MPLCONFIGDIR=/home/sharifm/teaching/mlsec-0368-3423/ofirblechman/.cache/matplotlib

source /home/sharifm/teaching/mlsec-0368-3423/ofirblechman/miniconda3/etc/profile.d/conda.sh
conda activate /home/sharifm/teaching/mlsec-0368-3423/ofirblechman/conda_envs/breaking-eggs

echo "Running on host: $(hostname)"
nvidia-smi

python -u -m experiments.initialization_study.run.run_experiment