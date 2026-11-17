#!/bin/bash
#SBATCH --job-name=breaking-eggs-experiment
#SBATCH --output=logs/exp_%j.out
#SBATCH --error=logs/exp_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=60
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

mkdir -p logs
source /home/sharifm/teaching/mlsec-0368-3423/ofirblechman/miniconda3/etc/profile.d/conda.sh
conda activate /home/sharifm/teaching/mlsec-0368-3423/ofirblechman/conda_envs/breaking-eggs

echo "Running on host: $(hostname)"
nvidia-smi

# דגל -u מבטיח שכל פקודת print תופיע מיד בקובץ ה-out בזמן אמת
python -u -m experiments.initialization_study.run.run_experiment