#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -A mp309
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/flash_%j.log
#SBATCH --error=logs/flash_err_%j.log

# make sure you're in the scripts directory when running this
module load conda
conda activate ring

set -x

cd "$SLURM_SUBMIT_DIR"
srun python ../assert_flash.py  --causal --cuda-kernel
