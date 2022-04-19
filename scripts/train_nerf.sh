#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=4-00:00:00
#SBATCH --job-name=train_nerf
#SBATCH --output=nerf.log

# Discord notifs on start and end
source notify

# Fail on error
set -e

# Copy dataset
rsync -azh --info=progress2 ada.iiit.ac.in:/share3/ishaanshah/nerf_dataset /ssd_scratch/cvit/ishaanshah/

# Activate Conda environment
source /home2/ishaanshah/anaconda3/bin/activate SemGCN

# Training script
pushd ~/nerf
wandb on

args=("$@")
python main.py /ssd_scratch/cvit/ishaanshah/nerf_dataset/nerf_synthetic/$1 --gpus=1 --chunk_size 16384 "${args[@]:1}"

popd
