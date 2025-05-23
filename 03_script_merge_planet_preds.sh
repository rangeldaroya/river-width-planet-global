#!/bin/bash 
#SBATCH -o slurm_merge_planet_v2/merge-planet.%j-%a.out 
#SBATCH --mail-type=ALL 
#SBATCH --partition=cpu
#SBATCH --nodes=1 
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=16G 
#SBATCH --time=01:00:00 
#SBATCH --job-name=merge-planet
#SBATCH --array=0-122

num=$((SLURM_ARRAY_TASK_ID))

python 03_merge_planet_preds.py --idx ${num}