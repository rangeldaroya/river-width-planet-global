#!/bin/bash 
#SBATCH -o slurm_width_planet_err/width-planet.%j-%a.out 
#SBATCH --mail-type=ALL 
#SBATCH --partition=cpu
#SBATCH --nodes=1 
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=16G 
#SBATCH --time=04:00:00 
#SBATCH --job-name=width-planet
#SBATCH --array=0-122

num=$((SLURM_ARRAY_TASK_ID))

python 05_effective_widths.py --idx ${num}