#!/bin/bash 
#SBATCH -o slurm_width_planet_nodepoly/width-planet.%j-%a.out 
#SBATCH --mail-type=ALL 
#SBATCH --partition=cpu
#SBATCH --nodes=1 
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=16G 
#SBATCH --time=04:00:00 
#SBATCH --job-name=width-planet-nodepoly
#SBATCH --array=0-122

num=$((SLURM_ARRAY_TASK_ID))

python 06_new_effective_width_node_poly.py --idx ${num}