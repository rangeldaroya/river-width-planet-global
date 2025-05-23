#!/bin/bash 
#SBATCH -o slurm_tile_planet-err2/tile-planet.%j-%a.out 
#SBATCH --mail-type=ALL 
#SBATCH --partition=cpu
#SBATCH --nodes=1 
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=16G 
#SBATCH --time=01:00:00 
#SBATCH --job-name=tile-planet
# #SBATCH --array=1-121
#SBATCH --array=8,14,21,33,39,59,60,75,79,84

num=$((SLURM_ARRAY_TASK_ID))

python 01_make_tiles_planet.py --idx ${num}