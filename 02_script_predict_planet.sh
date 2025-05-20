#!/bin/bash 
# #SBATCH -p gypsum-1080ti
# #SBATCH -p gypsum-rtx8000
# #SBATCH -p gypsum-2080ti
# #SBATCH -p gpu
# #SBATCH --constraint=[l40s|a100|a40]
#SBATCH -p gpupod-l40s
#SBATCH -q gpu-quota-16
#SBATCH -A pi_cjgleason_umass_edu
#SBATCH --nodes=1 
#SBATCH -c 4  # Number of Cores per Task
#SBATCH -G 1 # Number of GPUs
#SBATCH --mem=31G 
#SBATCH --time=48:00:00 
#SBATCH --job-name=planet-water-eval
#SBATCH --mail-type=ALL 
#SBATCH -o slurm_planet_eval/planet-water-eval.%j-%a.out


# conda_segment
python 02_predict_w_planet.py