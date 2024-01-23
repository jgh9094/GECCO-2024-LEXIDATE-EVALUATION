#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:10:00
#SBATCH --mem=1GB
#SBATCH --job-name=collect
#SBATCH -p defq,moore
#SBATCH --exclude=esplhpc-cp040

##################################
# Setup required dependencies
##################################

source /home/hernandezj45/anaconda3/etc/profile.d/conda.sh
conda activate tpot2-env-3.9
pip install -e /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/tpot2-base/

##################################
# Setup relevant directories
##################################


python /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Data-Tools/Collect/collect_all.py