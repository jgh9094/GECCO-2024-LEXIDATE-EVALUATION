#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 240:00:00
#SBATCH --mem=1000GB
#SBATCH --job-name=ran-1
#SBATCH --exclusive
#SBATCH -p defq,moore
#SBATCH --exclude=esplhpc-cp040



source /home/hernandezj45/anaconda3/etc/profile.d/conda.sh
conda activate tpot2-env-3.9
pip install -e /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/tpot2/

python /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Source/main.py \
--n_jobs 48 \
--savepath /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Experiments/Random/Results \
--num_reps 10 \
--sel_scheme 2 \