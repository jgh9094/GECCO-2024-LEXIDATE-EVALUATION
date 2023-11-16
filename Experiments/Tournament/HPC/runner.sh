#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH -t 240:00:00
#SBATCH --mem=0
#SBATCH --job-name=jgh-tor
#SBATCH -p defq,moore
#SBATCH --exclude=esplhpc-cp040



source /home/hernandezj45/anaconda3/etc/profile.d/conda.sh
conda activate tpot2-env
pip install -e /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/tpot2/

python /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Source/main.py \
--n_jobs 20 \
--savepath /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Experiments/Tournament/Results \
--num_reps 10 \
--sel_scheme 1 \