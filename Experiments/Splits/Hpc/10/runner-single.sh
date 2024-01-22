#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 384:00:00
#SBATCH --mem=1000GB
#SBATCH --job-name=lex-1r
#SBATCH --exclusive
#SBATCH -p moore,defq
#SBATCH --exclude=esplhpc-cp040

source /home/hernandezj45/anaconda3/etc/profile.d/conda.sh
conda activate tpot2-env-3.9
pip install -e /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/tpot2-sel-obj/

SAVE_DIR=/home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Results/Lex-10
mkdir -p ${SAVE_DIR}

python /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Source/single_runs_sel_obj.py \
--n_jobs 48 \
--savepath ${SAVE_DIR} \