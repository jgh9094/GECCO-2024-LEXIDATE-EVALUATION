#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 240:00:00
#SBATCH --mem=1000GB
#SBATCH --job-name=lex-5
#SBATCH --exclusive
#SBATCH -p defq,moore
#SBATCH --exclude=esplhpc-cp040

source /home/hernandezj45/anaconda3/etc/profile.d/conda.sh
conda activate tpot2-env-3.9
pip install -e /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/tpot2/

DATA_DIR=/home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Results_1/50/Lexicase
mkdir -p ${DATA_DIR}

python /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Source/main-sel-obj.py \
--n_jobs 48 \
--savepath ${SAVE_DIR} \
--num_reps 10 \
--sel_scheme 0 \
--proportion .50 \
--seed_offset 2000 \