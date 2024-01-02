#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:00:10
#SBATCH --mem=1GB
#SBATCH --job-name=jgh-chk
#SBATCH --array=1-4
#SBATCH -p defq,moore
#SBATCH --exclude=esplhpc-cp040

##################################
# Setup required dependencies
##################################

source /home/hernandezj45/anaconda3/etc/profile.d/conda.sh
conda activate tpot2-env-3.9

##################################
# Setup relevant directories
##################################

DATA_DIR=/home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Results_Prelim/

##################################
# Setup relevant directories
##################################

SELECTION_SCHEME_BASE=1
SELECTION_SCHEME_LEX_10=2
SELECTION_SCHEME_LEX_30=3
SELECTION_SCHEME_LEX_50=4

# what are we checking

if [ ${SLURM_ARRAY_TASK_ID} -eq ${SELECTION_SCHEME_BASE} ] ; then
  EXPERIMENT=0
  SEED=0
elif [ ${SLURM_ARRAY_TASK_ID} -eq ${SELECTION_SCHEME_LEX_10} ] ; then
  EXPERIMENT=1
  SEED=1200
elif [ ${SLURM_ARRAY_TASK_ID} -eq ${SELECTION_SCHEME_LEX_30} ] ; then
  EXPERIMENT=2
  SEED=2400
elif [ ${SLURM_ARRAY_TASK_ID} -eq ${SELECTION_SCHEME_LEX_30} ] ; then
  EXPERIMENT=3
  SEED=3600
else
  echo "${SEED} from ${PROBLEM} failed to launch" >> /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Experiments/failtolaunch.txt
fi

# let it rip

NUM_REPS=30

python /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Data-Tools/Cleaner/checker.py \
--data_dir ${DATA_DIR} \
--num_reps ${NUM_REPS} \
--seed ${SEED} \
--experiment ${EXPERIMENT} \