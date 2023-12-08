#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -t 00:00:10
#SBATCH --mem=1GB
#SBATCH --job-name=jgh-chk
#SBATCH --array=1-3
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

DATA_DIR=/home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Experiments

##################################
# Setup relevant directories
##################################

SELECTION_SCHEME_LEXICASE=1
SELECTION_SCHEME_TOURNAMENT=2
SELECTION_SCHEME_RANDOM=3

# what are we checking

if [ ${SLURM_ARRAY_TASK_ID} -eq ${SELECTION_SCHEME_LEXICASE} ] ; then
  SELECTION=0

elif [ ${SLURM_ARRAY_TASK_ID} -eq ${SELECTION_SCHEME_TOURNAMENT} ] ; then
  SELECTION=1

elif [ ${SLURM_ARRAY_TASK_ID} -eq ${SELECTION_SCHEME_RANDOM} ] ; then
  SELECTION=2

else
  echo "${SEED} from ${PROBLEM} failed to launch" >> /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Experiments/failtolaunch.txt
fi

# let it rip

NUM_REPS=10

python /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Data-Tools/Cleaner/checker.py \
--data_dir ${DATA_DIR} \
--num_reps ${NUM_REPS} \
--scheme ${SELECTION} \