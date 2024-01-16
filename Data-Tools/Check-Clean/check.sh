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

DATA_DIR=/home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Results/

##################################
# Setup relevant directories
##################################

SELECTION_SCHEME_BASE=1
SELECTION_SCHEME_LEX_50=2
SELECTION_SCHEME_LEX_30=3
SELECTION_SCHEME_LEX_10=4

# what are we checking

if [ ${SLURM_ARRAY_TASK_ID} -eq ${SELECTION_SCHEME_BASE} ] ; then
  EXPERIMENT=0
  SEED=5000

elif [ ${SLURM_ARRAY_TASK_ID} -eq ${SELECTION_SCHEME_LEX_50} ] ; then
  EXPERIMENT=1
  SEED=5900

elif [ ${SLURM_ARRAY_TASK_ID} -eq ${SELECTION_SCHEME_LEX_30} ] ; then
  EXPERIMENT=2
  SEED=5600

elif [ ${SLURM_ARRAY_TASK_ID} -eq ${SELECTION_SCHEME_LEX_10} ] ; then
  EXPERIMENT=3
  SEED=5300

else
  echo "${SEED} from ${PROBLEM} failed to launch" >> /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Experiments/failtolaunch.txt
fi

# let it rip

NUM_REPS=40

python /home/hernandezj45/Repos/GECCO-2024-TPOT2-Selection-Objectives/Data-Tools/Check-Clean/check.py \
-d ${DATA_DIR} \
-r ${NUM_REPS} \
-s ${SEED} \
-e ${EXPERIMENT} \