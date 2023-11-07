#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 1:00:00
#SBATCH --mem=0
#SBATCH --job-name=tpot2
#SBATCH -p defq
#SBATCH --exclusive
#SBATCH --exclude=esplhpc-cp040

#SBATCH -o ./logs/output.%j.out # STDOUT

source /home/hernandezj45/anaconda3/etc/profile.d/conda.sh
conda activate tpot2-env
pip install -e ../../../tpot2/

srun -u python ../../../Source/main.py \
--n_jobs 48 \
--savepath ../Results/ \
--num_reps 10 \
--sel_scheme 0 \