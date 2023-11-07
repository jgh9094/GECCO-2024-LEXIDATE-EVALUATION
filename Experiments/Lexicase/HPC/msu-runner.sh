#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -t 00:10:00
#SBATCH --mem=10G
#SBATCH --job-name=tpot2

conda activate tpot2-env
pip install -e ../../../tpot2/

srun -u python ../../../Source/main.py \
--n_jobs 48 \
--savepath ../Results/ \
--num_reps 10 \
--sel_scheme 0 \