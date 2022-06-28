#!/bin/bash
#SBATCH --job-name=ub_setup
#SBATCH --partition=broadwl-lc
##SBATCH --qos=gm4
#SBATCH --output=ub_setup.out
#SBATCH --nodes=1            # SET NUM NODES 
#SBATCH --ntasks-per-node=1  # SETS NUM MPI RANKS (1 PER GPU)
#SBATCH --cpus-per-task=8   # SET NUM THREADS 
#SBATCH --mail-type=ALL
##SBATCH --gres=gpu:1
##SBATCH --nodelist=midway2-0740

# SET NUMBER OF MPI TASKS 
NTASKS=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))
# SET NUMBER OF OPENMP THREADS
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load gromacs
#'nvidia-smi'
source /home/yma3/hoomd_env/bin/activate
python3 setup_ub_sampling.py
