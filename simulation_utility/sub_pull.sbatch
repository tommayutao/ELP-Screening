#!/bin/bash
#SBATCH --job-name=pull
#SBATCH --partition=gm4-pmext
#SBATCH --qos=gm4
#SBATCH --output=pull.out
#SBATCH --nodes=1            # SET NUM NODES 
#SBATCH --ntasks-per-node=1  # SETS NUM MPI RANKS (1 PER GPU)
#SBATCH --cpus-per-task=10   # SET NUM THREADS 
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu
##SBATCH --nodelist=midway2-0740

# SET NUMBER OF MPI TASKS 
NTASKS=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))
# SET NUMBER OF OPENMP THREADS
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load gromacs/2019.3+intelmpi-2018.2.199+intel-18.0
#'nvidia-smi'

gmx_mpi grompp -f mdp_files/pull1.mdp -c md.gro -t md.cpt -n index.ndx -p system.top -o pull1.tpr
time gmx_mpi mdrun -v -deffnm pull1 -ntomp $OMP_NUM_THREADS

gmx_mpi grompp -f mdp_files/pull2.mdp -c pull1.gro -t pull1.cpt -n index.ndx -p system.top -o pull2.tpr 
time gmx_mpi mdrun -v -deffnm pull2 -ntomp $OMP_NUM_THREADS

gmx_mpi grompp -f mdp_files/pull3.mdp -c pull2.gro -t pull2.cpt -n index.ndx -p system.top -o pull3.tpr 
time gmx_mpi mdrun -v -deffnm pull3 -ntomp $OMP_NUM_THREADS
