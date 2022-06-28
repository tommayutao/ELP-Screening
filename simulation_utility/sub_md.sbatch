#!/bin/bash
#SBATCH --job-name=ELP_md
#SBATCH --partition=fela
##SBATCH --qos=gm4
#SBATCH --output=md.out
#SBATCH --nodes=1            # SET NUM NODES 
#SBATCH --ntasks-per-node=1  # SETS NUM MPI RANKS (1 PER GPU)
#SBATCH --cpus-per-task=10   # SET NUM THREADS 
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
##SBATCH --nodelist=midway2-0740

# SET NUMBER OF MPI TASKS 
NTASKS=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))
# SET NUMBER OF OPENMP THREADS
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load gromacs/2019.3+intelmpi-2018.2.199+intel-18.0
#'nvidia-smi'

gmx_mpi grompp -f mdp_files/em.mdp -c bilayer_ions.gro -p system.top -o em.tpr
time gmx_mpi mdrun -v -deffnm em -ntomp $OMP_NUM_THREADS

gmx_mpi grompp -f mdp_files/equilibration_nvt.mdp -c em.gro -p system.top -o nvt.tpr
time gmx_mpi mdrun -v -deffnm nvt -ntomp $OMP_NUM_THREADS

gmx_mpi grompp -f mdp_files/equilibration_npt.mdp -c nvt.gro -p system.top -t nvt.cpt -o npt.tpr
time gmx_mpi mdrun -v -deffnm npt -ntomp $OMP_NUM_THREADS

gmx_mpi grompp -f mdp_files/dynamics.mdp -c npt.gro -t npt.cpt -p system.top -o md.tpr
time gmx_mpi mdrun -v -deffnm md -ntomp $OMP_NUM_THREADS
