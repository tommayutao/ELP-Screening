#!/bin/bash
#SBATCH --job-name=XXX
#SBATCH --output=XXX/XXX.out
#SBATCH --partition=fela
##SBATCH --qos=gm4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --constraint=gpu

NTASKS=$(($SLURM_NTASKS_PER_NODE * $SLURM_JOB_NUM_NODES))
OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load gromacs/2019.3+intelmpi-2018.2.199+intel-18.0
#'nvidia-smi'

cd XXX

gmx_mpi grompp -f ../XXX_npt.mdp -c ../YYY -p ../../system.top -n ../../index.ndx -o XXX_npt.tpr -po mdout_npt.mdp
gmx_mpi mdrun -v -deffnm XXX_npt -ntomp $OMP_NUM_THREADS
gmx_mpi grompp -f ../XXX_md.mdp -c XXX_npt.gro -t XXX_npt.cpt -p ../../system.top -n ../../index.ndx -o XXX_md.tpr -po mdout_md.mdp
gmx_mpi mdrun -v -deffnm XXX_md -ntomp $OMP_NUM_THREADS

mv ../sub_XXX.sbatch .
mv ../XXX_npt.mdp .
mv ../XXX_md.mdp .
