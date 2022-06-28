#!/bin/bash
#SBATCH --job-name=wham_setup
#SBATCH --output=setup_wham.out
#SBATCH --partition=fela-cpu
##SBATCH --qos=gm4-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
##SBATCH --mem=64G
##SBATCH --nodelist=midway2-0740

source /home/yma3/hoomd_env/bin/activate
python3 setup_wham2d.py wham_whole_ub
mv setup_wham.out wham_whole_ub
