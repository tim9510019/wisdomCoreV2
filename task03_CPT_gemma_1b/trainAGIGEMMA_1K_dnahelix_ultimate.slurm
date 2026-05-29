#!/bin/bash
#SBATCH --job-name=g3_ultimate
#SBATCH --nodes=1
#SBATCH --partition=p03
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=train_log_dnahelix_ultimate_scratch.txt
#SBATCH --error=train_err_dnahelix_ultimate_scratch.txt

srun --mpi=pmix \
--chdir ~/ \
--container-image ~/agi/torch_cu30.sqsh \
--container-mount-home \
~/venv/bin/python ~/wisdomCoreV2/task03_CPT_gemma_1b/trainCPTAC_AGI_GEMMA3_1K_dnahelix_ultimate_scratch.py
