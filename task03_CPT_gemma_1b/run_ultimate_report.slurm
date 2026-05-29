#!/bin/bash
#SBATCH --job-name=run_report
#SBATCH --nodes=1
#SBATCH --partition=p03
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=run_report_log.txt
#SBATCH --error=run_report_err.txt

srun --mpi=pmix \
--chdir ~/ \
--container-image ~/agi/torch_cu30.sqsh \
--container-mount-home \
~/venv/bin/python ~/wisdomCoreV2/task03_CPT_gemma_1b/generate_ultimate_report_and_upload.py
