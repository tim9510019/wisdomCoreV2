#!/bin/bash
#SBATCH --job-name=skill_eval
#SBATCH --nodes=1
#SBATCH --partition=p03
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=skill_eval_scratch_log.txt
#SBATCH --error=skill_eval_scratch_err.txt

srun --mpi=pmix \
--chdir ~/ \
--container-image ~/agi/torch_cu30.sqsh \
--container-mount-home \
bash -c "~/venv/bin/python ~/wisdomCoreV2/task03_CPT_gemma_1b/skill_eval_scratch_5way.py && ~/venv/bin/python ~/wisdomCoreV2/task03_CPT_gemma_1b/calculate_usf_scores.py"

