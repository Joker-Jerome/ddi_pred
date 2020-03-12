#!/bin/bash
#SBATCH --output dsq-tuning_task_left-%A_%1a-%N.out
#SBATCH --array 0-8
#SBATCH --job-name scv_left_tuning_xgb
#SBATCH --mem-per-cpu=8G -t 4:00:00 -p scavenge

# DO NOT EDIT LINE BELOW
/gpfs/loomis/apps/avx/software/dSQ/0.96/dSQBatch.py /gpfs/loomis/pi/zhao2/zy92/projects/ddipred/ddi_pred/code/xgboost/submit/tuning_task_left.txt /gpfs/loomis/pi/zhao2/zy92/projects/ddipred/ddi_pred/code/xgboost/submit

