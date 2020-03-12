#!/bin/bash
#SBATCH --output dsq-tuning_task-%A_%2a-%N.out
#SBATCH --array 0-19
#SBATCH --job-name tuning_xgb
#SBATCH --mem-per-cpu=16G -t 5:00:00 -p bigmem,pi_zhao,general,scavenge
#SBATCH --requeue

# DO NOT EDIT LINE BELOW
/ysm-gpfs/apps/software/dSQ/0.96/dSQBatch.py /gpfs/loomis/pi/zhao2/zy92/projects/ddipred/ddi_pred/code/xgboost/submit/tuning_task.txt /gpfs/loomis/pi/zhao2/zy92/projects/ddipred/ddi_pred/code/xgboost/submit

