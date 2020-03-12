#!/bin/bash
#SBATCH --output dsq-tuning_task-%A_%2a-%N.out
#SBATCH --array 0-19
#SBATCH --job-name tuning_xgb_100
#SBATCH --mem-per-cpu=8G -t 12:00:00 -p bigmem,day

# DO NOT EDIT LINE BELOW
/gpfs/loomis/apps/avx/software/dSQ/0.96/dSQBatch.py /gpfs/loomis/pi/zhao2/zy92/projects/ddipred/ddi_pred/code/xgboost/submit/tuning_task.txt /gpfs/loomis/pi/zhao2/zy92/projects/ddipred/ddi_pred/code/xgboost/submit

