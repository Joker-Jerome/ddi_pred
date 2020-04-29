#!/bin/bash
#SBATCH --output dsq-tuning_task_dnn-%A_%1a-%N.out
#SBATCH --array 0-4
#SBATCH --job-name dnn_tuning_learning_rate
#SBATCH --mem-per-cpu=16G -t 24:00:00 -p bigmem,day

# DO NOT EDIT LINE BELOW
/gpfs/loomis/apps/avx/software/dSQ/0.96/dSQBatch.py /gpfs/loomis/pi/zhao2/zy92/projects/ddipred/ddi_pred/code/dnn/submit/tuning_task_dnn.txt /gpfs/loomis/pi/zhao2/zy92/projects/ddipred/ddi_pred/code/dnn/submit

