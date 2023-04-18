#!/bin/bash
#SBATCH --job-name=training_set_surrogate_model_MRST_LOOP_%A_%a
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc
#SBATCH --time=10:00:00
#SBATCH -o /oak/stanford/schools/ees/jcaers/mzechner/MRST_CCS/eng_geo_no_global/output_cluster/error_output_Run_%A_%a.out
#SBATCH -e /oak/stanford/schools/ees/jcaers/mzechner/MRST_CCS/eng_geo_no_global/output_cluster/error_output_Run_%A_%a.err
#SBATCH --mail-type=ALL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mzechner@stanford.edu
#SBATCH --array=1-5

# This is an example script that combines array tasks with
# bash loops to process many short runs. Array jobs are convenient
# for running lots of tasks, but if each task is short, they
# quickly become inefficient, taking more time to schedule than
# they spend doing any work and bogging down the scheduler for
# all users.
pwd; hostname; date

#Set the number of runs that each SLURM task should do
PER_TASK=10

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
START_NUM=$(( ($SLURM_ARRAY_TASK_ID - 1) * $PER_TASK + 1 ))
END_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

# Run the loop of runs for this task.
for (( run=$START_NUM; run<=END_NUM; run++ )); do
  echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run

  cd /home/users/mzechner/code-dev/CCS_dev/CCS-core/matlab/MRST_training_samples_surrogate/
  module load matlab

  matlab  -nodisplay -r "simulation_master_with_eng_params_loop($run)"

done
