#!/bin/bash
#SBATCH --job-name training_set_surrogate_model_MRST_%A_%a
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=serc
#SBATCH --time=10:00:00
#SBATCH -o /oak/stanford/schools/ees/jcaers/mzechner/MRST_CCS/eng_geo_no_global/output_cluster/error_output_Run_%A_%a.out
#SBATCH -e /oak/stanford/schools/ees/jcaers/mzechner/MRST_CCS/eng_geo_no_global/output_cluster/error_output_Run_%A_%a.err
#SBATCH --mail-user=mzechner@stanford.edu
#SBATCH --array=1-2000



cd /home/users/mzechner/code-dev/CCS_dev/CCS-core/matlab/MRST_training_samples_surrogate/


echo -n 'Job is running on node '
echo $SLURM_JOB_NODELIST


module load matlab
matlab -nodisplay < simulation_master_with_eng_params_JobArray.m
