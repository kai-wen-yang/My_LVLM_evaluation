#!/bin/bash
# Supply script name as parameter and names of experiments you want to run
# set +x
echo "Submitting to sbatch $1"
chmod +x $1
SCRIPT_NAME=$(basename $1 .sh)

DATE=$(date "+%m%d%y_%H_%M_%S")
DATE_DAY_ONLY=$(date "+%m%d%y")
OUTPUT_DIR="slurm_output/$SCRIPT_NAME/$DATE_DAY_ONLY/"
mkdir -p $OUTPUT_DIR
sbatch --account=nexus \
--qos=medium \
--time 24:00:00 \
--gres=gpu:rtxa6000:1 \
--job-name=$1 \
--mem=32g \
-n 2 \
--output="slurm_output/$SCRIPT_NAME/$DATE_DAY_ONLY/${DATE}_misc_explore.out" \
--error="slurm_output/$SCRIPT_NAME/$DATE_DAY_ONLY/${DATE}_output_explore.out" \
$1