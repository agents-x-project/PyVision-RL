#!/bin/bash
#SBATCH --job-name=ray-job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --partition=Gveval-T
#SBATCH --quotatype=spot
#SBATCH --time=1200

set -xe
head_node_ip=10.140.60.38


RAY_ADDRESS='http://10.140.60.38:8265' ray job submit --working-dir . -- bash /mnt/petrelfs/zhaoshitian/vis_tool_train/verl_agents/slurm_scripts/run_train_single_node.sh