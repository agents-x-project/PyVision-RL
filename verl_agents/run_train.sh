#!/bin/bash
#SBATCH --job-name=pv-rl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --partition=eaigc1_t
#SBATCH --quotatype=reserved

export LLM_AS_A_JUDGE_BASE="http://10.140.60.133:18901/v1" # 10-140-1-174
export no_proxy='10.140.60.133:18901'

# umber of training nodes
export WORLD_SIZE=1

# config for 7B
bash examples/agent/final_merged_v1v8_thinklite_single_node.sh

# config for 32B
# bash examples/agent/final_merged_v1v8_thinklite_32b.sh