#!/bin/bash
#SBATCH --job-name=ray-job
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --partition=Gveval-T
#SBATCH --quotatype=reserved
#SBATCH --output=/mnt/petrelfs/zhaoshitian/vis_tool_train/logs/pyvision-rl-%j.log

export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5

set -xe
head_node_ip=10.140.66.34


RAY_ADDRESS='http://10.140.66.34:8265' ray job submit --working-dir . -- bash /mnt/petrelfs/zhaoshitian/vis_tool_train/verl_agents/slurm_scripts/run_train_single_node.sh