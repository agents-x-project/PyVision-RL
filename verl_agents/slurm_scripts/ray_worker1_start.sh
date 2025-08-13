#!/bin/bash
#SBATCH --job-name=ray-worker1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --partition=Gveval-T
#SBATCH --quotatype=reserved


set -xe

echo "Run Node: $SLURM_NODELIST"
# 自定义路径，过长或者机器上已有此目录会报错
RAYLOG="/tmp/ma-ray"
echo "RAYLOG $RAYLOG"


head_node_ip=10.140.66.34

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

ray start --address "$ip_head" --temp-dir=$RAYLOG --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block

sleep infinity