#!/bin/bash
#SBATCH --job-name=judge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --no-requeue
#SBATCH --partition=eaigc1_t
#SBATCH --quotatype=reserved

vllm serve /mnt/petrelfs/zhaoshitian/gveval_zhaoshitian/Qwen2.5-72B-Instruct \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 4 \
    --served-model-name "judge" \
    --trust-remote-code \
    --disable-log-requests