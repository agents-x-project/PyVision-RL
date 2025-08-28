#!/bin/bash
#SBATCH --job-name=judge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --requeue
#SBATCH --partition=Gveval-T
#SBATCH --quotatype=reserved

# export CUDA_VISIBLE_DEVICES=4,5,6,7

vllm serve /mnt/petrelfs/zhaoshitian/gveval_zhaoshitian/Qwen2.5-72B-Instruct \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 4 \
    --served-model-name "judge" \
    --trust-remote-code \
    --disable-log-requests