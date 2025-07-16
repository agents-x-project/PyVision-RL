srun -p Gveval-T --gres=gpu:8 --cpus-per-task=1 -n1 --ntasks-per-node=1 --quotatype=reserved --job-name=pyvision \
vllm serve /mnt/petrelfs/zhaoshitian/gveval_zhaoshitian/Qwen2.5-72B-Instruct \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 8 \
    --served-model-name "judge" \
    --trust-remote-code \
    --disable-log-requests