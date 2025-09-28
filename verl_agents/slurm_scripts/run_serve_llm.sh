srun -p Gveval-T --gres=gpu:2 --cpus-per-task=1 -n1 --ntasks-per-node=1 --quotatype=spot --job-name=pyvision \
vllm serve /mnt/petrelfs/zhaoshitian/eaigc3_t_zhaoshitian/models/DeepEyes-7B \
    --port 18800 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 2 \
    --served-model-name "qwen" \
    --trust-remote-code \
    --disable-log-requests \
    --limit-mm-per-prompt "image=50" &