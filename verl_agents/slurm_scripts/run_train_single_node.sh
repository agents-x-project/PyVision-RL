# export http_proxy=http://zhaoshitian:1gEBQdKIt8bvEDpX13EWdyThtw00WazVwRKeZAAkhx2MPoUJfUYnqXrRFTfn@10.1.20.51:23128/
# export https_proxy=https://zhaoshitian:1gEBQdKIt8bvEDpX13EWdyThtw00WazVwRKeZAAkhx2MPoUJfUYnqXrRFTfn@10.1.20.51:23128/
# export HTTP_PROXY=http://zhaoshitian:1gEBQdKIt8bvEDpX13EWdyThtw00WazVwRKeZAAkhx2MPoUJfUYnqXrRFTfn@10.1.20.51:23128/
# export HTTPS_PROXY=https://zhaoshitian:1gEBQdKIt8bvEDpX13EWdyThtw00WazVwRKeZAAkhx2MPoUJfUYnqXrRFTfn@10.1.20.51:23128/
# export LLM_AS_A_JUDGE_BASE="http://10.140.60.133:18901/v1" # 10-140-1-174
# export no_proxy='10.140.60.133:18901'
# export HYDRA_FULL_ERROR=1
# export WANDB_MODE=offline

# umber of training nodes
export WORLD_SIZE=2

# srun -p eaigc1_t  --gres=gpu:8 --cpus-per-task=1 -N2 --ntasks-per-node=8 --quotatype=reserved --job-name=pyvision \
bash /mnt/petrelfs/zhaoshitian/vis_tool_train/verl_agents/examples/agent/train_pyvision_rl_7b_v3.sh