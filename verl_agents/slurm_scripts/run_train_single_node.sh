export LLM_AS_A_JUDGE_BASE="http://10.140.60.106:18901/v1" # 10-140-1-174
export no_proxy='10.140.60.106:18901'

# umber of training nodes
export WORLD_SIZE=1

# srun -p eaigc1_t  --gres=gpu:8 --cpus-per-task=1 -N2 --ntasks-per-node=8 --quotatype=reserved --job-name=pyvision \
bash /mnt/petrelfs/zhaoshitian/vis_tool_train/verl_agents/examples/agent/final_merged_v1v8_thinklite.sh