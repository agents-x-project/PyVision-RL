#!/bin/bash
#SBATCH --job-name=ray-header
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=0
#SBATCH --ntasks-per-node=1
#SBATCH --partition=Gveval-T
#SBATCH --quotatype=reserved
#SBATCH --nodelist=HOST-10-140-66-34

set -xe
# 自定义路径，过长或者机器上已有此目录会报错
RAYLOG="/tmp/ma-ray"

# 获取节点列表
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# 获取节点ip
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# 如果机器开启IPv6会执行 确保获取IPv4地址
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --temp-dir=$RAYLOG \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block --dashboard-host=0.0.0.0 --disable-usage-stats

sleep infinity