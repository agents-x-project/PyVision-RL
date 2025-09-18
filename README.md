# PyVision RL Training

### Prepare the RL data and SFT ckpts

SFT ckpts: https://huggingface.co/Agents-X/sft-data-v1-Qwen2.5-VL-7B-1epoch

RL data: https://huggingface.co/datasets/Agents-X/de_visual_search_filtered

### Installation
see https://github.com/Visual-Agent/DeepEyes

```bash
####################################################
# you should just follow deepeyes and ignore below.#
####################################################
conda create -n pyvision-agent python=3.10
conda activate pyvision-agent

# if your cuda version is 12.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
# if fail to install pip install vllm==0.8.3
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
pip install vllm

pip install qwen-vl-utils[decord]==0.0.8

# pip install -r env_files/requirements_agent.txt
# Follow the VeRL official installation procedure
pip install -e .
bash ./verl_agents/scripts/install_deepeyes.sh
```

### Serve Qwen2.5-72B-Instruct for LLM-as-a-Judge
```bash
cd verl_agents
sbatch ./slurm_scripts/sbatch_serve_llm.sh
```
Test if the served llm works:
```bash
python test_call_qwen_serve.py
```

### Train
After serving the llm-as-a-judge and making sure it works well, you could start to train. 
Note: I strictly followed `verl`'s doc to start the multi-node RL training. If you want more detail, please check `verl`'s doc.

#### Single Node
```bash
# TODO
bash run_train.sh
```

#### Multiple Nodes
First, setup a ray cluster, including the head cluster (pure CPU) and worker cluster (GPU nodes). For example, if you want to train on 16 GPUs, you need to start two GPU nodes as the ray worker clusters.
```bash
# Start the ray head cluster
sbatch ./slurm_scripts/ray_head_start.sh
```

Then, setup the ray worker cluster. After starting the ray head cluster, you need to acqure the head IP address, and put it into `./slurm_scripts/ray_worker1_start.sh` and `./slurm_scripts/ray_worker2_start.sh`
```bash
# Start the first worker cluster.
sbatch ./slurm_scripts/ray_worker1_start.sh

# Start the second worker cluster.
sbatch ./slurm_scripts/ray_worker2_start.sh
```
After setup the ray head cluster and worker cluser, you need to go to the ray dash board and check the resources.

Then, you need to change the llm-as-a-judge IP address to the Python file: `./verl_agents/verl/utils/reward_score/vl_agent.py`

Finally, start training.
```bash
sbatch ./slurm_scripts/ray_train.sh
```

### Evaluation

#### Model Merge
```bash
bash scripts/run_merge.sh
```

#### Test
see https://github.com/agents-x-project/TIR-Data-Synthesis

### Some details about this codebase.
1. Where is the agent inference code?
- ./verl_agents/verl/workers/agent/envs/agents_x
- ./verl_agents/verl/workers/agent/parallel_env.py

2. Where is the partial dynamic filtering code?
- ./verl_agents/verl/trainer/ppo/ray_trainer.py (L1089 ~ L1136)

3. Where is the reward function definition code?
- ./verl_agents/verl/utils/reward_score/vl_agent.py (L214 ~ L316)

4. There are some thicky points about calling llm-as-a-judge serving API and logging the training record to WandB.
- If your machine does not need a proxy to access the internet, just ignore this section. If not, remember, to call the llm-as-a-judge API scussessfully, no proxy on the API's IP.


## 3. Reference Resources
- https://github.com/volcengine/verl/blob/c803b1f76936f134f919f13ebd668473f4f661ed/verl/utils/dataset/multiturn_sft_dataset.py#L108
- https://github.com/volcengine/verl/blob/867d3024bf7af6aee2cd785cfd573aec561f212d/verl/trainer/ppo/ray_trainer.py#L206
- https://github.com/volcengine/verl/blob/867d3024bf7af6aee2cd785cfd573aec561f212d/verl/trainer/ppo/ray_trainer.py#L153
- https://github.com/bytedance/SandboxFusion
- https://github.com/QwenLM/Qwen-Agent
