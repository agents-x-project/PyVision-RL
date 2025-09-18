# PyVision RL Training

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
After serving the LLM-as-a-judge and making sure it works well, you could start to train.

#### Single Node
```bash
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
After setup the ray head cluster and worker cluser, you need to do to the ray dash board to check the resources.

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


## 3. Reference Resources
- https://github.com/volcengine/verl/blob/c803b1f76936f134f919f13ebd668473f4f661ed/verl/utils/dataset/multiturn_sft_dataset.py#L108
- https://github.com/volcengine/verl/blob/867d3024bf7af6aee2cd785cfd573aec561f212d/verl/trainer/ppo/ray_trainer.py#L206
- https://github.com/volcengine/verl/blob/867d3024bf7af6aee2cd785cfd573aec561f212d/verl/trainer/ppo/ray_trainer.py#L153
- https://github.com/bytedance/SandboxFusion
- https://github.com/QwenLM/Qwen-Agent
