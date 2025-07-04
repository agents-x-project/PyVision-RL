# Training o3-like Visual Agent with RL

## 1. Single Turn RL

### Installation
```bash
cd visagent
conda create -n visagent python=3.10
pip install torch==2.6.0
pip install vllm==0.8.3
pip install -e .
pip install flash-attn --no-build-isolation # or build from source
pip install -r requirements.txt
```

### Data Preparation
```bash
huggingface-cli login # fill token
cd visagent && PYTHONPATH=. python examples/data_preprocess/mmk12.py --local_dir data/mmk12 --n_test 707
```

### Training
```bash
cd visagent && wandb login
bash own_scripts/7b_single_grpo.sh vllm data/mmk12 /home/checkpoints/Qwen2.5-VL-7B-Instruct
```

## 2. Multi-Turn with Tool Using
To be built for multi-turn with tool using.

### Installation
```bash
conda create -n pyvision-agent python=3.10
conda activate pyvision-agent

# if your cuda version is 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install vllm

pip install -r env_files/requirements_agent.txt
# Follow the VeRL official installation procedure
pip install -e .
```

### Serve Qwen2.5-72B-Instruct for LLM-as-a-Judge
```bash
cd verl_agents
bash serve_llm.sh
```
Test if the served llm works:
```bash
python test_call_qwen_serve.py
```

### Train
```bash
bash run_train.sh
```


## 3. Reference Resources
- https://github.com/volcengine/verl/blob/c803b1f76936f134f919f13ebd668473f4f661ed/verl/utils/dataset/multiturn_sft_dataset.py#L108
- https://github.com/volcengine/verl/blob/867d3024bf7af6aee2cd785cfd573aec561f212d/verl/trainer/ppo/ray_trainer.py#L206
- https://github.com/volcengine/verl/blob/867d3024bf7af6aee2cd785cfd573aec561f212d/verl/trainer/ppo/ray_trainer.py#L153
- https://github.com/bytedance/SandboxFusion
- https://github.com/QwenLM/Qwen-Agent
