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
cd visagent
bash own_scripts/7b_single_grpo.sh vllm data/mmk12 ~/checkpoints/Qwen2.5-VL-7B-Instruct
```

## 2. Multi-Turn with Tool Using
To be built for multi-turn with tool using.

### TO DO
- [❌] Adapt from LLM multiturn tool to visual agent multiturn tool.
- [❌] Integrate sandbox in the inference pipeline.
- [❌] Build multi-turn with tool using.


## 3. Reference Resources
- https://github.com/volcengine/verl/blob/c803b1f76936f134f919f13ebd668473f4f661ed/verl/utils/dataset/multiturn_sft_dataset.py#L108
- https://github.com/volcengine/verl/blob/867d3024bf7af6aee2cd785cfd573aec561f212d/verl/trainer/ppo/ray_trainer.py#L206
- https://github.com/volcengine/verl/blob/867d3024bf7af6aee2cd785cfd573aec561f212d/verl/trainer/ppo/ray_trainer.py#L153
- https://github.com/bytedance/SandboxFusion
https://github.com/QwenLM/Qwen-Agent
