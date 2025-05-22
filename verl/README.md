<h1 style="text-align: center;">verl: Volcano Engine Reinforcement Learning for LLMs</h1>

verl is a flexible, efficient and production-ready RL training library for large language models (LLMs).

verl is the open-source version of **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** paper.

verl is flexible and easy to use with:

- **Easy extension of diverse RL algorithms**: The hybrid-controller programming model enables flexible representation and efficient execution of complex Post-Training dataflows. Build RL dataflows such as GRPO, PPO in a few lines of code.

- **Seamless integration of existing LLM infra with modular APIs**: Decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks, such as FSDP, Megatron-LM, vLLM, SGLang, etc

- **Flexible device mapping**: Supports various placement of models onto different sets of GPUs for efficient resource utilization and scalability across different cluster sizes.

- Ready integration with popular HuggingFace models

verl is fast with:

- **State-of-the-art throughput**: SOTA LLM training and inference engine integrations and SOTA RL throughput.

- **Efficient actor model resharding with 3D-HybridEngine**: Eliminates memory redundancy and significantly reduces communication overhead during transitions between training and generation phases.

## Key Features

- **FSDP**, **FSDP2** and **Megatron-LM** for training.
- **vLLM**, **SGLang** and **HF Transformers** for rollout generation.
- Compatible with Hugging Face Transformers and Modelscope Hub: [Qwen-3](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-8b.sh), Qwen-2.5, Llama3.1, Gemma2, DeepSeek-LLM, etc
- Supervised fine-tuning.
- Reinforcement learning with [PPO](examples/ppo_trainer/), [GRPO](examples/grpo_trainer/), [ReMax](examples/remax_trainer/), [REINFORCE++](https://verl.readthedocs.io/en/latest/examples/config.html#algorithm), [RLOO](examples/rloo_trainer/), [PRIME](recipe/prime/), [DAPO](recipe/dapo/), [DrGRPO](recipe/drgrpo), etc.
  - Support model-based reward and function-based reward (verifiable reward) for math, [coding](https://github.com/volcengine/verl/tree/main/recipe/dapo), etc
  - Support vision-language models (VLMs) and [multi-modal RL](examples/grpo_trainer/run_qwen2_5_vl-7b.sh)
  - [Multi-turn with tool calling](https://github.com/volcengine/verl/tree/main/examples/sglang_multiturn)
- LLM alignment recipes such as [Self-play preference optimization (SPPO)](https://github.com/volcengine/verl/tree/main/recipe/sppo)
- Flash attention 2, [sequence packing](examples/ppo_trainer/run_qwen2-7b_seq_balance.sh), [sequence parallelism](examples/ppo_trainer/run_deepseek7b_llm_sp2.sh) support via DeepSpeed Ulysses, [LoRA](examples/sft/gsm8k/run_qwen_05_peft.sh), [Liger-kernel](examples/sft/gsm8k/run_qwen_05_sp2_liger.sh).
- Scales up to 70B models and hundreds of GPUs.
- Experiment tracking with wandb, swanlab, mlflow and tensorboard.

## Upcoming Features and Changes

- Roadmap https://github.com/volcengine/verl/issues/710
- DeepSeek 671b optimizations with Megatron v0.11 https://github.com/volcengine/verl/issues/708
- Multi-turn rollout optimizations https://github.com/volcengine/verl/pull/1037 https://github.com/volcengine/verl/pull/1138
- Environment interactions https://github.com/volcengine/verl/issues/1172
- List of breaking changes since v0.3 https://github.com/volcengine/verl/discussions/943

## Getting Started

<a href="https://verl.readthedocs.io/en/latest/index.html"><b>Documentation</b></a>

**Quickstart:**

- [Installation](https://verl.readthedocs.io/en/latest/start/install.html)
- [Quickstart](https://verl.readthedocs.io/en/latest/start/quickstart.html)
- [Programming Guide](https://verl.readthedocs.io/en/latest/hybrid_flow.html)

**Running a PPO example step-by-step:**

- Data and Reward Preparation
  - [Prepare Data for Post-Training](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
  - [Implement Reward Function for Dataset](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- Understanding the PPO Example
  - [PPO Example Architecture](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
  - [Config Explanation](https://verl.readthedocs.io/en/latest/examples/config.html)
  - [Run GSM8K Example](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html)

**Reproducible algorithm baselines:**

- [PPO, GRPO, ReMax](https://verl.readthedocs.io/en/latest/experiment/ppo.html)

**For code explanation and advance usage (extension):**

- PPO Trainer and Workers
  - [PPO Ray Trainer](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP Backend](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM Backend](https://verl.readthedocs.io/en/latest/index.html)
- Advance Usage and Extension
  - [Ray API design tutorial](https://verl.readthedocs.io/en/latest/advance/placement.html)
  - [Extend to Other RL(HF) algorithms](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [Add Models with the FSDP Backend](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [Add Models with the Megatron-LM Backend](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)
  - [Deployment using Separate GPU Resources](https://github.com/volcengine/verl/tree/main/examples/split_placement)

**Blogs from the community**

- [SGLang, verl, OpenBMB and Tsinghua University: Pioneering End-to-End Multi-Turn RLHF](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/verl-multiturn-rollout-Release.md)
- [Reinforcement Learning from Human Feedback on AMD GPUs with verl and ROCm Integration](https://rocm.blogs.amd.com/artificial-intelligence/verl-large-scale/README.html)
- [veMLP x verl ：玩转强化学习训练](https://mp.weixin.qq.com/s/7nbqxk4knMGd-hQE9ls2tA)
- [使用 verl 进行 GRPO 分布式强化学习训练最佳实践](https://www.volcengine.com/docs/6459/1463942)
- [HybridFlow verl 原文浅析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/readme.md)

## Performance Tuning Guide

The performance is essential for on-policy RL algorithm. We have written a detailed [performance tuning guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) to help you optimize performance.

## Upgrade to vLLM >= v0.8.2

verl now supports vLLM>=0.8.2 when using FSDP as the training backend. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) for the installation guide and more information. Please avoid vllm 0.7.x, which contains bugs that may lead to OOMs and unexpected errors.

## Use Latest SGLang

SGLang is fully supported with verl, and SGLang RL Group is working extensively on building unique features, including multi-turn agentic RL, VLM RLHF, server-based RL, and partial rollout. Please refer to [this document](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html) for the installation guide and more information.

## Upgrade to FSDP2

verl is fully embracing FSDP2! FSDP2 is recommended by torch distributed team, providing better throughput and memory usage, and is composible with other features (e.g. torch.compile). To enable FSDP2, simply use verl main and set the following options:
```
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2 
reward_model.strategy=fsdp2 
```
Furthermore, FSDP2 cpu offloading is compatible with gradient accumulation. You can turn it on to save memory with `actor_rollout_ref.actor.offload_policy=True`. For more details, see https://github.com/volcengine/verl/pull/1026

## [Hardware] Support AMD (ROCm Kernel)

verl now supports FSDP as the training engine (Megatron support coming soon) and both integrates with vLLM and SGLang as inference engines. Please refer to [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst) for the installation guide and more information, and [this document](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_vllm_page.rst) for the vLLM performance tuning for ROCm.


## Citation and acknowledgement

If you find the project helpful, please cite:

- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
- [A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization](https://i.cs.hku.hk/~cwu/papers/gmsheng-NL2Code24.pdf)

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```