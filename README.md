<div align="center">

##  PyVision-RL: Forging Open Agentic Vision Models via RL.

<a href="https://arxiv.org/abs/2602.20739" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-PyVision-RL-red?logo=arxiv" height="20" />
</a>
<a href="https://agent-x.space/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Homepage-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/papers/2602.20739" target="_blank">
    <img alt="HF Model: ViGaL" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Model&Data-PyVision-RL-ffc107?color=ffc107&logoColor=white" height="20" />
</a>


</div>

## 🎯Overview
Reinforcement learning for agentic multimodal models often suffers from interaction collapse, where models learn to reduce tool usage and multi-turn reasoning, limiting the benefits of agentic behavior. We introduce `PyVision-RL`, a reinforcement learning framework for open-weight multimodal models that stabilizes training and sustains interaction. Our approach combines an oversampling–filtering–ranking rollout strategy with an accumulative tool reward to prevent collapse and encourage multi-turn tool use. Using a unified training pipeline, we develop `PyVision-Image` and `PyVision-Video` for image and video understanding. For video reasoning, PyVision-Video employs on-demand context construction, selectively sampling task-relevant frames during reasoning to significantly reduce visual token usage. Experiments show strong performance and improved efficiency, demonstrating that sustained interaction and on-demand visual processing are critical for scalable multimodal agents.

## 🚩News
- [2026-7-8] 🚀🚀🚀 We are excited to release `PyVision-RL`, inluding:
  - [Techniqual report](https://arxiv.org/abs/2602.20739), code, [models&data of PyVision-Image](https://huggingface.co/collections/Agents-X/pyvision-image) and [models&data of PyVision-Video](https://huggingface.co/collections/Agents-X/pyvision-video).

## 📋Contents
- [Prepare the RL data and SFT ckpts](#prepare_the_rl_data_and_sft_ckpts)
- [Installation](#installation)
- [Train](#train)
- [Evaluation](#evaluation)
- [Citation](#citation)

## 📦Prepare the RL data and SFT ckpts

#### PyVision-Image

SFT ckpt: https://huggingface.co/Agents-X/PyVision-Image-7B-SFT

RL data: https://huggingface.co/datasets/Agents-X/PyVision-Image-RL-Data

#### PyVision-Video

SFT ckpt: https://huggingface.co/Agents-X/PyVision-Video-7B-SFT

RL data: https://huggingface.co/datasets/Agents-X/PyVision-Video-RL-Data

## 🔧Installation

see https://github.com/Visual-Agent/DeepEyes

```bash
####################################################
#        you should just follow deepeyes.          #
####################################################

# Notes:
# 1. transformers==4.54.0
# 2. vllm==0.9.1
```

#### Installation For PyVision-Interaction

For the interaction environment, make sure these Python packages have been installed.
```bash
pip install -r pv_requirements.txt
```

#### Serve Qwen2.5-72B-Instruct for LLM-as-a-Judge
```bash
cd verl_agents
sbatch ./slurm_scripts/sbatch_serve_llm.sh
```
Test if the served llm works:
```bash
python test_call_qwen_serve.py
```

Finally, prepare the llm-as-a-judge config file. the `api_key` is useless, just keep it as `"[EMPTY]"`.

```bash
# ./configs/llm_as_a_judge.json

{
    "api_key": "[EMPTY]",
    "base_url": "",
    "model_name": ""
}
```

## 💥Train
After serving the llm-as-a-judge and making sure it works well, you could start to train. 
<!-- Note: I strictly followed `verl`'s doc to start the multi-node RL training. If you want more detail, please check `verl`'s doc. -->

#### Validation Dataset Format (Optional)

If you want to incorporate validation in the RL training, first prepare the validation data.

For image validation dataset:
```bash
# *_image_val_dataset.json
[
    {
        "id": "0",
        "question": "What is the material of the glove?\n(A) rubber\n(B) cotton\n(C) kevlar\n(D) leather\nAnswer with the option's letter from the given choices directly.",
        "answer": "A",
        "ability": "visual_search",
        "data_source": "vstar",
        "image_path": "/path/to/vstar_bench/direct_attributes/sa_4690.jpg"
    },
    ...
]
```

For video validation dataset:
```bash
# *_video_val_dataset.json
[
    {
        "id": "0",
        "question": "What is the material of the glove?\n(A) rubber\n(B) cotton\n(C) kevlar\n(D) leather\nAnswer with the option's letter from the given choices directly.",
        "answer": "A",
        "ability": "spatial_reasoning",
        "data_source": "vsi", # or "vsi_numerical"
        "video_path": "xxxx/xxxx.mp4"
    },
    ...
]
```

#### Training Script
```bash
# This is an example training script, you could create a new one.
# For every training parameter explaination, check ./verl_agents/verl/trainer/config/ppo_trainer.yaml

PROJECT_NAME="pyvision-rl"
EXPERIMENT_NAME="pyvision-image"

current_time=$(date '+%m%d-%H%M%S')
EXPERIMENT_NAME="${EXPERIMENT_NAME}-${current_time}"
export OUTPUT_BASE_DIR="/path/to/the/training/output/${EXPERIMENT_NAME}"


##################################################################################################
#                                           WandB Setup                                          #
##################################################################################################
export WANDB_MODE=offline  # setup the wandb, you could also set it to 'online'.
export WANDB_RUN_ID=$EXPERIMENT_NAME
export WANDB_RESUME="allow"
export WANDB_DIR=$OUTPUT_BASE_DIR

export SAVE_CHECKPOINT_DIR="${OUTPUT_BASE_DIR}/ckpt"
ROLLOUT_SAVE_DIR_PATH="${OUTPUT_BASE_DIR}/rollouts"
FIRST_ROLLOUT_SAVE_DIR_PATH="${OUTPUT_BASE_DIR}/first_rollouts"

mkdir -p "$OUTPUT_BASE_DIR"
cp "$0" "$OUTPUT_BASE_DIR/"

export TMPDIR="$HOME/tmp/ray"
export HYDRA_FULL_ERROR=1
export LLM_AS_A_JUDGE_CONFIG_PATH="/path/to/configs/llm_as_a_judge.json"


##################################################################################################
#                                 Training Data Path Parameter                                   #
##################################################################################################                                                               
                                                                                                                                                        
# For PyVision-Image, the data json file path should be:
PYVISION_IMAGE_RL_DATA=/path/to/PyVision-Image-RL-Data/pyvision_image_rl_data.json
                                                                                                                
# For PyVision-Video, the data json file path should be:
PYVISION_VIDEO_RL_DATA=/path/to/PyVision-Video-RL-Data/pyvision_video_rl_data.json

VSTAR_BENCH=/path/to/vstar/vstar_pv_form_image_val_dataset.json 

##################################################################################################
#                                    Data Loading Parameter                                      #
##################################################################################################
gen_batch_size=32   
max_video_gen_batch_size=16     # 32 might cause OOM for long videos
gen_batch_size_align_method="up_resample_image"     # up_resample_image: resample prompts from dataloader to fill discarded prompts with video


##################################################################################################
#                                     Filter-Ranker Parameter                                    #
##################################################################################################
enable_filter_groups=True  
std_sort_enable=True                                                                      
filter_groups_metric='seq_reward,hasimage,trajlength,vtoken_images_num_consis,end_reason'   # end_reason_filter_reserve_names for filtering trajs that is truncated by `verl_agents/verl/workers/agent/parallel_env.py`
end_reason_filter_reserve_names='DONE,EXCEED_MAX_TURNS,ERROR_IN_ACTION'     # Options: [ON_GONIG, DONE, OVER_LENGTH, EXCEED_MAX_TURNS, EXCEED_MAX_IMAGE_NUM_32, ERROR_IN_ACTION]
max_num_gen_batches=0                                                                            


##################################################################################################
#                                       Other RL Parameter                                       #
##################################################################################################
rollout_num=8                                                                                    
max_turn=5 
max_turn_in_val=30                                                                                      
tool_using_cumulative_reward_per_turn=0.1
concurrent_workers=64  # the worker num used for vlm-env interaction
prompt_template_path=./verl_agents/verl/utils/dataset/rl_system_prompt_template.json
min_pixels=3136
max_pixels=2000000

norm_adv_by_std_in_grpo=False
                                                                                                 
with_mm_hint=True # 'True' for PyVision-Image and 'False' for PyVision-Video                                                                             
WORLD_SIZE=1  

REF_MODEL_PATH=/the/path/to/your/download/sft/ckpt
TRAIN_DATA_JSON_TOTAL="${PYVISION_IMAGE_RL_DATA}"
TEST_DATA_JSON_TOTAL="${VSTAR_BENCH}"

echo "============================"
echo "Launched Training: ${PROJECT_NAME}"
echo "From CKPT: ${REF_MODEL_PATH}"
echo "Datasets: ${TRAIN_DATA_JSON_TOTAL}"
echo "============================"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=True \
    +vs_debug=True \
    data.train_files=[${TRAIN_DATA_JSON_TOTAL}] \
    data.val_files=[${TEST_DATA_JSON_TOTAL}] \
    data.train_batch_size=64 \
    data.max_prompt_length=32000 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.with_mm_hint=${with_mm_hint} \
    +data.min_pixels=${min_pixels} \
    +data.max_pixels=${max_pixels} \
    +data.prompt_template_path=${prompt_template_path} \
    +data.gen_batch_size=${gen_batch_size} \
    +data.max_video_gen_batch_size=${max_video_gen_batch_size} \   
    +data.gen_batch_size_align_method=${gen_batch_size_align_method} \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    algorithm.norm_adv_by_std_in_grpo=${norm_adv_by_std_in_grpo} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    +algorithm.filter_groups.std_sort_enable=${std_sort_enable} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=[${filter_groups_metric}] \
    +algorithm.filter_groups.end_reason_filter_reserve_names=[${end_reason_filter_reserve_names}] \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=10240 \
    actor_rollout_ref.rollout.agent.max_turns=${max_turn} \
    actor_rollout_ref.rollout.agent.concurrent_workers=${concurrent_workers} \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    actor_rollout_ref.rollout.agent.tool_using_cumulative_reward_per_turn=${tool_using_cumulative_reward_per_turn} \
    +actor_rollout_ref.rollout.val_kwargs.max_turn_in_val=${max_turn_in_val} \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_DIR_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.the_first_batch_rollout_data_dir=${FIRST_ROLLOUT_SAVE_DIR_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','rl_logging_board'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=10 \
    trainer.test_freq=20 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=5 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log

```

#### Single Node
```bash
bash run_train.sh
```

<!-- #### Multiple Nodes
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
``` -->

## 📊Evaluation

#### Model Merge
```bash
bash scripts/run_merge.sh
```

#### Test
see https://github.com/agents-x-project/PyVision-RL-Eval

## 📜Citation
```bibtex
@article{zhao2026pyvisionrl,
  title={PyVision-RL: Forging Open Agentic Vision Models via RL.},
  author={Zhao, Shitian and Lin, Shaoheng and Li, Ming and Zhang, Haoquan and Peng, Wenshuo and Zhang, Kaipeng and Wei, Chen},
  journal={arxiv preprint arxiv:2602.20739},
  year={2026},
}
```
