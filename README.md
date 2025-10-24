# PyVision RL Training

### Prepare the RL data and SFT ckpts

#### With Multi-modal Hint in the Input

SFT ckpts: https://huggingface.co/Agents-X/sft-data-v2-Qwen2.5-VL-7B-1epoch

RL data: https://huggingface.co/datasets/Agents-X/de_visual_search_filtered
(parquet files)

#### Without Multi-modal Hint in the Input

Image Dataset wo Image Hint: https://huggingface.co/datasets/Agents-X/rl_data_de_visual_search_filtered_wo_image_hint

Video Dataset wo Video Hint: https://huggingface.co/datasets/Agents-X/rl_data_vsi_filtered_wo_video_hint

SFT ckpts: https://huggingface.co/Agents-X/qwen2_5vl_7b_full_sft_251013_all_wo_hint-EMA

### Installation For RL

see https://github.com/Visual-Agent/DeepEyes

```bash
####################################################
#        you should just follow deepeyes.          #
####################################################

# Notes:
# 1. transformers==4.54.0
# 2. vllm==0.9.1
```

### Installation For PyVision-Interaction

For the interaction environment, make sure these Python packages have been installed.
```bash
pillow
matplotlib
numpy
timeout-decorator
pebble
regex
markdown
pytesseract
scikit-image
scipy
scikit-learn
easyocr
webcolors<24.6.0
nltk
decord
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

Finally, prepare the llm-as-a-judge config file. the `api_key` is useless, just keep it as `"[EMPTY]"`.

```bash
# ./configs/llm_as_a_judge.json

{
    "api_key": "[EMPTY]",
    "base_url": "",
    "model_name": ""
}
```

### Train
After serving the llm-as-a-judge and making sure it works well, you could start to train. 
Note: I strictly followed `verl`'s doc to start the multi-node RL training. If you want more detail, please check `verl`'s doc.

#### Training Script
```bash
# ./verl_agents/examples/agent/train_pyvision_rl_7b_v4.sh
# This is an example training script, you could create a new one.
# For every training parameter explaination, check ./verl_agents/verl/trainer/config/ppo_trainer.yaml

PROJECT_NAME="pyvision-rl-v0"
EXPERIMENT_NAME="pyvision-rl"
export SAVE_CHECKPOINT_DIR=/the/path/to/save/the/rl/ckpts
export TMPDIR="$HOME/tmp/ray"
export WANDB_MODE=offline  # setup the wandb

ROLLOUT_SAVE_DIR_PATH=./rollouts
FIRST_ROLLOUT_SAVE_DIR_PATH=./first_rollouts
export LLM_AS_A_JUDGE_CONFIG_PATH=./configs/llm_as_a_judge.json


####################################################### Training Data Path Parameter ####################################################################
                                                                                                                                                        #
# If with mm hint in the input, the data path should be the dir path containing the parquet files.                                                      #
PYVISION_DATASET_DIR_DEEPEYES=./rl_data/filtered_deepeyes_visual_search_parquet_files                                                                   #
                                                                                                                                                        #
# If without mm hint in the input, the data path should be json file path.                                                                              #
                                                                                                                                                        #
PYVISION_IMAGE_DATASET_WO_MM_HINT=./rl_data/deepeyes/train_data_wo_mm_hint_full_path.json                                                               #
PYVISION_VIDEO_DATASET_WO_MM_HINT=./rl_data/vsi/train_data_wo_mm_hint_full_path.json                                                                    #                                                                                
                                                                                                                                                        #
#########################################################################################################################################################

#################################### Data Loading Parameter ######################################
gen_batch_size=64   
max_video_gen_batch_size=32     # 32 might cause OOM in longvila
gen_batch_size_align_method="up_resample_image"     # up_resample_image: resample prompts from dataloader to fill discarded prompts with video
##################################################################################################

#################################### Dynamic Sampling Parameter ##################################
enable_filter_groups=True                                                                        
filter_groups_metric='seq_reward,hasimage,trajlength,end_reason'   # end_reason_filter_reserve_names for filtering trajs that is truncated by `verl_agents/verl/workers/agent/parallel_env.py`                                                        
end_reason_filter_reserve_names='DONE,EXCEED_MAX_IMAGE_NUM_32'     # Options: [ON_GONIG, DONE, OVER_LENGTH, EXCEED_MAX_TURNS, EXCEED_MAX_IMAGE_NUM_32]
max_num_gen_batches=0                                                                            
##################################################################################################

#################################### Other RL Parameter ##########################################
rollout_num=8                                                                                    
max_turn=5                                                                                       
tool_using_cumulative_reward_per_turn=0.0
                                                                                                 
with_mm_hint=False                                                                               
WORLD_SIZE=1                                                                                     
##################################################################################################


REF_MODEL_PATH=/the/path/to/your/download/sft/ckpts

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=True \
    +vs_debug=True \
    data.train_files=[${PYVISION_IMAGE_DATASET_WO_MM_HINT},${PYVISION_VIDEO_DATASET_WO_MM_HINT}] \
    data.train_batch_size=64 \
    data.max_prompt_length=32000 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.with_mm_hint=${with_mm_hint} \
    +data.gen_batch_size=${gen_batch_size} \
    +data.max_video_gen_batch_size=${max_video_gen_batch_size} \   
    +data.gen_batch_size_align_method=${gen_batch_size_align_method} \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path=${REF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra'] \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=[${filter_groups_metric}] \
    +algorithm.filter_groups.end_reason_filter_reserve_names=[${end_reason_filter_reserve_names}] \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${rollout_num} \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
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
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    actor_rollout_ref.rollout.agent.tool_using_cumulative_reward_per_turn=${tool_using_cumulative_reward_per_turn} \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_DIR_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.the_first_batch_rollout_data_dir=${FIRST_ROLLOUT_SAVE_DIR_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','rl_logging_board'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=10 \
    trainer.test_freq=0 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=32 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log

```

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
