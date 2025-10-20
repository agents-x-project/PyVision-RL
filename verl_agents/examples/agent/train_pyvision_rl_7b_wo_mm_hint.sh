# ./verl_agents/examples/agent/train_pyvision_rl_7b_v4.sh
# This is an example training script, you could create a new one.
# For every training parameter explaination, check ./verl_agents/verl/trainer/config/ppo_trainer.yaml

PROJECT_NAME="pyvision-rl-v4-wo-mm-hint"
EXPERIMENT_NAME="sft-wo-mm-hint-v1-rl-test"
export SAVE_CHECKPOINT_DIR=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/agents_x/rl_ckpts

ROLLOUT_SAVE_DIR_PATH=/mnt/petrelfs/zhaoshitian/vis_tool_train/rollouts
FIRST_ROLLOUT_SAVE_DIR_PATH=/mnt/petrelfs/zhaoshitian/vis_tool_train/the_first_batch_rollouts


####################################################### Training Data Path Parameter ####################################################################
                                                                                                                                                        #
# If with mm hint in the input, the data path should be the dir path containing the parquet files.                                                      #
PYVISION_DATASET_DIR_DEEPEYES=/mnt/petrelfs/zhaoshitian/eaigc2_t_zhaoshitian/agents_x/rl_data/filtered_deepeyes_visual_search_parquet_files             #
                                                                                                                                                        #
# If without mm hint in the input, the data path should be json file path.                                                                              #
                                                                                                                                                        #
PYVISION_DATASET_WO_MM_HINT=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/DeepEyes-Datasets-47k/train_data_wo_mm_hint_full_path.json                                                                                                                                                #
                                                                                                                                                        #
#########################################################################################################################################################

#################################### Dynamic Sampling Parameter ##################################
enable_filter_groups=True                                                                        #
filter_groups_metric=seq_reward                                                                  #
max_num_gen_batches=0                                                                            #
##################################################################################################

#################################### Other RL Parameter ##########################################
rollout_num=8                                                                                    #
interaction_budget=4                                                                             #
max_turn=5                                                                                       #
overbudget_masking=False                                                                          #
# NOTE: interaction_budget = max_turn - 1                                                        #
                                                                                                 #
with_mm_hint=False                                                                                #
WORLD_SIZE=1
##################################################################################################


# REF_MODEL_PATH=/mnt/petrelfs/zhaoshitian/eaigc3_t_zhaoshitian/models/qwen2_5vl_7b_full_sft_251013_all_wo_hint-EMA
REF_MODEL_PATH=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/agents_x/sft_ckpts/qwen2_5vl-7b-1epoch_v2/full/sft
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=True \
    +vs_debug=True \
    data.train_files=[${PYVISION_DATASET_WO_MM_HINT}] \
    data.train_batch_size=64 \
    data.max_prompt_length=32000 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.with_mm_hint=${with_mm_hint} \
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
    algorithm.filter_groups.metric=${filter_groups_metric} \
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
    actor_rollout_ref.actor.interaction_budget=${interaction_budget} \
    actor_rollout_ref.actor.overbudget_masking=${overbudget_masking} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=10240 \
    actor_rollout_ref.rollout.agent.max_turns=${max_turn} \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    +trainer.rollout_data_dir=${ROLLOUT_SAVE_DIR_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.the_first_batch_rollout_data_dir=${FIRST_ROLLOUT_SAVE_DIR_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
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
