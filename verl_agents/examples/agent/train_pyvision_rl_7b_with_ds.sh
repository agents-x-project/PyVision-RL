set -x

export LLM_AS_A_JUDGE_BASE="http://10.140.60.133:18901/v1" # 10-140-1-174
export no_proxy='10.140.60.133:18901'
# export no_proxy='10.140.60.38:8265'
export HYDRA_FULL_ERROR=1
# export WANDB_MODE=offline

PROJECT_NAME="pyvision-rl-v0"
EXPERIMENT_NAME="qwen25vl_7b_sft_1epoch_v1_16gpu_maxturn4_with_ds"

export SAVE_CHECKPOINT_DIR=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/agents_x/rl_ckpts
# export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

BASEDIR=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/DeepEyes-Datasets-47k
VISUAL_DATASET_TRAIN_0_6_2=${BASEDIR}/data_v0.6.2_reason.parquet
VISUAL_DATASET_TRAIN_0_1_2=${BASEDIR}/data_0.1.2_visual_toolbox_v2.parquet
VISUAL_DATASET_TRAIN_0_8=${BASEDIR}/data_v0.8_visual_toolbox_v2.parquet
VISUAL_DATASET_TEST=${BASEDIR}/seekworld_test.parquet
EUREKA_DATASET_TRAIN=${BASEDIR}/data_thinklite_reasoning_acc.parquet

PYVISION_DATASET_DEEPEYES_TRAIN_0=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/DeepEyes-Datasets-47k/parqurt_files_1/data_0.1.2_visual_toolbox_v2/train_1.parquet
PYVISION_DATASET_DEEPEYES_TRAIN_1=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/DeepEyes-Datasets-47k/parqurt_files_1/data_0.1.2_visual_toolbox_v2/train_2.parquet
PYVISION_DATASET_DEEPEYES_TRAIN_2=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/DeepEyes-Datasets-47k/parqurt_files_1/data_0.1.2_visual_toolbox_v2/train_3.parquet
PYVISION_DATASET_DEEPEYES_TRAIN_3=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/DeepEyes-Datasets-47k/parqurt_files_1/data_0.1.2_visual_toolbox_v2/train_4.parquet

PYVISION_DATASET_ZEBRA_COT_TRAIN_0=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/Visual-Search/parquet_files/train_1.parquet
PYVISION_DATASET_ZEBRA_COT_TRAIN_1=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/Visual-Search/parquet_files/train_2.parquet
PYVISION_DATASET_ZEBRA_COT_TRAIN_2=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/Visual-Search/parquet_files/train_3.parquet
PYVISION_DATASET_ZEBRA_COT_TRAIN_3=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/Visual-Search/parquet_files/train_4.parquet

PYVISION_DATASET_VIGORL_TRAIN_0=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/vigorl_datasets/parquet_files/train_1.parquet
PYVISION_DATASET_VIGORL_TRAIN_1=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/vigorl_datasets/parquet_files/train_2.parquet
PYVISION_DATASET_VIGORL_TRAIN_2=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/vigorl_datasets/parquet_files/train_3.parquet
PYVISION_DATASET_VIGORL_TRAIN_3=/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/data/vigorl_datasets/parquet_files/train_4.parquet

enable_filter_groups=True
filter_groups_metric=seq_reward
max_num_gen_batches=0

REF_MODEL_PATH=/mnt/petrelfs/zhaoshitian/gveval_zhaoshitian/agents_x_data/sft_ckpt/qwen2_5vl-7b-2/full/sft
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=True \
    +vs_debug=True \
    data.train_files=[${PYVISION_DATASET_DEEPEYES_TRAIN_0},${PYVISION_DATASET_DEEPEYES_TRAIN_1},${PYVISION_DATASET_DEEPEYES_TRAIN_2},${PYVISION_DATASET_DEEPEYES_TRAIN_3},${PYVISION_DATASET_ZEBRA_COT_TRAIN_0},${PYVISION_DATASET_ZEBRA_COT_TRAIN_1},${PYVISION_DATASET_ZEBRA_COT_TRAIN_2},${PYVISION_DATASET_ZEBRA_COT_TRAIN_3},${PYVISION_DATASET_VIGORL_TRAIN_0},${PYVISION_DATASET_VIGORL_TRAIN_1},${PYVISION_DATASET_VIGORL_TRAIN_2},${PYVISION_DATASET_VIGORL_TRAIN_3}] \
    data.val_files=[${EUREKA_DATASET_TRAIN}] \
    data.train_batch_size=64 \
    data.max_prompt_length=32000 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
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
    actor_rollout_ref.rollout.n=8 \
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
    actor_rollout_ref.rollout.agent.max_turns=5 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    +trainer.rollout_data_dir=/mnt/petrelfs/zhaoshitian/vis_tool_train/rollouts/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb','rl_logging_board'] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=${WORLD_SIZE} \
    trainer.save_freq=10 \
    trainer.test_freq=10000 \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    +trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard \
    +trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board \
    trainer.total_epochs=32 2>&1 | tee ./logs/${EXPERIMENT_NAME}.log
