# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import json
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.metric_utils_oversample_pool import (
    compute_data_metrics_oversample_pool,
    compute_agent_metrics_oversample_pool,
    compute_end_reason_metrics_oversample_pool
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.dataset.rl_dataset_wo_mm_hint import RLHF_wo_mm_hint_Dataset
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.ppo.filter_fn_utils import rollout_filtering_function

from verl.trainer.ppo.metric_utils import compute_agent_metrics

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        print(f"############# node_available_resources: {node_available_resources}")
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    action_or_attn_mask = data.batch['action_mask'] if 'action_mask' in data.batch.keys() else data.batch['attention_mask']
    response_mask = action_or_attn_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    action_or_attn_mask = data.batch['action_mask'] if 'action_mask' in data.batch.keys() else data.batch['attention_mask']
    return action_or_attn_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, norm_adv_by_std_in_grpo=True):
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns, samples_std_list = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.non_tensor_batch["sample_level_stds"] = samples_std_list
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}'"
                        + "is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.import_utils import load_extern_type

        with_mm_hint = self.config.data.with_mm_hint

        if "custom_cls" in self.config.data and self.config.data.custom_cls.get("path", None) is not None:
            dataset_cls = load_extern_type(self.config.data.custom_cls.path, self.config.data.custom_cls.name)
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(
                    f"The custom dataset class '{self.config.data.custom_cls.name}' from "
                    f"'{self.config.data.custom_cls.path}' must inherit from torch.utils.data.Dataset"
                )
        else:
            if with_mm_hint:
                dataset_cls = RLHFDataset
            else:
                dataset_cls = RLHF_wo_mm_hint_Dataset

        self.train_dataset = dataset_cls(
            data_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = dataset_cls(
            data_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            config=self.config.data,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        # assert len(self.val_dataloader) == 1, (
        #     "Validation dataloader must have a single batch,"
        #     + " which inference engines will schedule the memory themselves."
        # )

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.json")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(entry)

        with open(filename, "w") as f:
            json.dump(lines, f, ensure_ascii=False, indent=4)

        print(f"Dumped generations to {filename}")

    def _dump_batch_generations(self, batch: DataProto, dump_path: str):
        """
        Helper method to dump a batch's generations to disk.
        
        Args:
            batch: DataProto containing prompts, responses, and reward info
            dump_path: Directory path to dump the generations
        """
        # Extract reward extra info from batch
        reward_extra_infos_dict: dict[str, list] = {}
        for key in batch.non_tensor_batch.keys():
            # Skip standard keys, only collect reward-related extra info
            if key not in ["uid", "data_source", "raw_prompt_ids", "raw_prompt", 
                          "multi_modal_data", "origin_multi_modal_data", "multi_modal_inputs",
                          "reward_model"]:
                values = batch.non_tensor_batch[key]
                # Convert numpy arrays to lists
                reward_extra_infos_dict[key] = [
                    v.tolist() if hasattr(v, 'tolist') else v for v in values
                ]
        
        # Decode inputs and outputs
        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
        
        # Use existing dump method
        self._dump_generations(
            inputs=inputs,
            outputs=outputs,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=dump_path,
        )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        timing_raw = {}

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            # if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
            #     return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # test_batch.meta_info = {
            #     "eos_token_id": self.tokenizer.eos_token_id,
            #     "pad_token_id": self.tokenizer.pad_token_id,
            #     "recompute_log_prob": False,
            #     # "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            #     "do_sample": True,
            #     "validate": True,
            # }
            # print(f"test_gen_batch meta info: {test_batch.meta_info}")

            # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_batch = self._generate_and_score_batch_validation(test_batch, timing_raw)

            # if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
            #     test_gen_batch = test_batch.pop(
            #         batch_keys=['input_ids', 'attention_mask', 'position_ids'],
            #         non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'origin_multi_modal_data', 'multi_modal_inputs'],
            #     )
            # else:
            #     test_gen_batch = test_batch.pop(
            #         batch_keys=["input_ids", "attention_mask", "position_ids"],
            #         non_tensor_batch_keys=["raw_prompt_ids"],
            #     )

            # if 'raw_prompt' in test_batch.non_tensor_batch.keys():
            #     test_gen_batch.non_tensor_batch['raw_prompt'] = test_batch.non_tensor_batch.pop('raw_prompt')

            # if self.config.actor_rollout_ref.rollout.agent.activate_agent:
            #     tool_name_key = self.config.actor_rollout_ref.rollout.agent.tool_name_key
            #     if tool_name_key and tool_name_key in test_batch.non_tensor_batch.keys():
            #         test_gen_batch.non_tensor_batch[tool_name_key] = test_batch.non_tensor_batch.pop(tool_name_key)

            # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            # print("validation generation end")

            # # Store generated outputs
            # output_ids = test_output_gen_batch.batch["responses"]
            # output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            # sample_outputs.extend(output_texts)

            # test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # result = self.val_reward_fn(test_batch, return_dict=True)
            # reward_tensor = result["reward_tensor"]
            # scores = reward_tensor.sum(-1).cpu().tolist()
            is_correct = test_batch.non_tensor_batch['is_answer_right']
            # print(f"############ is_correct: {is_correct}")
            scores = []
            for is_correct_item in is_correct:
                if is_correct_item:
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            sample_scores.extend(scores)
            print(f"[Val Dataset]: acc: {sum(sample_scores)/len(sample_scores)}")
            print(f"[Val Dataset]: scores: {sample_scores}")

            reward_extra_infos_dict["reward"].extend(scores)
            # if "reward_extra_info" in result:
            #     for key, lst in result["reward_extra_info"].items():
            #         reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(sample_scores)))

        # self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        print(f"##################### validation metric dict: {metric_dict}")
        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, **wg_kwargs
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _filter_video_samples_in_dict(self, batch_dict: dict, max_video_samples: int):
        """
        Filter video samples in a batch dictionary before generation to control GPU memory.
        
        Args:
            batch_dict: Raw batch dictionary from dataloader
            max_video_samples: Maximum number of video samples allowed
            
        Returns:
            tuple: (filtered_batch_dict, video_count, filtered_count)
                - filtered_batch_dict: Filtered batch dict (may be smaller)
                - video_count: Total number of video samples in original batch
                - filtered_count: Number of video samples removed
        """
        if "origin_multi_modal_data" not in batch_dict:
            # No multi-modal data, no filtering needed
            return batch_dict, 0, 0
        
        origin_mm_data_list = batch_dict["origin_multi_modal_data"]
        
        # Identify which samples contain video
        video_mask = []
        for origin_mm_data in origin_mm_data_list:
            has_video = False
            if isinstance(origin_mm_data, dict) and "video" in origin_mm_data:
                has_video = True
            video_mask.append(has_video)
        
        video_count = sum(video_mask)
        
        if video_count <= max_video_samples:
            # Within limit, no filtering needed
            return batch_dict, video_count, 0
        
        # Exceeded limit, need to filter
        num_to_filter = video_count - max_video_samples
        print(f"⚠️  WARNING: Batch contains {video_count} video samples, exceeds limit of {max_video_samples}")
        print(f"   Discarding {num_to_filter} video samples before generation to control GPU memory")
        
        # Keep all non-video samples and only max_video_samples video samples
        keep_indices = []
        video_kept = 0
        
        for idx, has_video in enumerate(video_mask):
            if not has_video:
                # Always keep non-video samples
                keep_indices.append(idx)
            elif video_kept < max_video_samples:
                # Keep video sample if under limit
                keep_indices.append(idx)
                video_kept += 1
            # else: discard this video sample
        
        # Create filtered batch_dict
        filtered_batch_dict = {}
        for key, value in batch_dict.items():
            if isinstance(value, list):
                filtered_batch_dict[key] = [value[i] for i in keep_indices]
            elif isinstance(value, torch.Tensor):
                filtered_batch_dict[key] = value[keep_indices]
            elif isinstance(value, np.ndarray):
                filtered_batch_dict[key] = value[keep_indices]
            else:
                # For non-sequence values, keep as is
                filtered_batch_dict[key] = value
        
        original_size = len(origin_mm_data_list)
        filtered_size = len(keep_indices)
        print(f"   Pre-generation filtering: {original_size} -> {filtered_size} samples "
              f"({video_kept} video, {filtered_size - video_kept} non-video)")
        
        return filtered_batch_dict, video_count, num_to_filter

    def _align_batch_size_for_generation(self, batch_dict: dict, dataloader_iter, method: str, target_gen_batch_size: int):
        """
        Align batch size for generation to meet requirements (multiple of 8).
        
        Args:
            batch_dict: Raw batch dictionary (after video filtering)
            dataloader_iter: Iterator to get additional batches if needed
            method: Alignment method
                - "up_resample_image": Resample image data to reach target_gen_batch_size
                - "discard_image_to_chunk_8": Discard image data to align to multiple of 8
            target_gen_batch_size: Target generation batch size
            
        Returns:
            tuple: (aligned_batch_dict, alignment_info)
                - aligned_batch_dict: Aligned batch dict
                - alignment_info: dict with alignment statistics
        """
        # Get current batch size
        if "origin_multi_modal_data" in batch_dict:
            current_size = len(batch_dict["origin_multi_modal_data"])
        elif "input_ids" in batch_dict:
            current_size = len(batch_dict["input_ids"])
        else:
            print(f"⚠️  WARNING: Cannot determine batch size for alignment")
            return batch_dict, {"method": method, "original_size": 0, "aligned_size": 0, "changed": False}
        
        if method == "up_resample_image":
            # Always resample to reach target_gen_batch_size, even if already multiple of 8
            if current_size >= target_gen_batch_size:
                # Already at or above target, no need to resample
                return batch_dict, {
                    "method": method,
                    "original_size": current_size,
                    "aligned_size": current_size,
                    "changed": False
                }
            
            needed = target_gen_batch_size - current_size
            
            # Get a new batch from dataloader to extract image data
            try:
                # if self.global_steps % (dataloader_len) == 0 and self.global_steps > 0:
                #     dataloader_iter = iter(self.train_dataloader)
                new_batch_dict = next(dataloader_iter)
            except:
                dataloader_iter = iter(self.train_dataloader)
                batch_dict = next(dataloader_iter)
                # print(f"⚠️  WARNING: Cannot get new batch for resampling (dataloader exhausted)")
                # return batch_dict, {
                #     "method": method,
                #     "original_size": current_size,
                #     "aligned_size": current_size,
                #     "changed": False
                # }
            
            # Identify image samples in the new batch
            image_indices = []
            if "origin_multi_modal_data" in new_batch_dict:
                origin_mm_data_list = new_batch_dict["origin_multi_modal_data"]
                for idx, origin_mm_data in enumerate(origin_mm_data_list):
                    has_image = False
                    if isinstance(origin_mm_data, dict) and "image" in origin_mm_data:
                        has_image = True
                    if has_image:
                        image_indices.append(idx)
            
            if len(image_indices) == 0:
                print(f"⚠️  WARNING: No image data in new batch for resampling")
                return batch_dict, {
                    "method": method,
                    "original_size": current_size,
                    "aligned_size": current_size,
                    "changed": False
                }
            
            # Sample with replacement if needed
            import random
            if len(image_indices) < needed:
                # Need to sample with replacement
                sampled_indices = random.choices(image_indices, k=needed)
            else:
                # Can sample without replacement
                sampled_indices = random.sample(image_indices, k=needed)
            
            print(f"📊 Batch alignment ({method}): Adding {needed} image samples from new batch "
                  f"to reach target size {current_size} -> {target_gen_batch_size}")
            
            # Create aligned batch_dict by appending sampled image data
            aligned_batch_dict = {}
            for key, value in batch_dict.items():
                if isinstance(value, list):
                    aligned_batch_dict[key] = value + [new_batch_dict[key][i] for i in sampled_indices]
                elif isinstance(value, torch.Tensor):
                    sampled_data = new_batch_dict[key][sampled_indices]
                    aligned_batch_dict[key] = torch.cat([value, sampled_data], dim=0)
                elif isinstance(value, np.ndarray):
                    sampled_data = new_batch_dict[key][sampled_indices]
                    aligned_batch_dict[key] = np.concatenate([value, sampled_data], axis=0)
                else:
                    # Non-sequence values, keep as is
                    aligned_batch_dict[key] = value
            
            return aligned_batch_dict, {
                "method": method,
                "original_size": current_size,
                "aligned_size": target_gen_batch_size,
                "changed": True,
                "images_added": needed
            }
        
        elif method == "discard_image_to_chunk_8":
            # Discard image data to align to multiple of 8
            if current_size % 8 == 0:
                # Already aligned
                return batch_dict, {
                    "method": method,
                    "original_size": current_size,
                    "aligned_size": current_size,
                    "changed": False
                }
            
            to_discard = current_size % 8
            
            # Identify image samples
            image_indices = []
            if "origin_multi_modal_data" in batch_dict:
                origin_mm_data_list = batch_dict["origin_multi_modal_data"]
                for idx, origin_mm_data in enumerate(origin_mm_data_list):
                    has_image = False
                    if isinstance(origin_mm_data, dict) and "image" in origin_mm_data:
                        has_image = True
                    if has_image:
                        image_indices.append(idx)
            
            num_images = len(image_indices)
            
            if num_images < to_discard:
                print(f"⚠️  WARNING: Need to discard {to_discard} samples but only {num_images} image samples available")
                to_discard = num_images
            
            if to_discard == 0 or num_images == 0:
                print(f"⚠️  WARNING: Cannot align batch size (no image data to discard)")
                return batch_dict, {
                    "method": method,
                    "original_size": current_size,
                    "aligned_size": current_size,
                    "changed": False
                }
            
            # Discard the last to_discard image samples
            indices_to_discard = set(image_indices[-to_discard:])
            keep_indices = [i for i in range(current_size) if i not in indices_to_discard]
            
            aligned_size = len(keep_indices)
            print(f"📊 Batch alignment ({method}): Discarding {to_discard} image samples "
                  f"to align to multiple of 8: {current_size} -> {aligned_size}")
            
            # Create aligned batch_dict with kept samples
            aligned_batch_dict = {}
            for key, value in batch_dict.items():
                if isinstance(value, list):
                    aligned_batch_dict[key] = [value[i] for i in keep_indices]
                elif isinstance(value, torch.Tensor):
                    aligned_batch_dict[key] = value[keep_indices]
                elif isinstance(value, np.ndarray):
                    aligned_batch_dict[key] = value[keep_indices]
                else:
                    aligned_batch_dict[key] = value
            
            return aligned_batch_dict, {
                "method": method,
                "original_size": current_size,
                "aligned_size": aligned_size,
                "changed": True,
                "images_discarded": to_discard
            }
        
        else:
            raise ValueError(f"Unknown alignment method '{method}'. "
                           f"Supported methods: 'up_resample_image', 'discard_image_to_chunk_8'")

    def _generate_and_score_batch(self, batch_dict, timing_raw):
        """
        Generate rollouts and compute rewards for a single batch.
        
        Args:
            batch_dict: Raw batch dictionary from dataloader
            timing_raw: Dictionary to accumulate timing information
            
        Returns:
            DataProto with generated responses and computed rewards
        """
        if isinstance(batch_dict, DataProto):
            new_batch: DataProto = batch_dict
        else:
            new_batch: DataProto = DataProto.from_single_dict(batch_dict)
        
        # Pop keys needed for generation
        if "multi_modal_inputs" in new_batch.non_tensor_batch.keys():
            gen_batch = new_batch.pop(
                batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'origin_multi_modal_data', 'multi_modal_inputs'],
            )
        else:
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", 'origin_multi_modal_data'],
            )
        
        if 'raw_prompt' in new_batch.non_tensor_batch.keys():
            gen_batch.non_tensor_batch['raw_prompt'] = new_batch.non_tensor_batch.pop('raw_prompt')
        
        # Handle agent-specific keys
        if self.config.actor_rollout_ref.rollout.agent.activate_agent:
            tool_name_key = self.config.actor_rollout_ref.rollout.agent.tool_name_key
            if tool_name_key and tool_name_key in new_batch.non_tensor_batch.keys():
                gen_batch.non_tensor_batch[tool_name_key] = new_batch.non_tensor_batch.pop(tool_name_key)
        
        # Generate sequences
        with _timer("gen", timing_raw):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        
        # Handle REMAX baseline generation if needed
        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            with _timer("gen_max", timing_raw):
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["do_sample"] = False
                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                
                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor = self.reward_fn(new_batch)
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                
                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                
                del gen_baseline_batch, gen_baseline_output
        
        # Assign unique IDs and repeat for multiple rollouts per prompt
        new_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
        )
        new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
        new_batch = new_batch.union(gen_batch_output)
        
        # Compute values if using critic
        if self.use_critic:
            with _timer("values", timing_raw):
                values = self.critic_wg.compute_values(new_batch)
                new_batch = new_batch.union(values)
        
        # Compute rewards
        with _timer("reward", timing_raw):
            # Compute reward model scores if enabled
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                new_batch = new_batch.union(reward_tensor)
            
            # Compute final rewards (combining model-based and rule-based)
            reward_result = self.reward_fn(new_batch, return_dict=True)
            reward_tensor = reward_result["reward_tensor"]
            reward_extra_infos_dict = reward_result["reward_extra_info"]
            
            new_batch.batch["token_level_scores"] = reward_tensor
            
            if reward_extra_infos_dict:
                new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
            
            # Apply KL penalty if configured
            # Note: KL metrics will be accumulated at the batch level, not here
            if self.config.algorithm.use_kl_in_reward:
                new_batch, _ = apply_kl_penalty(
                    new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
            else:
                new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
        
        return new_batch

    def _generate_and_score_batch_validation(self, batch_dict, timing_raw):
        """
        Generate rollouts and compute rewards for a single batch.
        
        Args:
            batch_dict: Raw batch dictionary from dataloader
            timing_raw: Dictionary to accumulate timing information
            
        Returns:
            DataProto with generated responses and computed rewards
        """
        if isinstance(batch_dict, DataProto):
            new_batch: DataProto = batch_dict
        else:
            new_batch: DataProto = DataProto.from_single_dict(batch_dict)

        print(f"######### world size: {self.actor_rollout_wg.world_size}")
        print(f"################## length of new_batch: {len(new_batch)}")

        gen_batch_padded, pad_size = pad_dataproto_to_divisor(new_batch, self.actor_rollout_wg.world_size)
        print(f"################## length of gen_batch_padded: {len(gen_batch_padded)}")
        
        # Pop keys needed for generation
        if "multi_modal_inputs" in gen_batch_padded.non_tensor_batch.keys():
            gen_batch = gen_batch_padded.pop(
                batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'origin_multi_modal_data', 'multi_modal_inputs'],
            )
        else:
            gen_batch = gen_batch_padded.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", 'origin_multi_modal_data'],
            )
        
        if 'raw_prompt' in gen_batch_padded.non_tensor_batch.keys():
            gen_batch.non_tensor_batch['raw_prompt'] = gen_batch_padded.non_tensor_batch.pop('raw_prompt')
        
        # Handle agent-specific keys
        if self.config.actor_rollout_ref.rollout.agent.activate_agent:
            tool_name_key = self.config.actor_rollout_ref.rollout.agent.tool_name_key
            if tool_name_key and tool_name_key in gen_batch_padded.non_tensor_batch.keys():
                gen_batch.non_tensor_batch[tool_name_key] = gen_batch_padded.non_tensor_batch.pop(tool_name_key)

        gen_batch.meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            # "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "do_sample": True,
            "validate": True,
            "max_turn_of_validation": 30
        }
        print(f"test_gen_batch meta info: {gen_batch.meta_info}")
        
        # Generate sequences
        with _timer("gen", timing_raw):
            gen_batch_output_padded = self.actor_rollout_wg.generate_sequences(gen_batch)

        print(f"################## length of gen_batch_output_padded: {len(gen_batch_output_padded)}")
        
        gen_batch_output = unpad_dataproto(gen_batch_output_padded, pad_size=pad_size)
        new_batch = unpad_dataproto(gen_batch_padded, pad_size=pad_size)
        
        # Assign unique IDs and repeat for multiple rollouts per prompt
        new_batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
        )
        new_batch = new_batch.union(gen_batch_output)
        
        # Compute rewards
        with _timer("reward", timing_raw):
            # Compute reward model scores if enabled
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                new_batch = new_batch.union(reward_tensor)
            
            # Compute final rewards (combining model-based and rule-based)
            reward_result = self.reward_fn(new_batch, return_dict=True)
            reward_tensor = reward_result["reward_tensor"]
            reward_extra_infos_dict = reward_result["reward_extra_info"]
            
            new_batch.batch["token_level_scores"] = reward_tensor
            
            if reward_extra_infos_dict:
                new_batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
        
        return new_batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            trainer_config=self.config,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        # if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        dataloader_len = len(self.train_dataloader)
        print(f"################## dataloader length: {dataloader_len}")

        num_epoch = 0

        for epoch in range(self.config.trainer.total_epochs):
            dataloader_iter = iter(self.train_dataloader)
            print(f"############## start training on epoch: {epoch}")
            
            while True:
                # Check if training is complete
                # print(f"########### global_steps: {self.global_steps}")
                # if self.global_steps % (dataloader_len + 1) == 0 and self.global_steps > 0:
                #     break
                is_last_step = self.global_steps >= self.total_training_steps
                
                # ========== Phase 1: Collect enough valid rollouts ==========
                metrics = {}
                timing_raw = {}
                
                # Rollout accumulation state
                accumulated_batch = None
                accumulated_rollout_count = 0
                num_gen_batches_for_this_step = 0
                first_gen_batch = None  # Keep the first generated batch for logging
                oversample_data_pool = None
                
                # Video filtering statistics
                total_video_samples_seen = 0
                total_video_samples_filtered = 0
                
                # Batch alignment statistics
                total_batches_aligned = 0
                total_images_added = 0
                total_images_discarded = 0
                
                # Target batch size
                target_traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                
                # Get configuration for generation batch processing
                max_video_gen_batch_size = self.config.data.get("max_video_gen_batch_size", 32)
                gen_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
                align_method = self.config.data.get("gen_batch_size_align_method", "up_resample_image")
                
                # Generate and filter rollouts until we have enough
                with _timer("step", timing_raw):
                    while accumulated_rollout_count < target_traj_bsz:
                        # Get next batch from dataloader
                        try:
                            # if self.global_steps % (dataloader_len) == 0 and self.global_steps > 0:
                            #     dataloader_iter = iter(self.train_dataloader)
                            batch_dict = next(dataloader_iter)
                        except:
                            # Dataloader exhausted, should not happen if total_training_steps is set correctly
                            dataloader_iter = iter(self.train_dataloader)
                            batch_dict = next(dataloader_iter)
                            num_epoch += 1
                            # break
                        
                        num_gen_batches_for_this_step += 1
                        
                        # Filter video samples before generation to control GPU memory
                        batch_dict, video_count, video_filtered = self._filter_video_samples_in_dict(
                            batch_dict, max_video_gen_batch_size
                        )
                        total_video_samples_seen += video_count
                        total_video_samples_filtered += video_filtered
                        
                        # Align batch size for generation requirements (multiple of 8)
                        batch_dict, align_info = self._align_batch_size_for_generation(
                            batch_dict, dataloader_iter, align_method, gen_batch_size
                        )
                        if align_info["changed"]:
                            total_batches_aligned += 1
                            if "images_added" in align_info:
                                total_images_added += align_info["images_added"]
                            if "images_discarded" in align_info:
                                total_images_discarded += align_info["images_discarded"]
                        
                        # Generate rollouts for this batch
                        new_batch = self._generate_and_score_batch(batch_dict, timing_raw)

                        # Compute advantages
                        new_batch.batch["response_mask"] = compute_response_mask(new_batch)
                        with _timer("adv", timing_raw):
                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            new_batch = compute_advantage(
                                new_batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            )
                        
                        # Store the first batch for logging
                        if first_gen_batch is None:
                            first_gen_batch = new_batch

                        if oversample_data_pool is None:
                            oversample_data_pool = new_batch
                        else:
                            # oversample_data_pool = DataProto.concat([oversample_data_pool, new_batch])
                            pass
                        
                        # Apply filtering if enabled
                        if self.config.algorithm.filter_groups.enable:
                            metric_name_list = self.config.algorithm.filter_groups.metric
                            extra_filtering_config = {
                                "end_reason_filter_reserve_names": self.config.algorithm.filter_groups.get("end_reason_filter_reserve_names", None),
                            }
                            filtered_batch = rollout_filtering_function(new_batch, metric_name_list, extra_filtering_config)
                            filtered_count = len(filtered_batch)
                            print(f"Filtered {len(new_batch)} -> {filtered_count} rollouts")
                        else:
                            filtered_batch = new_batch
                            filtered_count = len(filtered_batch)
                        
                        # Accumulate filtered rollouts
                        if accumulated_batch is None:
                            accumulated_batch = filtered_batch
                        else:
                            accumulated_batch = DataProto.concat([accumulated_batch, filtered_batch])
                        accumulated_rollout_count += filtered_count
                        
                        # Check if filtering is enabled and we need more rollouts
                        if self.config.algorithm.filter_groups.enable and accumulated_rollout_count < target_traj_bsz:
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches > 0 and num_gen_batches_for_this_step >= max_num_gen_batches:
                                raise ValueError(
                                    f"Generated {num_gen_batches_for_this_step} batches but only collected "
                                    f"{accumulated_rollout_count}/{target_traj_bsz} valid rollouts. "
                                    f"Data may be too difficult. Consider increasing max_num_gen_batches or setting it to 0."
                                )
                            print(f"Accumulated {accumulated_rollout_count}/{target_traj_bsz} rollouts, generating more...")
                            continue
                        
                        # If filtering is disabled, we only need one generation batch
                        if not self.config.algorithm.filter_groups.enable:
                            break
                    
                    if self.config.algorithm.filter_groups.std_sort_enable:
                        # Trim to exact target size if we have more than needed
                        if accumulated_rollout_count > target_traj_bsz:
                            # 1. 获取 sample_level_stds 数据
                            sample_level_stds = accumulated_batch.non_tensor_batch["sample_level_stds"]
                            
                            # 2. 找到值最大的 target_traj_bsz 个样本的索引
                            # np.argsort 返回的是排序后的索引，[::-1] 表示将其反转，得到降序排列
                            # [:target_traj_bsz] 选取前 target_traj_bsz 个索引
                            # top_k_indices = np.argsort(sample_level_stds)[::-1][:target_traj_bsz]
                            top_k_indices = np.argsort(-sample_level_stds)[:target_traj_bsz]
                            
                            # 3. 根据索引筛选 accumulated_batch
                            # 注意：这里假设 accumulated_batch 支持通过索引列表进行选择（如 PyTorch/TensorFlow 的张量或列表）
                            # 如果 accumulated_batch 是一个列表，列表推导式是更通用的方法
                            try:
                                # 尝试直接索引，适用于张量或支持此操作的自定义对象
                                new_batch = accumulated_batch[top_k_indices]
                            except (TypeError, IndexError):
                                # 如果直接索引失败，回退到列表推导式，更通用
                                new_batch = [accumulated_batch[i] for i in top_k_indices]

                            # 5. 更新所有相关变量
                            accumulated_batch = new_batch
                            accumulated_rollout_count = target_traj_bsz
                    else:
                        # Trim to exact target size if we have more than needed
                        if accumulated_rollout_count > target_traj_bsz:
                            accumulated_batch = accumulated_batch[:target_traj_bsz]
                            accumulated_rollout_count = target_traj_bsz

                    print(f"Collected {accumulated_rollout_count} rollouts from {num_gen_batches_for_this_step} generation batch(es)")
                    
                    # ========== Phase 2: Process batch for training ==========
                    batch = accumulated_batch

                    # batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # Compute global valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Compute old log probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                        )
                        metrics["actor/entropy_loss"] = entropy_loss.detach().item()
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # Compute reference log probs if using reference policy
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # # Compute advantages
                    # with _timer("adv", timing_raw):
                    #     norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                    #     batch = compute_advantage(
                    #         batch,
                    #         adv_estimator=self.config.algorithm.adv_estimator,
                    #         gamma=self.config.algorithm.gamma,
                    #         lam=self.config.algorithm.lam,
                    #         num_repeat=self.config.actor_rollout_ref.rollout.n,
                    #         norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    #     )

                    # ========== Phase 3: Model updates ==========

                    # Update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_metrics)

                    # Update actor (with optional critic warmup)
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_metrics)
                    
                    # ========== Phase 4: Logging and checkpointing ==========
                    
                    # Dump first generation batch if configured
                    if oversample_data_pool is not None:
                        oversample_data_pool_dump_dir = self.config.trainer.get("the_oversample_data_pool_rollout_data_dir", None)
                        if oversample_data_pool_dump_dir:
                            with _timer("dump_oversample_data_pool", timing_raw):
                                self._dump_batch_generations(oversample_data_pool, oversample_data_pool_dump_dir)
                    
                    # Dump accumulated/filtered batch if configured
                    rollout_dump_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_dump_dir:
                        with _timer("dump_rollout_batch", timing_raw):
                            self._dump_batch_generations(batch, rollout_dump_dir)
                    
                    # Validation
                    if (
                        self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Checkpointing
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # ========== Phase 5: Collect and log metrics ==========
                
                # Add various metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_data_metrics_oversample_pool(batch=oversample_data_pool, use_critic=self.use_critic))
                metrics.update(compute_end_reason_metrics_oversample_pool(batch=oversample_data_pool, extra_filtering_config=extra_filtering_config))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if self.config.actor_rollout_ref.rollout.agent.activate_agent:
                    metrics.update(compute_agent_metrics(batch=batch))  # compute_agent_metrics_oversample_pool
                    metrics.update(compute_agent_metrics_oversample_pool(batch=oversample_data_pool))

                # Add rollout collection metrics
                metrics["train/num_gen_batches"] = num_gen_batches_for_this_step
                metrics["train/accumulated_rollout_count"] = accumulated_rollout_count
                metrics["train/num_epoch"] = num_epoch
                
                # Add video filtering metrics
                if total_video_samples_seen > 0:
                    metrics["train/video_samples_seen"] = total_video_samples_seen
                    metrics["train/video_samples_filtered"] = total_video_samples_filtered
                    metrics["train/video_filter_rate"] = total_video_samples_filtered / total_video_samples_seen if total_video_samples_seen > 0 else 0.0
                
                # Add batch alignment metrics
                if total_batches_aligned > 0:
                    metrics["train/batches_aligned"] = total_batches_aligned
                    metrics["train/alignment_method"] = align_method
                    if total_images_added > 0:
                        metrics["train/images_added"] = total_images_added
                    if total_images_discarded > 0:
                        metrics["train/images_discarded"] = total_images_discarded
                
                # Log all metrics
                logger.log(data=metrics, step=self.global_steps, batch=batch, tokenizer=self.tokenizer)

                # Check if training is complete
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # Advance to next step
                progress_bar.update(1)
                self.global_steps += 1
