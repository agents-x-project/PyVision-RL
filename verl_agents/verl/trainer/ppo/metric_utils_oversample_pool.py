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
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from verl import DataProto
from verl.workers.agent.parallel_env import EndReasonEnum


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    if 'action_mask' in batch.batch:
        action_mask = batch.batch['action_mask'][:, -batch.batch['responses'].shape[-1]:]
        obs_mask = response_mask * (1 - action_mask)
        obs_length = obs_mask.sum(-1).float()
    else:
        obs_length = torch.zeros_like(response_length)
    response_length -= obs_length

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
        obs_length=obs_length,
    )

def compute_data_metrics_oversample_pool(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    # TODO: add response length
    sample_level_stds = batch.non_tensor_batch["sample_level_stds"]
    sequence_score = batch.batch["token_level_scores"].sum(-1)
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)

    is_correct_list = batch.non_tensor_batch["is_answer_right"]

    num_correct = sum([1 if _ else 0 for _ in is_correct_list])

    acc_batch = num_correct / len(is_correct_list)

    # data source distribution computation
    data_source_list = batch.non_tensor_batch["data_source"]

    data_source_counts = Counter(data_source_list)
    total_data_source = len(data_source_list)

    # 构造一个字典，value 为比例
    source_ratios = {k: v / total_data_source for k, v in data_source_counts.items()}

    # ability distribution computation
    data_ability_list = batch.non_tensor_batch["ability"]

    data_ability_counts = Counter(data_ability_list)
    total_data_ability = len(data_ability_list)

    # 构造一个字典，value 为比例
    ability_ratios = {k: v / total_data_ability for k, v in data_ability_counts.items()}

    # 使用 wandb.Histogram 记录比例分布
    # wandb.log({"source_ratio_per_step": wandb.Histogram(list(source_ratios.values()))})

    uid_list = batch.non_tensor_batch["uid"]

    # 假设 uid 和 reward 已定义
    # uid: [512,]，reward: [512,]

    # 步骤1：计算唯一 uid 的数量（即 64）
    unique_uids = np.unique(uid_list)
    unique_uid_count = len(unique_uids)

    # 步骤3：按 uid 分组并 reshape
    _, inverse_indices = np.unique(uid_list, return_inverse=True)

    acc_per_group_list = []

    for i in range(unique_uid_count):
        mask = (inverse_indices == i)
        sequence_score_one_group = sequence_score[mask]
        # reward_reformed[i] = sequence_score_one_group
        correct_num_one_group = [1 if _ > 0 else 0 for _ in sequence_score_one_group]
        acc_one_group = sum(correct_num_one_group) / len(sequence_score_one_group)
        acc_per_group_list.append(acc_one_group)

    per_group_one_ratio = sum([1 if _ == 1.0 else 0 for _ in acc_per_group_list]) / len(acc_per_group_list)
    per_group_zero_ratio = sum([1 if _ == 0.0 else 0 for _ in acc_per_group_list]) / len(acc_per_group_list)


    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]
    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    action_or_attn_mask = batch.batch['action_mask'] if 'action_mask' in batch.batch else batch.batch['attention_mask']
    response_mask = action_or_attn_mask[:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]
    obs_length = response_info["obs_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    # 检查是否有tool_cnt数据
    has_tool_data = 'tool_cnt' in batch.batch
    tool_cnt_tensor = None
    if has_tool_data:
        tool_cnt_tensor = batch.batch['tool_cnt']

    # 按data source分组计算指标
    data_source_metrics = {}
    for source in data_source_counts.keys():
        # 获取当前source的mask
        source_mask = torch.tensor([ds == source for ds in data_source_list], device=sequence_score.device)
        # 创建一个与advantages形状相同的source mask
        source_mask_2d = source_mask.unsqueeze(1).expand_as(advantages)
        
        # 计算每个sequence的adv和reward
        seq_rewards = sequence_reward  # 已经是sequence级别的reward
        sample_level_stds_tensor = torch.tensor(sample_level_stds)

        
        # 获取当前source的指标
        source_advantages = torch.masked_select(advantages, source_mask_2d & response_mask)
        source_seq_rewards = torch.masked_select(seq_rewards, source_mask)
        source_sample_level_std = torch.masked_select(sample_level_stds_tensor, source_mask)
        source_response_lengths = torch.masked_select(response_length, source_mask)
        source_obs_lengths = torch.masked_select(obs_length, source_mask)
        
        # 计算当前source的数量
        source_count = data_source_counts[source]
        
        # 计算当前source的准确率
        source_is_correct = [is_correct_list[i] for i, ds in enumerate(data_source_list) if ds == source]
        source_acc = sum(source_is_correct) / len(source_is_correct) if source_is_correct else 0.0
        
        # 计算统计量
        if len(source_advantages) > 0:
            data_source_metrics[f"oversample_pool_data_source_count/{source}"] = source_count
            data_source_metrics[f"oversample_pool_data_source_acc/{source}"] = source_acc

            data_source_metrics[f"oversample_pool_data_source_adv/{source}/adv_mean"] = torch.mean(source_advantages).detach().item()
            data_source_metrics[f"oversample_pool_data_source_adv/{source}/adv_max"] = torch.max(source_advantages).detach().item()
            data_source_metrics[f"oversample_pool_data_source_adv/{source}/adv_min"] = torch.min(source_advantages).detach().item()
            
            data_source_metrics[f"oversample_pool_data_source_reward/{source}/reward_mean"] = torch.mean(source_seq_rewards).detach().item()
            data_source_metrics[f"oversample_pool_data_source_reward/{source}/reward_max"] = torch.max(source_seq_rewards).detach().item()
            data_source_metrics[f"oversample_pool_data_source_reward/{source}/reward_min"] = torch.min(source_seq_rewards).detach().item()
            
            data_source_metrics[f"oversample_pool_data_source_response_length/{source}/response_length_mean"] = torch.mean(source_response_lengths.float()).detach().item()
            data_source_metrics[f"oversample_pool_data_source_response_length/{source}/response_length_max"] = torch.max(source_response_lengths).detach().item()
            data_source_metrics[f"oversample_pool_data_source_response_length/{source}/response_length_min"] = torch.min(source_response_lengths).detach().item()
            
            data_source_metrics[f"oversample_pool_data_source_obs_length/{source}/obs_length_mean"] = torch.mean(source_obs_lengths.float()).detach().item()
            data_source_metrics[f"oversample_pool_data_source_obs_length/{source}/obs_length_max"] = torch.max(source_obs_lengths).detach().item()
            data_source_metrics[f"oversample_pool_data_source_obs_length/{source}/obs_length_min"] = torch.min(source_obs_lengths).detach().item()

            data_source_metrics[f"oversample_pool_data_source_std/{source}/std_mean"] = torch.mean(source_sample_level_std.float()).detach().item()
            data_source_metrics[f"oversample_pool_data_source_std/{source}/std_max"] = torch.max(source_sample_level_std).detach().item()
            data_source_metrics[f"oversample_pool_data_source_std/{source}/std_min"] = torch.min(source_sample_level_std).detach().item()
            
            # 添加tool相关指标
            if has_tool_data:
                source_tool_cnt = torch.masked_select(tool_cnt_tensor.squeeze(), source_mask)
                if len(source_tool_cnt) > 0:
                    # 计算 0 的占比
                    source_zero_ratio = (source_tool_cnt == 0).float().mean().item()
                    # 计算 4 的占比
                    source_max_value = torch.max(source_tool_cnt).item()
                    source_max_ratio = (source_tool_cnt == source_max_value).float().mean().item()
                    
                    data_source_metrics[f"oversample_pool_data_source_tool/{source}/tool_call_mean"] = torch.mean(source_tool_cnt).item()
                    data_source_metrics[f"oversample_pool_data_source_tool/{source}/tool_call_max"] = torch.max(source_tool_cnt).item()
                    data_source_metrics[f"oversample_pool_data_source_tool/{source}/tool_call_min"] = torch.min(source_tool_cnt).item()
                    data_source_metrics[f"oversample_pool_data_source_tool/{source}/tool_call_zero_ratio"] = source_zero_ratio
                    data_source_metrics[f"oversample_pool_data_source_tool/{source}/tool_call_max_ratio"] = source_max_ratio

    metrics = {
        # data
        "oversample_pool_data_static/std/min": np.min(sample_level_stds),
        "oversample_pool_data_static/std/mean": np.mean(sample_level_stds),
        "oversample_pool_data_static/std/max": np.max(sample_level_stds),
        # score
        "oversample_pool_critic/acc/acc_of_this_batch": acc_batch,
        "oversample_pool_critic/acc/max_group_acc_of_this_batch": max(acc_per_group_list),
        "oversample_pool_critic/acc/min_group_acc_of_this_batch": min(acc_per_group_list),
        "oversample_pool_critic/acc/one_group_acc_ratio_of_this_batch": per_group_one_ratio,
        "oversample_pool_critic/acc/zero_group_acc_ration_of_this_batch": per_group_zero_ratio,
        "oversample_pool_critic/score/mean_score_of_this_batch": torch.mean(sequence_score).detach().item(),
        "oversample_pool_critic/score/max_score_of_this_batch": torch.max(sequence_score).detach().item(),
        "oversample_pool_critic/score/min_score_of_this_batch": torch.min(sequence_score).detach().item(),
        # reward
        "oversample_pool_critic/rewards/mean_reward_of_this_batch": torch.mean(sequence_reward).detach().item(),
        "oversample_pool_critic/rewards/max_reward_of_this_batch": torch.max(sequence_reward).detach().item(),
        "oversample_pool_critic/rewards/min_reward_of_this_batch": torch.min(sequence_reward).detach().item(),
        # adv
        "oversample_pool_critic/advantages/mean_adv_of_this_batch": torch.mean(valid_adv).detach().item(),
        "oversample_pool_critic/advantages/max_adv_of_this_batch": torch.max(valid_adv).detach().item(),
        "oversample_pool_critic/advantages/min_adv_of_this_batch": torch.min(valid_adv).detach().item(),
        # returns
        "oversample_pool_critic/returns/mean_returns_of_this_batch": torch.mean(valid_returns).detach().item(),
        "oversample_pool_critic/returns/max_returns_of_this_batch": torch.max(valid_returns).detach().item(),
        "oversample_pool_critic/returns/min_returns_of_this_batch": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "oversample_pool_critic/values/mean": torch.mean(valid_values).detach().item(),
                "oversample_pool_critic/values/max": torch.max(valid_values).detach().item(),
                "oversample_pool_critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "oversample_pool_critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "oversample_pool_response_length/mean": torch.mean(response_length).detach().item(),
        "oversample_pool_response_length/max": torch.max(response_length).detach().item(),
        "oversample_pool_response_length/min": torch.min(response_length).detach().item(),
        "oversample_pool_response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),

        # obs length
        'oversample_pool_obs_length/mean': torch.mean(obs_length).detach().item(),
        'oversample_pool_obs_length/min': torch.min(obs_length).detach().item(),
        'oversample_pool_obs_length/max': torch.max(obs_length).detach().item(),

        # prompt length
        "oversample_pool_prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "oversample_pool_prompt_length/max": torch.max(prompt_length).detach().item(),
        "oversample_pool_prompt_length/min": torch.min(prompt_length).detach().item(),
        "oversample_pool_prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    
    # 添加各data source的指标
    metrics.update(data_source_metrics)
    
    return metrics

def compute_end_reason_metrics_oversample_pool(batch: DataProto, extra_filtering_config=None) -> Dict[str, Any]:
    """
    计算与结束原因相关的指标，包括分布和准确率统计。
    
    参数:
        batch: 批次数据
        extra_filtering_config: 可选配置字典，可包含自定义保留的结束原因列表
    
    返回:
        包含各种 end reason 相关指标的字典
    """
    # 获取保留的结束原因列表
    end_reason_filter_reserve_names = [EndReasonEnum.DONE.name]
    
    if extra_filtering_config is not None and "end_reason_filter_reserve_names" in extra_filtering_config:
        end_reason_filter_reserve_names = extra_filtering_config["end_reason_filter_reserve_names"]

    # 初始化统计变量
    end_reason_counts = {}
    end_reason_correct_counts = {}
    reserved_correct = 0
    reserved_total = 0
    
    # 获取数据
    end_reasons = batch.non_tensor_batch["end_reason"]
    is_answer_right = batch.non_tensor_batch["is_answer_right"]
    
    # 统计每种结束原因的数量和准确率
    for idx, end_reason in enumerate(end_reasons):
        reason_name = end_reason.name
        
        # 初始化计数器
        if reason_name not in end_reason_counts:
            end_reason_counts[reason_name] = 0
            end_reason_correct_counts[reason_name] = 0
        
        # 增加计数
        end_reason_counts[reason_name] += 1
        
        # 检查是否正确
        if is_answer_right[idx]:
            end_reason_correct_counts[reason_name] += 1
        
        # 检查是否为保留的结束原因
        if reason_name in end_reason_filter_reserve_names:
            reserved_total += 1
            if is_answer_right[idx]:
                reserved_correct += 1
    
    # 计算总样本数
    total_samples = len(end_reasons)
    
    # 计算保留样本的准确率
    reserved_accuracy = reserved_correct / reserved_total if reserved_total > 0 else 0.0
    
    # 构建 metrics 字典
    metrics = {}
    
    # 总体统计
    metrics["oversample_pool_end_reason/total_samples"] = total_samples
    metrics["oversample_pool_end_reason/reserved_samples"] = reserved_total
    metrics["oversample_pool_end_reason/reserved_accuracy"] = reserved_accuracy
    metrics["oversample_pool_end_reason/reserved_correct"] = reserved_correct
    
    # 按结束原因统计
    for reason_name in sorted(end_reason_counts.keys()):
        count = end_reason_counts[reason_name]
        correct = end_reason_correct_counts[reason_name]
        accuracy = correct / count if count > 0 else 0.0
        is_reserved = reason_name in end_reason_filter_reserve_names
        
        # 基础统计
        metrics[f"oversample_pool_end_reason/{reason_name}/count"] = count
        metrics[f"oversample_pool_end_reason/{reason_name}/correct"] = correct
        # metrics[f"oversample_pool_end_reason/{reason_name}/accuracy"] = accuracy
        # metrics[f"oversample_pool_end_reason/{reason_name}/is_reserved"] = float(is_reserved)
        
        # 比例统计
        # metrics[f"oversample_pool_end_reason/{reason_name}/ratio"] = count / total_samples if total_samples > 0 else 0.0
        # if is_reserved:
        #     metrics[f"oversample_pool_end_reason/{reason_name}/reserved_ratio"] = count / reserved_total if reserved_total > 0 else 0.0
    
    # 保留原因的详细统计
    for reason_name in end_reason_filter_reserve_names:
        if reason_name in end_reason_counts:
            count = end_reason_counts[reason_name]
            correct = end_reason_correct_counts[reason_name]
            accuracy = correct / count if count > 0 else 0.0
            
            metrics[f"oversample_pool_end_reason/reserved/{reason_name}/count"] = count
            metrics[f"oversample_pool_end_reason/reserved/{reason_name}/correct"] = correct
            metrics[f"oversample_pool_end_reason/reserved/{reason_name}/accuracy"] = accuracy
            # metrics[f"oversample_pool_end_reason/reserved/{reason_name}/contribution"] = count / reserved_total if reserved_total > 0 else 0.0
    
    # 非保留原因的统计
    non_reserved_correct = 0
    non_reserved_total = 0
    for reason_name in end_reason_counts:
        if reason_name not in end_reason_filter_reserve_names:
            non_reserved_total += end_reason_counts[reason_name]
            non_reserved_correct += end_reason_correct_counts[reason_name]
    
    non_reserved_accuracy = non_reserved_correct / non_reserved_total if non_reserved_total > 0 else 0.0
    # metrics["oversample_pool_end_reason/non_reserved_samples"] = non_reserved_total
    # metrics["oversample_pool_end_reason/non_reserved_accuracy"] = non_reserved_accuracy
    # metrics["oversample_pool_end_reason/non_reserved_correct"] = non_reserved_correct
    
    # 打印统计信息
    stats_parts = []
    for reason_name in sorted(end_reason_counts.keys()):
        count = end_reason_counts[reason_name]
        correct = end_reason_correct_counts[reason_name]
        accuracy = correct / count if count > 0 else 0.0
        is_reserved = reason_name in end_reason_filter_reserve_names
        marker = "[R]" if is_reserved else ""
        stats_parts.append(f"{reason_name}{marker}: {count} (acc={accuracy:.2f})")
    stats_str = ", ".join(stats_parts)
    
    # print(f"[INFO end reason metrics] distribution: {stats_str}")
    # print(f"[INFO end reason metrics] reserved accuracy: {reserved_accuracy:.2f} ({reserved_correct}/{reserved_total})")
    # print(f"[INFO end reason metrics] non-reserved accuracy: {non_reserved_accuracy:.2f} ({non_reserved_correct}/{non_reserved_total})")
    
    return metrics

def compute_agent_metrics_oversample_pool(batch: DataProto):
    if 'tool_cnt' not in batch.batch.keys():
        return {}

    # tool_cnt_tensor = batch.batch.pop('tool_cnt').detach().cpu()
    tool_cnt_tensor = batch.batch['tool_cnt'].detach().cpu()

    # 计算 0 的占比
    zero_ratio = (tool_cnt_tensor == 0).float().mean().item()

    # 计算 4 的占比
    max_ratio = (tool_cnt_tensor == torch.max(tool_cnt_tensor).item()).float().mean().item()

    return {
        "oversample_pool_agent/tool_call_mean": torch.mean(tool_cnt_tensor).item(),
        "oversample_pool_agent/tool_call_max": torch.max(tool_cnt_tensor).item(),
        "oversample_pool_agent/tool_call_min": torch.min(tool_cnt_tensor).item(),
        "oversample_pool_agent/tool_call_zero_ratio": zero_ratio,
        "oversample_pool_agent/tool_call_max_retio": max_ratio,
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> list[tuple[float, float]]:
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate the majority voting metric
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(
    data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42
) -> dict[str, dict[str, dict[str, float]]]:
    """Process validation metrics into a structured format.

    Args:
        data_sources: Array of data source identifiers for each sample
        sample_inputs: List of input prompts
        infos_dict: variable name -> list of values for each sample

    Returns:
        dict[str, dict[str, dict[str, float]]]: data source -> variable name -> metric value
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue
                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)
                metric[f"std@{n_resps}"] = np.std(var_vals)

                ns = []
                n = 2
                while n < n_resps:
                    ns.append(n)
                    n *= 2
                ns.append(n_resps)

                for n in ns:
                    # Best/Worst-of-N
                    [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(
                        data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed
                    )
                    metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                    metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                    # Majority voting
                    if var2vals.get("pred", None) is not None:
                        vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                        [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                            data=vote_data,
                            subset_size=n,
                            reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                            seed=seed,
                        )
                        metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val