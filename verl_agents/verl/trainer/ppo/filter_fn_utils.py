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
import torch

from verl.workers.agent.parallel_env import EndReasonEnum

def dynamic_sampling_fn(new_batch):

    new_batch.non_tensor_batch["seq_reward"] = (
        new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
    )

    # Collect the sequence reward for each trajectory
    prompt_uid2metric_vals = defaultdict(list)
    for uid, metric_val in zip(
        new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch["seq_reward"], strict=True
    ):
        prompt_uid2metric_vals[uid].append(metric_val)

    prompt_uid2metric_std = {}
    prompt_uid2metric_mean = {}
    for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
        prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
        prompt_uid2metric_mean[prompt_uid] = np.mean(metric_vals)

    kept_prompt_uids = []
    mean_0_traj_uids = []
    mean_1_traj_uids = []

    for uid, std in prompt_uid2metric_std.items():
        if std > 0 or len(prompt_uid2metric_vals[uid]) == 1:
            kept_prompt_uids.append(uid)
        else:
            if prompt_uid2metric_mean[uid] == 0:
                mean_0_traj_uids.append(uid)
            else:
                mean_1_traj_uids.append(uid)

    # TODO: number of trajectories in one group should be config.actor_rollout_ref.rollout.n ?

    kept_traj_idxs = []
    for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
        if traj_from_prompt_uid in kept_prompt_uids:
            kept_traj_idxs.append(idx)

    print(f"[INFO batch filter] dynamic sampling std=0 filtering: {len(new_batch)} -> {len(kept_traj_idxs)} trajs, ({len(mean_0_traj_uids)} mean=0, {len(mean_1_traj_uids)} mean=1)")

    new_batch = new_batch[kept_traj_idxs]

    filter_reason = {
        "dynamic_sampling:_kept": kept_prompt_uids,
        "dynamic_sampling:std_0_mean_0": mean_0_traj_uids,
        "dynamic_sampling:std_0_mean_1": mean_1_traj_uids,
    }

    return new_batch, filter_reason

def hasimage_filtering_fn(new_batch):

    kept_traj_idxs = []
    hasimage_filter_reason = {
        "hasimage:_kept": [],
        "hasimage:no_image": [],
    }
    for idx, has_image in enumerate(new_batch.non_tensor_batch["hasimage"]):
        if has_image:
            kept_traj_idxs.append(idx)
            hasimage_filter_reason["hasimage:_kept"].append(idx)
        else:
            hasimage_filter_reason["hasimage:no_image"].append(idx)

    print(f"[INFO batch filter] has image filtering: {len(new_batch)} -> {len(kept_traj_idxs)} trajs")

    new_batch = new_batch[kept_traj_idxs]

    return new_batch, hasimage_filter_reason

def trajlength_filtering_fn(new_batch, max_length=128000):

    kept_traj_idxs = []
    trajlength_filter_reason = {
        "trajlength:_kept": [],
        f"trajlength:exceed_max_length_{max_length}": [],
    }
    for idx, trajlength in enumerate(new_batch.non_tensor_batch["trajlength"]):
        if trajlength <= max_length:
            kept_traj_idxs.append(idx)
            trajlength_filter_reason["trajlength:_kept"].append(idx)
        else:
            trajlength_filter_reason[f"trajlength:exceed_max_length_{max_length}"].append(idx)

    print(f"[INFO batch filter] traj length filtering: {len(new_batch)} -> {len(kept_traj_idxs)} trajs")

    new_batch = new_batch[kept_traj_idxs]

    return new_batch, trajlength_filter_reason

def trajlength_filtering_fn_validation(new_batch, max_length=128000):
    """
    根据轨迹长度过滤批次，并返回保留的批次、被过滤掉的批次以及过滤原因。

    Args:
        new_batch: 输入的批次对象，需要包含 non_tensor_batch["trajlength"]。
        max_length: 轨迹的最大允许长度。

    Returns:
        tuple:
            - new_batch (Batch): 包含所有满足长度条件的轨迹的新批次。
            - filtered_out_batch (Batch): 包含所有因长度超限而被过滤掉的轨迹的批次。
            - trajlength_filter_reason (dict): 记录了保留和过滤轨迹索引的字典。
    """
    kept_traj_idxs = []
    # 使用更清晰的变量名来存储被过滤掉的索引
    filtered_traj_idxs = []
    
    trajlength_filter_reason = {
        "trajlength:_kept": [],
        f"trajlength:exceed_max_length_{max_length}": [],
    }

    for idx, trajlength in enumerate(new_batch.non_tensor_batch["trajlength"]):
        if trajlength <= max_length:
            kept_traj_idxs.append(idx)
            trajlength_filter_reason["trajlength:_kept"].append(idx)
        else:
            filtered_traj_idxs.append(idx)
            trajlength_filter_reason[f"trajlength:exceed_max_length_{max_length}"].append(idx)

    print(f"[INFO batch filter] traj length filtering: {len(new_batch)} -> {len(kept_traj_idxs)} trajs")
    
    # --- 修正后的逻辑顺序 ---
    # 1. 首先，保存原始批次的引用，以便后续切片
    original_batch = new_batch
    
    # 2. 创建保留的批次
    new_batch = original_batch[kept_traj_idxs]
    
    # 3. 创建被过滤掉的批次
    filtered_out_batch = original_batch[filtered_traj_idxs]
    # --- 修改结束 ---

    # 返回三个对象：保留的批次、被过滤的批次和原因字典
    return new_batch, filtered_out_batch, trajlength_filter_reason, kept_traj_idxs, filtered_traj_idxs


def vision_token_nums_image_nums_consistent_filtering_fn(new_batch):

    kept_traj_idxs = []
    vision_token_nums_image_nums_consistent_filter_reason = {
        "consis:_kept": [],
        "consis:no_image": [],
    }
    for idx, consis in enumerate(new_batch.non_tensor_batch["is_vision_token_nums_image_nums_consistent"]):
        if consis:
            kept_traj_idxs.append(idx)
            vision_token_nums_image_nums_consistent_filter_reason["consis:_kept"].append(idx)
        else:
            vision_token_nums_image_nums_consistent_filter_reason["consis:no_image"].append(idx)

    print(f"[INFO batch filter] vision_token_nums_image_nums_consistent filtering: {len(new_batch)} -> {len(kept_traj_idxs)} trajs")

    new_batch = new_batch[kept_traj_idxs]

    return new_batch, vision_token_nums_image_nums_consistent_filter_reason

def end_reason_filtering_fn(new_batch, extra_filtering_config=None):
    """
    根据轨迹的结束原因对批次进行过滤。

    参数:
        new_batch: 原始批次数据，需支持切片操作。
        extra_filtering_config: 可选配置字典，可包含自定义保留的结束原因列表。

    返回:
        new_batch: 过滤后的批次。
        end_reason_filter_reason: 字典，记录每种结束原因对应的轨迹索引。
    """
    end_reason_filter_reserve_names = [EndReasonEnum.DONE.name]

    if extra_filtering_config is not None and "end_reason_filter_reserve_names" in extra_filtering_config:
        end_reason_filter_reserve_names = extra_filtering_config["end_reason_filter_reserve_names"]

    kept_traj_idxs = []
    end_reason_filter_reason = {}

    # 遍历批次中的每个轨迹，记录其结束原因并决定是否保留
    for idx, end_reason in enumerate(new_batch.non_tensor_batch["end_reason"]):
        if end_reason.name in end_reason_filter_reserve_names:
            kept_traj_idxs.append(idx)

        # 统计每种结束原因出现的轨迹索引
        key = f"end_reason:{end_reason.name}"
        if key not in end_reason_filter_reason:
            end_reason_filter_reason[key] = []
        end_reason_filter_reason[key].append(idx)

    # 构建详细的统计信息字符串
    stats_parts = []
    for key in sorted(end_reason_filter_reason.keys()):
        count = len(end_reason_filter_reason[key])
        stats_parts.append(f"{key}={count}")
    stats_str = ", ".join(stats_parts)

    filter_reason_str = ""
    for key in end_reason_filter_reason:
        filter_reason_str += f"{key}: {len(end_reason_filter_reason[key])}, "

    # 打印过滤前后的轨迹数量及保留原因和统计信息
    print(f"[INFO batch filter] end reason filtering: {len(new_batch)} -> {len(kept_traj_idxs)} trajs, {filter_reason_str} | "
          f"reserve_names={end_reason_filter_reserve_names} | {stats_str}")

    new_batch = new_batch[kept_traj_idxs]
    return new_batch, end_reason_filter_reason

# def pos_sample_neg_adv_filtering_fn(new_batch, extra_filtering_config=None):
#     is_correct_list = batch.non_tensor_batch["is_answer_right"]
#     advantages = batch.batch["advantages"]

#     max_response_length = batch.batch["responses"].shape[-1]
#     prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
#     action_or_attn_mask = batch.batch['action_mask'] if 'action_mask' in batch.batch else batch.batch['attention_mask']
#     response_mask = action_or_attn_mask[:, -max_response_length:].bool()

#     # 计算每个样本的sequence级别的优势
#     sequence_advantages = []
#     for i in range(len(is_correct_list)):
#         # 选择当前样本的有效token的优势
#         sample_advantages = advantages[i][response_mask[i]]
#         # 计算平均优势
#         if len(sample_advantages) > 0:
#             seq_adv = torch.mean(sample_advantages).item()
#         else:
#             seq_adv = 0.0
#         sequence_advantages.append(seq_adv)

#     pos_sample_indices = [i for i, is_correct in enumerate(is_correct_list) if is_correct]
#     pos_sample_count = len(pos_sample_indices)
#     pos_sample_pos_adv_num = 0
#     pos_sample_neg_adv_num = 0
    
#     if pos_sample_count > 0:
#         for idx in pos_sample_indices:
#             if sequence_advantages[idx] > 0:
#                 pos_sample_pos_adv_num += 1
#             elif sequence_advantages[idx] < 0:
#                 pos_sample_neg_adv_num += 1

#     return new_batch

def pos_sample_neg_adv_filtering_fn(new_batch):
    """
    过滤掉批次中“positive sample”但其序列级别优势为负的样本。

    Args:
        new_batch: 包含批次数据的对象。

    Returns:
        Tuple[MockBatch, dict]: 过滤后的批次和过滤原因记录。
    """
    # --- 1. 预先计算所有样本的序列级别优势 ---
    is_correct_list = new_batch.non_tensor_batch["is_answer_right"]
    advantages = new_batch.batch["advantages"]
    max_response_length = new_batch.batch["responses"].shape[-1]
    action_or_attn_mask = new_batch.batch.get('action_mask', new_batch.batch['attention_mask'])
    response_mask = action_or_attn_mask[:, -max_response_length:].bool()

    sequence_advantages = []
    for i in range(len(is_correct_list)):
        sample_advantages = advantages[i][response_mask[i]]
        seq_adv = torch.mean(sample_advantages).item() if len(sample_advantages) > 0 else 0.0
        sequence_advantages.append(seq_adv)

    # --- 2. 遵循参考格式进行过滤 ---
    kept_indices = []
    filter_reason = {
        "pos_neg_adv:kept": [],
        "pos_neg_adv:filtered_pos_with_neg_adv": [],
    }

    for idx in range(len(is_correct_list)):
        is_correct = is_correct_list[idx]
        seq_adv = sequence_advantages[idx]

        # 过滤条件：是 positive sample 且优势为负
        if is_correct and seq_adv < 0:
            filter_reason["pos_neg_adv:filtered_pos_with_neg_adv"].append(idx)
        else:
            # 保留所有其他情况
            kept_indices.append(idx)
            filter_reason["pos_neg_adv:kept"].append(idx)

    print(f"[INFO batch filter] pos/neg adv filtering: {len(new_batch)} -> {len(kept_indices)} samples")

    # 假设 new_batch 对象支持通过索引列表进行切片
    filtered_batch = new_batch[kept_indices]

    return filtered_batch, filter_reason

def end_reason_filtering_fn_validation(new_batch, extra_filtering_config=None):
    """
    根据轨迹的结束原因对批次进行过滤。

    参数:
        new_batch: 原始批次数据，需支持切片操作。
        extra_filtering_config: 可选配置字典，可包含自定义保留的结束原因列表。

    返回:
        tuple:
            - new_batch: 过滤后，包含所有满足条件的轨迹的批次。
            - filtered_out_batch: 包含所有被过滤掉的轨迹的批次。
            - end_reason_filter_reason: 字典，记录每种结束原因对应的轨迹索引。
    """
    # 默认只保留 "DONE" 的轨迹
    end_reason_filter_reserve_names = [EndReasonEnum.DONE.name]

    if extra_filtering_config is not None and "end_reason_filter_reserve_names" in extra_filtering_config:
        end_reason_filter_reserve_names = extra_filtering_config["end_reason_filter_reserve_names"]

    kept_traj_idxs = []
    # --- 核心修改部分 1: 新增一个列表来存储被过滤掉的轨迹索引 ---
    filtered_traj_idxs = []
    
    end_reason_filter_reason = {}

    # 遍历批次中的每个轨迹，记录其结束原因并决定是否保留
    for idx, end_reason in enumerate(new_batch.non_tensor_batch["end_reason"]):
        if end_reason.name in end_reason_filter_reserve_names:
            kept_traj_idxs.append(idx)
        else:
            # 如果结束原因不在保留列表中，则将其索引加入过滤列表
            filtered_traj_idxs.append(idx)

        # 统计每种结束原因出现的轨迹索引
        key = f"end_reason:{end_reason.name}"
        if key not in end_reason_filter_reason:
            end_reason_filter_reason[key] = []
        end_reason_filter_reason[key].append(idx)

    # 构建详细的统计信息字符串
    stats_parts = []
    for key in sorted(end_reason_filter_reason.keys()):
        count = len(end_reason_filter_reason[key])
        stats_parts.append(f"{key}={count}")
    stats_str = ", ".join(stats_parts)

    filter_reason_str = ""
    for key in end_reason_filter_reason:
        filter_reason_str += f"{key}: {len(end_reason_filter_reason[key])}, "

    # 打印过滤前后的轨迹数量及保留原因和统计信息
    print(f"[INFO batch filter] end reason filtering: {len(new_batch)} -> {len(kept_traj_idxs)} trajs, {filter_reason_str} | "
          f"reserve_names={end_reason_filter_reserve_names} | {stats_str}")

    # --- 核心修改部分 2: 创建被过滤掉的批次 ---
    # 使用原始的 new_batch 和 filtered_traj_idxs 创建被过滤掉的批次
    # 这一步必须在 new_batch 变量被重新赋值之前执行
    filtered_out_batch = new_batch[filtered_traj_idxs]
    
    # 创建保留的批次
    new_batch = new_batch[kept_traj_idxs]
    
    # --- 核心修改部分 3: 更新返回值 ---
    # 返回保留的批次、被过滤的批次和原因字典
    return new_batch, filtered_out_batch, end_reason_filter_reason, kept_traj_idxs, filtered_traj_idxs

def rollout_filtering_function(new_batch, metric_name_list, extra_filtering_config=None):
    print(f"[INFO batch filter] rolling out filtering on metrics: {metric_name_list}")

    if "seq_reward" in metric_name_list:
        new_batch, dynamic_sampling_filter_reason = dynamic_sampling_fn(new_batch)

    if "hasimage" in metric_name_list:
        new_batch, hasimage_filter_reason = hasimage_filtering_fn(new_batch)
        
    if "vtoken_images_num_consis" in metric_name_list:
        new_batch, vision_token_nums_image_nums_consistent_filter_reason = vision_token_nums_image_nums_consistent_filtering_fn(new_batch)

    if "trajlength" in metric_name_list:
        new_batch, trajlength_filter_reason = trajlength_filtering_fn(new_batch)

    if "end_reason" in metric_name_list:
        new_batch, end_reason_filter_reason = end_reason_filtering_fn(new_batch, extra_filtering_config)

    if "pos_sample_neg_adv" in metric_name_list:
        new_batch, pos_sample_neg_adv_filter_reason = pos_sample_neg_adv_filtering_fn(new_batch)

    return new_batch

def rollout_filtering_function_validation(new_batch, metric_name_list, extra_filtering_config=None):
    """
    按顺序应用多个过滤函数，并基于索引返回最终过滤后的批次、所有被过滤掉的批次、
    所有过滤原因以及对应的原始索引。

    参数:
        new_batch (Batch): 原始批次数据。
        metric_name_list (list[str]): 需要应用的过滤指标名称列表，例如 ["trajlength", "end_reason"]。
        extra_filtering_config (dict, optional): 可选配置字典，用于传递给具体的过滤函数。

    返回:
        tuple:
            - final_batch (Batch): 经过所有过滤后的最终批次。
            - final_filtered_out_batch (Batch): 一个批次，包含所有在过滤过程中被移除的样本。
            - all_filter_reasons (list[dict]): 一个列表，包含每个被过滤样本的详细原因字典。
            - kept_indices (list[int]): 最终保留的样本在原始批次中的索引列表。
            - filtered_out_indices (list[int]): 被过滤掉的样本在原始批次中的索引列表。
    """
    print(f"[INFO batch filter validation] Rolling out filtering on metrics: {metric_name_list}")

    # --- 核心修改部分 1: 初始化索引和结果收集器 ---
    
    # 保存原始批次的引用，用于最后提取数据
    original_batch = new_batch
    
    # 初始化当前保留的索引，一开始是所有样本的索引
    currently_kept_indices = list(range(len(original_batch)))
    
    # 初始化一个列表来收集所有被过滤掉的样本信息 (索引, 原因)
    all_filtered_out_info = []

    # --- 核心修改部分 2: 顺序应用过滤器并更新索引 ---
    
    for metric_name in metric_name_list:
        # 如果当前没有样本剩下，则提前终止过滤
        if not currently_kept_indices:
            print(f"[INFO] No samples left to filter after previous steps. Skipping {metric_name}.")
            break

        # 根据当前保留的索引，从原始批次中选出子批次，用于本次过滤
        print(f"########### currently_kept_indices: {currently_kept_indices}")
        current_batch_to_filter = original_batch[currently_kept_indices]

        # 根据指标名称调用相应的过滤函数
        if metric_name == "trajlength":
            # 假设过滤函数返回的索引是相对于输入批次 的
            _, _, reasons_relative, kept_idxs_relative, filtered_idxs_relative = trajlength_filtering_fn_validation(
                current_batch_to_filter
            )
        elif metric_name == "end_reason":
            _, _, reasons_relative, kept_idxs_relative, filtered_idxs_relative = end_reason_filtering_fn_validation(
                current_batch_to_filter, extra_filtering_config
            )
        else:
            print(f"[WARNING] Unknown filtering metric: {metric_name}. Skipping.")
            continue

        # --- 核心修改部分 3: 更新索引和收集过滤信息 ---
        
        # 1. 将本次过滤掉的相对索引 映射回原始批次中的绝对索引
        newly_filtered_original_indices = [currently_kept_indices[i] for i in filtered_idxs_relative]
        
        # 2. 将被过滤掉的样本索引和对应的原因收集起来
        for original_idx, reason in zip(newly_filtered_original_indices, reasons_relative):
            all_filtered_out_info.append((original_idx, reason))
            
        # 3. 更新当前保留的索引列表，为下一次过滤做准备
        currently_kept_indices = [currently_kept_indices[i] for i in kept_idxs_relative]

    # --- 核心修改部分 4: 根据最终索引构建返回的批次 ---

    # 1. 构建最终保留的批次
    final_batch = original_batch[currently_kept_indices]

    # 2. 构建最终被过滤掉的批次
    final_filtered_out_indices = [info[0] for info in all_filtered_out_info]
    final_filtered_out_batch = original_batch[final_filtered_out_indices]

    # 3. 提取所有过滤原因
    all_filter_reasons = [info[1] for info in all_filtered_out_info]

    # --- 核心修改部分 5: 返回索引 ---
    # 将最终保留的索引和过滤掉的索引也一并返回
    return final_batch, final_filtered_out_batch, all_filter_reasons, currently_kept_indices, final_filtered_out_indices

