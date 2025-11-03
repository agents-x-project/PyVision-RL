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

# def end_reason_filtering_fn(new_batch, extra_filtering_config=None):

#     end_reason_filter_reserve_names = [EndReasonEnum.DONE.name]

#     if extra_filtering_config is not None and "end_reason_filter_reserve_names" in extra_filtering_config:
#         end_reason_filter_reserve_names = extra_filtering_config["end_reason_filter_reserve_names"]

#     kept_traj_idxs = []
#     end_reason_filter_reason = {}

#     for idx, end_reason in enumerate(new_batch.non_tensor_batch["end_reason"]):
#         if end_reason.name in end_reason_filter_reserve_names:
#             kept_traj_idxs.append(idx)
        
#         if end_reason.name not in end_reason_filter_reason:
#             end_reason_filter_reason[f"end_reason:{end_reason.name}"] = []

#         end_reason_filter_reason[f"end_reason:{end_reason.name}"].append(idx)

#     print(f"[INFO batch filter] end reason filtering: {len(new_batch)} -> {len(kept_traj_idxs)} trajs")

#     new_batch = new_batch[kept_traj_idxs]

#     return new_batch, end_reason_filter_reason

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

    return new_batch

