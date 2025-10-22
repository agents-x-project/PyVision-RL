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

def dynamic_sampling_fn(new_batch):
    # we skip to the next generation batch
    # metric_name = self.config.algorithm.filter_groups.metric
    # if metric_name == "seq_final_reward":
    #     # Turn to numpy for easier filtering
    #     new_batch.non_tensor_batch["seq_final_reward"] = (
    #         new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
    #     )
    # elif metric_name == "seq_reward":
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
    for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
        prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

    kept_prompt_uids = [
        uid
        for uid, std in prompt_uid2metric_std.items()
        if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
    ]

    kept_traj_idxs = []
    for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
        if traj_from_prompt_uid in kept_prompt_uids:
            kept_traj_idxs.append(idx)

    new_batch = new_batch[kept_traj_idxs]

    return new_batch

def hasimage_filtering_fn(new_batch):

    kept_traj_idxs = []
    for idx, has_image in enumerate(new_batch.non_tensor_batch["hasimage"]):
        if has_image:
            kept_traj_idxs.append(idx)

    new_batch = new_batch[kept_traj_idxs]

    return new_batch

def trajlength_filtering_fn(new_batch):

    kept_traj_idxs = []
    for idx, trajlength in enumerate(new_batch.non_tensor_batch["trajlength"]):
        if trajlength <= 130000:
            kept_traj_idxs.append(idx)

    new_batch = new_batch[kept_traj_idxs]

    return new_batch

def rollout_filtering_function(new_batch, metric_name_list):

    if "seq_reward" in metric_name_list:
        new_batch = dynamic_sampling_fn(new_batch)
    elif "hasimage" in metric_name_list:
        new_batch = hasimage_filtering_fn(new_batch)
    elif "trajlength" in metric_name_list:
        new_batch = trajlength_filtering_fn(new_batch)

    return new_batch

