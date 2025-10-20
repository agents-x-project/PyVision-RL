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

import copy
import os
import re
from collections import defaultdict
from typing import List, Optional, Union
import json

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import hashlib
import torch
from pathlib import Path

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

def process_prompt_init(question, image_path, tokenizer, prompt_template, prompt_type):
    with open(prompt_template, "r") as fin:
        sys = json.load(fin)
    prompt_prefix = sys[prompt_type]

    img_result = encode_image(image_path)
    image_base64 = img_result['base64']
    width = img_result['width']
    height = img_result['height']
    question_with_options = question

    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "<image_clue_0>"}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}] + [{"type": "text", "text": "</image_clue_0>\n\n"}] + [{"type": "text", "text": prompt_prefix.format(query=question_with_options, width=str(width), height=str(height))}]
        }
    ]

    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return chat_prompt, messages


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class RLHF_wo_mm_hint_Dataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]
        
        all_data_file_path_list = []
        for data_file_path in data_files:
            all_data_file_path_list.append(data_file_path)

        data_files = all_data_file_path_list

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        print("######################################################")
        print(f"min pixels in image processor: {self.processor.image_processor.min_pixels}")
        print(f"max pixels in image processor: {self.processor.image_processor.max_pixels}")
        print("######################################################")
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "/mnt/petrelfs/zhaoshitian/eaigc1_t_zhaoshitian/agents_x/rl_data/cache"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.mm_hint_key = config.get("mm_hint_key", "mm_hint")
        self.max_prompt_length = config.get("max_prompt_length", 1024)

        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file_path in self.data_files:
            data_list = json.load(open(data_file_path, "r"))
            dataframes += data_list

        self.dataframe = dataframes

        print(f"dataset len: {len(self.dataframe)}")
        # torch.save(self.dataframe, cache_file)

        print(f"Final dataset len: {len(self.dataframe)}")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages_pyvision(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages_pyvision(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_raw_image, process_video, process_video_pyvision

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}
            origin_multi_modal_data = {}

            multi_modal_data['image'] = []
            origin_multi_modal_data['image'] = []

            multi_modal_hint = {}

            images = None
            videos = None
            if self.mm_hint_key in row_dict:
                mm_hint_type = row_dict[self.mm_hint_key]['hint_type']
                mm_hint_path = row_dict[self.mm_hint_key]['hint_path']
                if mm_hint_type == "image":
                    image = Image.open(mm_hint_path).convert("RGB")
                    origin_images = [process_raw_image(image)]
                    images = [process_image(image)]
                    multi_modal_data["image"] = None
                    origin_multi_modal_data["image"] = origin_images
                    # multi_modal_hint['mm_hint_content'] = origin_images
                    # multi_modal_hint['mm_hint_type'] = "image"

                if mm_hint_type == "video":
                    videos = [process_video_pyvision(mm_hint_path)]
                    multi_modal_data["video"] = None
                    origin_multi_modal_data["video"] = videos
                    # multi_modal_hint['mm_hint_content'] = videos
                    # multi_modal_hint['mm_hint_type'] = "video"

            model_inputs = self.processor(text=[raw_prompt], return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict['origin_multi_modal_data'] = origin_multi_modal_data
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
            
        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
