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
from PIL import Image
from io import BytesIO
import base64

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import hashlib
import torch
from pathlib import Path
from tqdm import tqdm

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

def encode_image(image):
    """
    Convert a PIL.Image object or image file path to base64-encoded string, and get resolution info.
    
    Args:
        image: Can be a PIL.Image object or image file path.
    Returns:
        dict with keys:
        - 'base64': base64-encoded string
        - 'width': width in pixels
        - 'height': height in pixels
        - 'resolution': string "widthxheight"
    """
    img_obj = None
    
    if isinstance(image, str):
        # Handle file path
        img_obj = Image.open(image)
        with open(image, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # Handle PIL.Image object
        img_obj = image
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    width, height = img_obj.size
    
    return {
        'base64': base64_str,
        'width': width,
        'height': height
    }

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

def transfer_to_rl_form_image_w_mm_hint(data_list, prompt_template_path):
    if "mm_hint" in data_list[0]:
        return data_list
    else:
        rl_template_list = json.load(open(prompt_template_path, "r"))
        prompt_prefix = rl_template_list['vistool_with_img_info_v2']
        new_data_list = []
        for item in tqdm(data_list, desc=f"Processing dataset: {data_list[0]['data_source']}......"):

            image_path = item['image_path']
            question = item['question']
            answer = item['answer']

            img_result = encode_image(image_path)
            image_base64 = img_result['base64']
            width = img_result['width']
            height = img_result['height']
            prompt = "<image>\n" + prompt_prefix.format(query=question, width=width, height=height)

            new_item = {}
            new_item['prompt'] = [{"content": prompt, "role": "user"}]
            new_item['data_source'] = item['data_source']
            new_item['ability'] = item['ability']
            new_item['env_name'] = "pyvision_gym_w_image_hint"
            new_item['reward_model'] = {"ground_truth": answer, "style": "model"}
            new_item['extra_info'] = {
                "answer": answer,
                "index": int(item['id']),
                "question": question,
                "split": "train"
            }
            new_item['mm_hint'] = {
                "hint_path": image_path,
                "hint_type": "image"
            }

            new_data_list.append(new_item)

        return new_data_list

class RLHFDataset(Dataset):
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

        self.config = config
        self.min_pixels = config.min_pixels
        self.max_pixels = config.max_pixels
        print("######################################################")
        print(f"min pixels in image processor: {self.processor.image_processor.min_pixels}, real min pixels: {self.min_pixels}")
        print(f"max pixels in image processor: {self.processor.image_processor.max_pixels}, real max pixels: {self.max_pixels}")
        print("######################################################")

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "/inspire/hdd/project/qproject-assement/zhangkaipeng-24043/zhaoshitian/vis_tool_train/cache"))
        self.prompt_template_path = config.get("prompt_template_path", None)
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
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
            if "image_val_dataset" in data_file_path:
                save_data_path = data_file_path.replace("image_val_dataset", "processed_w_image_hint")
                if os.path.exists(save_data_path):
                    data_list = json.load(open(save_data_path, "r"))
                else:
                    data_list = json.load(open(data_file_path, "r"))
                    data_list = transfer_to_rl_form_image_w_mm_hint(data_list, self.prompt_template_path)
                    save_data_path = data_file_path.replace("image_val_dataset", "processed_w_image_hint")
                    with open(save_data_path, "w") as f:
                        json.dump(data_list, f, indent=4)  
            else:
                data_list = json.load(open(data_file_path, "r"))
            dataframes += data_list

        self.dataframe = dataframes

        print(f"dataset len: {len(self.dataframe)}")
        # torch.save(self.dataframe, cache_file)

        print(f"Final dataset len: {len(self.dataframe)}")

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages_pyvision(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    # images = [process_image(image) for image in doc[image_key]] if image_key in doc else None
                    # videos = [process_video(video) for video in doc[video_key]] if video_key in doc else None

                    images = [process_image(image) for image in [doc[image_key]]] if image_key in doc else None
                    videos = [process_video(video) for video in [doc[video_key]]] if video_key in doc else None

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True
                        )
                    )

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def _build_messages_pyvision(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        for message in messages:
            content = message["content"]
            content_list = []
            for segment in re.split("(<image>|<video>)", content):
                if segment == "<image>":
                    content_list.append({"type": "text", "text": "<image_clue_0>"})
                    content_list.append({"type": "image"})
                    content_list.append({"type": "text", "text": "</image_clue_0>"})
                else:
                    content_list.append({"type": "text", "text": segment})
                    # print(f"<image> is not in the init prompt !!!!!!!!!!!")

            message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages_pyvision(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_raw_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}
            origin_multi_modal_data = {}

            images = None
            hint_type = row_dict['mm_hint']['hint_type']
            assert hint_type == "image", ("For dataset with mm_hint, the hint_type must be image.")
            # if self.image_key in row_dict:
            if hint_type == "image":
                image_path = row_dict['mm_hint']['hint_path']
                origin_images = [process_raw_image(image_path)]
                images = [process_image(image_path, self.min_pixels, self.max_pixels)]
                multi_modal_data["image"] = images
                origin_multi_modal_data["image"] = origin_images

            model_inputs = self.processor(text=[raw_prompt], images=images,  return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict['origin_multi_modal_data'] = origin_multi_modal_data
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

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

        # if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
        if self.processor is not None and (self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor" or self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessorFast"):
            from verl.models.transformers.qwen2_vl import get_rope_index

            # position_ids = [
            #     get_rope_index(
            #         self.processor,
            #         input_ids=input_ids[0],
            #         image_grid_thw=model_inputs.get("image_grid_thw"),
            #         video_grid_thw=model_inputs.get("video_grid_thw"),
            #         second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
            #         attention_mask=attention_mask[0],
            #     )
            # ]  # (1, 3, seq_len)

            vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
            )    

            device = vision_position_ids.device
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long, device=device)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item(), device=device)
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)

            # position_ids_list += position_ids

        else:
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
