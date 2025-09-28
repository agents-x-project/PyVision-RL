import os
import json
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
import torch
from tqdm import tqdm
import math
from io import BytesIO
from PIL import Image
import base64
import io
from openai import OpenAI
import requests


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='qwen', help='Model name for result save')
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='http://10.39.19.140:8000/v1', help='API URL')
parser.add_argument('--vstar_bench_path', type=str, default=None, help='Path to the V* benchmark')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for evaluation')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()


openai_api_key = args.api_key
openai_api_base = args.api_url

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models['data'][0]['id']
else:
    eval_model_name = args.eval_model_name

vstar_bench_path = args.vstar_bench_path
save_path = args.save_path
save_path = os.path.join(save_path, args.model_name)
os.makedirs(save_path, exist_ok=True)
abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

instruction_prompt_system = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an optional object label.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

**Example**:  
<tool_call>  
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 100, 200], "label": "the apple on the desk"}}  
</tool_call>"""
USER_PROMPT_V2 = "\nThink first, call **image_zoom_in_tool** if needed, then answer. Format strictly as:  <think>...</think>  <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

instruction_prompt_before = """Question: {question}
""" + USER_PROMPT_V2

user_prompt = USER_PROMPT_V2

start_token = "<tool_call>"
end_token = "</tool_call>"

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# the following code is copied from qwen-vl-utils
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

from PIL import Image

def combine_images(image_paths, gap=20):
    """
    根据输入的图片路径列表返回PIL图像对象
    
    参数:
    image_paths (list): 图片路径列表
    gap (int): 两张图片之间的间隔像素，默认为20
    
    返回:
    PIL.Image.Image: 处理后的PIL图像对象
    
    异常:
    ValueError: 当图片路径数量不是1或2时抛出异常
    FileNotFoundError: 当指定的图片路径不存在时抛出异常
    """
    if not isinstance(image_paths, list):
        raise ValueError("输入必须是列表类型")
    
    if len(image_paths) == 0:
        raise ValueError("图片路径列表不能为空")
    
    if len(image_paths) == 1:
        # 单张图片，直接返回PIL对象
        return Image.open(image_paths[0])
    
    elif len(image_paths) == 2:
        # 两张图片，进行左右拼贴
        try:
            # 打开两张图片
            img1 = Image.open(image_paths[0])
            img2 = Image.open(image_paths[1])
            
            # 获取图片尺寸
            width1, height1 = img1.size
            width2, height2 = img2.size
            
            # 计算拼贴后图片的总尺寸
            total_width = width1 + gap + width2
            total_height = max(height1, height2)
            
            # 创建新的空白图片
            combined_img = Image.new('RGB', (total_width, total_height), color='white')
            
            # 粘贴第一张图片（左对齐）
            combined_img.paste(img1, (0, 0))
            
            # 粘贴第二张图片（右对齐，考虑高度差异）
            y_offset = 0
            if height1 > height2:
                y_offset = (height1 - height2) // 2
            elif height2 > height1:
                # 第一张图片需要垂直居中
                combined_img = Image.new('RGB', (total_width, total_height), color='white')
                combined_img.paste(img1, (0, (height2 - height1) // 2))
                y_offset = 0
            
            combined_img.paste(img2, (width1 + gap, y_offset))
            
            return combined_img
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"图片文件未找到: {e}")
        except Exception as e:
            raise Exception(f"处理图片时发生错误: {e}")
    
    else:
        raise ValueError(f"只支持1张或2张图片，当前输入了{len(image_paths)}张图片")


def process(item):

    return_save_path = f"/mnt/petrelfs/zhaoshitian/eaigc3_t_zhaoshitian/data/TIR-Bench/results/deepeyes/{item['doc_id']}.json"

    if os.path.exists(return_save_path):
        save_info = json.load(open(return_save_path, "r"))
        return save_info
    # img, test_path = img_arg
    img_path = item['images']
    # img_path = os.path.join(test_path, img)
    pil_img = combine_images(img_path, gap=20)


    question = item['problem'].split("<image>\n", -1)[-1]
    
    prompt = instruction_prompt_before.format(question=question)
    # pil_img = Image.open(img_path)

    base64_image = encode_pil_image_to_base64(pil_img)

    messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    print_messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_message = messages

    response_message = ""

    status = 'success'
    try_count = 0
    turn_idx = 0
    try:
        while '</answer>' not in response_message:
            if '</answer>' in response_message and '<answer>' in response_message:
                break

            if try_count > 10:
                break

            params = {
                "model": eval_model_name,
                "messages": chat_message,
                "temperature": 0.0,
                "max_tokens": 10240,
                "stop": ["<|im_end|>\n".strip()],
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content
            
            if start_token in response_message:
                action_list = response_message.split(start_token)[1].split(end_token)[0].strip()
                action_list = eval(action_list)

                bbox_list = []
                cropped_pil_image_content_list = []

                bbox_str = action_list['arguments']['bbox_2d']
                bbox = bbox_str
                left, top, right, bottom = bbox
                cropped_image = pil_img.crop((left, top, right, bottom))
                new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                cropped_pil_image = encode_pil_image_to_base64(cropped_image)
                bbox_list.append(bbox)
                cropped_pil_image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"}}
                cropped_pil_image_content_list.append(cropped_pil_image_content)

                if len(bbox_list) == 1:
                    bbox_list = bbox_list[0]
                user_msg = user_prompt

                content_f = []
                content_f.append({"type": "text", "text": "<tool_response>"})
                for cropped_pil_image_content in cropped_pil_image_content_list:
                    content_f.append(cropped_pil_image_content)
                content_f.append({"type": "text", "text": user_msg})
                content_f.append({"type": "text", "text": "</tool_response>"})

                _message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": content_f,
                    }
                ]

                chat_message.extend(_message)
            
                p_message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
                            {"type": "text", "text": user_msg},
                        ],
                    }
                ]
                print_messages.extend(p_message)
                turn_idx += 1
            else:
                p_message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    }
                ]
                print_messages.extend(p_message)


            try_count += 1
    except Exception as e:
        print(f"Error!!!!", e)
        status = 'error'
                

    if '</answer>' in response_message and '<answer>' in response_message:
        output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
    else:
        output_text = response_message

    save_info = {}
    save_info['id'] = item['doc_id']
    save_info['image'] = img_path
    save_info['task'] = item['task']
    save_info['question'] = question
    save_info['answer'] = item['solution']
    save_info['pred_ans'] = output_text
    save_info['pred_output'] = print_messages
    save_info['status'] = status

    with open(return_save_path, "w") as f:
        json.dump(save_info, f, indent=4)

    return save_info


if __name__ == "__main__":
    test_types = ['tir-bench']

    for test_type in test_types:
        save_name = f"result_{test_type}_{args.model_name}.json"
        save_json = []
        test_path = "/mnt/petrelfs/zhaoshitian/eaigc3_t_zhaoshitian/data/TIR-Bench/TIR-Bench-V3/TIR_collection_reform_minio3.json"
        pool = multiprocessing.Pool(processes=args.num_workers)
        item_list = json.load(open(test_path, "r"))

        with tqdm(total=len(item_list), desc="Processing V* "+test_type) as pbar:
            for result in pool.imap(process, item_list):
                # print(result)
                if result is not None:
                    save_json.append(result)
                    pbar.update(1)

        pool.close()
        pool.join()
    
        with open(os.path.join(save_path, save_name), 'w') as f:
            json.dump(save_json, f, indent=4)
