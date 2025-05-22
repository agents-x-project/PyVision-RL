import sys
sys.path.append("/mnt/petrelfs/zhaoshitian/vis_tool_train/verl/verl/workers/rollout/vllm_rollout")
import os
import re
import json
import base64
from io import BytesIO
from PIL import Image
import argparse
from inference_engine.shared_vis_python_exe import PythonExecutor, ImageRuntime
from openai import OpenAI

def encode_image(image):
    """
    将PIL.Image对象或图像文件路径转换为base64编码字符串，并获取分辨率信息
    
    参数:
        image: 可以是PIL.Image对象或图像文件路径
        
    返回:
        包含以下键的字典:
        - 'base64': base64编码的字符串
        - 'width': 图片宽度(像素)
        - 'height': 图片高度(像素)
        - 'resolution': 字符串形式的"宽度x高度"
    """
    img_obj = None
    
    if isinstance(image, str):
        # 处理文件路径的情况
        img_obj = Image.open(image)
        with open(image, "rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    else:
        # 处理PIL.Image对象的情况
        img_obj = image
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # 获取分辨率信息
    width, height = img_obj.size
    
    return {
        'base64': base64_str,
        'width': width,
        'height': height
    }

def check(evaluator, pred_ans, real_ans):
    if len(pred_ans) == 0:
        return []
    correctness = evaluator.score(pred_ans, real_ans)
    return correctness

def excute_codes(codes, messages, executor: PythonExecutor):
    no_code_idx = []
    codes_use = []
    for i, code in enumerate(codes):
        if code == "":
            no_code_idx.append(i)
        else:
            codes_use.append(code)
    batch_results = executor.batch_apply(codes_use, messages)
    return batch_results, no_code_idx

def process_prompt_init(question, image_path, prompt_template, prompt_type):
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

    return messages

def process_prompt_init_multi_images(question, image_path_list, prompt_template, prompt_type):
    with open(prompt_template, "r") as fin:
        sys = json.load(fin)
    prompt_prefix = sys[prompt_type]

    messages = None

    image_information = ""

    for i, image_path in enumerate(image_path_list):

        img_result = encode_image(image_path)
        image_base64 = img_result['base64']
        width = img_result['width']
        height = img_result['height']
        question_with_options = question

        if messages is None:

            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "<image_clue_0>"}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}] + [{"type": "text", "text": "</image_clue_0>\n\n"}]
                }
            ]

            image_information += f"width of image_clue_0: {width}, height of image_clue_0: {height}\n"

        else:
            image_information += f"width of image_clue_{i}: {width}, height of image_clue_{i}: {height}\n"
            messages[0]['content'] += [{"type": "text", "text": f"<image_clue_{i}>"}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}] + [{"type": "text", "text": f"</image_clue_{i}>\n\n"}]

    messages[0]['content'] += [{"type": "text", "text": prompt_prefix.format(query=question_with_options, image_information=image_information)}]

    return messages

def update_messages_with_excu_content(image_nums_in_input, messages, images_result, text_result, error_result, image_clue_idx):
    if error_result is None:
        new_messages = []
        image_content = []
        for message_item in messages[:-1]:
            new_messages.append(message_item)

        assistant_message_item = messages[-1]['content']
        interpreter_message_text_prefix = [{"type": "text", "text": f"<interpreter>\nText Result:\n{text_result}\nImage Result:\n"}]
        if images_result is not None:
            for image_base64_item in images_result[image_clue_idx-image_nums_in_input:]:
                interpreter_message_images = [{"type": "text", "text": f"<image_clue_{image_clue_idx}>"}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_item}"}}] + [{"type": "text", "text": f"</image_clue_{image_clue_idx}>"}]
                image_content += interpreter_message_images
                image_clue_idx += 1
        else:
            image_content = [{"type": "text", "text": "None"}]
        interpreter_message_text_profill = [{"type": "text", "text": "</interpreter>\n"}]

        assistant_message_item = assistant_message_item + interpreter_message_text_prefix + image_content + interpreter_message_text_profill
        new_messages.append({"role": "assistant", "content": assistant_message_item})
    else:
        new_messages = []
        for message_item in messages[:-1]:
            new_messages.append(message_item)
    
        assistant_message_item = messages[-1]['content']
        interpreter_message_text_prefix = [{"type": "text", "text": f"<interpreter>{error_result}"}]
        interpreter_message_text_profill = [{"type": "text", "text": "</interpreter>\n"}]
    
        assistant_message_item = assistant_message_item + interpreter_message_text_prefix + interpreter_message_text_profill
        new_messages.append({"role": "assistant", "content": assistant_message_item})

    return new_messages, image_clue_idx

def update_messages_with_code(messages, generated_content):
    message_item = {
        "role": "assistant",
        "content": [{"type": "text", "text": f"{generated_content}</code>\n"}]
    }

    messages.append(message_item)
    return messages

def update_messages_with_text(messages, generated_content):
    message_item = {
        "role": "assistant",
        "content": [{"type": "text", "text": f"{generated_content}"}]
    }

    messages.append(message_item)
    return messages

def call_chatgpt_api(messages, client, max_tokens=10000, stop=None, temperature=0.6):
    """Call ChatGPT API with the given messages"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",  # 使用支持视觉的模型
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            stop=stop
        )
        
        response_text = response.choices[0].message.content
        
        # 检查是否遇到停止标记
        stop_reason = None
        if stop and any(s in response_text for s in stop):
            for s in stop:
                if s in response_text:
                    stop_reason = s
                    break
        else:
            stop_reason = response.choices[0].finish_reason

        if "<code>" in response_text:
            stop_reason = "</code>"
        
        return response_text, stop_reason
    
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None, None

def evaluate_single_data(args, data, client, executor):

    try:
        prompt_template = args.prompt_template
        prompt = args.prompt
        exe_code = args.exe_code
        max_tokens = args.max_tokens
        temperature = args.temperature
    except:
        prompt_template = args['prompt_template']
        prompt = args['prompt']
        exe_code = args['exe_code']
        max_tokens = args['max_tokens']
        temperature = args['temperature']

    messages = process_prompt_init(data["question"], data['image_path'], prompt_template, prompt)
    
    # 生成初始响应
    response_text, pred_stop_reason = call_chatgpt_api(
        messages, 
        client,
        max_tokens=max_tokens,
        stop=["</code>"] if exe_code else None,
        temperature=temperature
    )
    
    # 处理响应
    final_response = response_text
    code_execution_count = 0
    image_clue_idx = 1
    
    while True:
        # 检查是否需要执行代码
        if exe_code and pred_stop_reason == "</code>":
            # 提取要执行的代码
            messages = update_messages_with_code(messages, response_text)
            code_to_execute = response_text.split("```python")[-1].split("```")[0].strip()
            
            # 执行代码
            exe_result = excute_codes([code_to_execute], messages, executor)[0][0]
            if exe_result is None:
                text_result = "None"
                images_result = None
            else:
                output, report = exe_result
                if report == "Done":
                    error_result = None
                    try:
                        text_result = exe_result[0]['text']
                    except:
                        text_result = None
                        print("text result is none.")
                    try:
                        images_result = exe_result[0]['images']
                    except:
                        images_result = None
                        print("image result is none.")
                else:
                    error_result = report
                    text_result = None
                    images_result = None

            messages, new_image_clue_idx = update_messages_with_excu_content(1, messages, images_result, text_result, error_result, image_clue_idx)
            image_clue_idx = new_image_clue_idx
            
            code_execution_count += 1
            
            # 生成下一部分响应
            response_text, pred_stop_reason = call_chatgpt_api(
                messages, 
                client,
                max_tokens=max_tokens,
                stop=["</code>"] if exe_code else None,
                temperature=temperature
            )

        else:
            final_response = response_text
            messages = update_messages_with_text(messages, response_text)
            break
       
    return messages, final_response


def evaluate_single_data_multi_images(args, data, client, executor):

    try:
        prompt_template = args.prompt_template
        prompt = args.prompt
        exe_code = args.exe_code
        max_tokens = args.max_tokens
    except:
        prompt_template = args['prompt_template']
        prompt = args['prompt']
        exe_code = args['exe_code']
        max_tokens = args['max_tokens']

    messages = process_prompt_init_multi_images(data["question"], data['image_path_list'], prompt_template, prompt)
    
    # 生成初始响应
    response_text, pred_stop_reason = call_chatgpt_api(
        messages, 
        client,
        max_tokens=max_tokens,
        stop=["</code>"] if exe_code else None
    )
    
    # 处理响应
    final_response = response_text
    code_execution_count = 0
    image_clue_idx = data['image_nums_in_input']
    
    while True:
        # 检查是否需要执行代码
        if exe_code and pred_stop_reason == "</code>":
            # 提取要执行的代码
            messages = update_messages_with_code(messages, response_text)
            code_to_execute = response_text.split("```python")[-1].split("```")[0].strip()
            
            # 执行代码
            exe_result = excute_codes([code_to_execute], messages, executor)[0][0]
            if exe_result is None:
                text_result = "None"
                images_result = None
            else:
                output, report = exe_result
                if report == "Done":
                    error_result = None
                    try:
                        text_result = exe_result[0]['text']
                    except:
                        text_result = None
                        print("text result is none.")
                    try:
                        images_result = exe_result[0]['images']
                    except:
                        images_result = None
                        print("image result is none.")
                else:
                    error_result = report
                    text_result = None
                    images_result = None

            messages, new_image_clue_idx = update_messages_with_excu_content(data['image_nums_in_input'], messages, images_result, text_result, error_result, image_clue_idx)
            image_clue_idx = new_image_clue_idx
            
            code_execution_count += 1
            
            # 生成下一部分响应
            response_text, pred_stop_reason = call_chatgpt_api(
                messages, 
                client,
                max_tokens=max_tokens,
                stop=["</code>"] if exe_code else None
            )

        else:
            final_response = response_text
            messages = update_messages_with_text(messages, response_text)
            break
       
    return messages, final_response

def evaluate_single_data_video(args, data, client, executor):

    try:
        prompt_template = args.prompt_template
        prompt = args.prompt
        exe_code = args.exe_code
        max_tokens = args.max_tokens
    except:
        prompt_template = args['prompt_template']
        prompt = args['prompt']
        exe_code = args['exe_code']
        max_tokens = args['max_tokens']

    messages = process_prompt_init_multi_images(data["question"], data['image_path_list'], prompt_template, prompt)
    
    # 生成初始响应
    response_text, pred_stop_reason = call_chatgpt_api(
        messages, 
        client,
        max_tokens=max_tokens,
        stop=["</code>"] if exe_code else None
    )
    
    # 处理响应
    final_response = response_text
    code_execution_count = 0
    image_clue_idx = data['image_nums_in_input']
    
    while True:
        # 检查是否需要执行代码
        if exe_code and pred_stop_reason == "</code>":
            # 提取要执行的代码
            messages = update_messages_with_code(messages, response_text)
            code_to_execute = response_text.split("```python")[-1].split("```")[0].strip()
            
            # 执行代码
            exe_result = excute_codes([code_to_execute], messages, executor)[0][0]
            if exe_result is None:
                text_result = "None"
                images_result = None
            else:
                output, report = exe_result
                if report == "Done":
                    error_result = None
                    try:
                        text_result = exe_result[0]['text']
                    except:
                        text_result = None
                        print("text result is none.")
                    try:
                        images_result = exe_result[0]['images']
                    except:
                        images_result = None
                        print("image result is none.")
                else:
                    error_result = report
                    text_result = None
                    images_result = None

            messages, new_image_clue_idx = update_messages_with_excu_content(data['image_nums_in_input'], messages, images_result, text_result, error_result, image_clue_idx)
            image_clue_idx = new_image_clue_idx
            
            code_execution_count += 1
            
            # 生成下一部分响应
            response_text, pred_stop_reason = call_chatgpt_api(
                messages, 
                client,
                max_tokens=max_tokens,
                stop=["</code>"] if exe_code else None
            )

        else:
            final_response = response_text
            messages = update_messages_with_text(messages, response_text)
            break
       
    return messages, final_response