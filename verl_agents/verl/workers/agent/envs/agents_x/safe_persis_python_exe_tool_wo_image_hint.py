import base64
import copy
import io
import regex
import pickle
import traceback
import multiprocessing
from multiprocessing import Queue, Process
from typing import Any, Dict, Optional, Tuple, List, Union
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from verl.workers.agent.tool_envs import ToolBase
from contextlib import redirect_stdout
import threading
import queue
import time

class PersistentWorker:
    """持久化的工作进程"""
    
    def __init__(self):
        self.input_queue = multiprocessing.Queue()
        self.output_queue = multiprocessing.Queue()
        self.process = None
        self.start()
    
    def start(self):
        """启动工作进程"""
        self.process = Process(target=self._worker_loop)
        self.process.daemon = True
        self.process.start()
    
    def _worker_loop(self):
        """工作进程主循环"""
        runtime = None
        
        while True:
            try:
                # 获取任务
                task = self.input_queue.get()
                
                if task is None:  # 终止信号
                    break
                
                task_type = task.get('type')
                
                if task_type == 'init':
                    # 初始化Runtime
                    messages = task.get('messages', [])
                    runtime = SafeImageRuntime(messages)
                    self.output_queue.put({
                        'status': 'success',
                        'result': 'Initialized'
                    })
                
                elif task_type == 'execute':
                    # 执行代码
                    if runtime is None:
                        runtime = SafeImageRuntime(task.get('messages', []))
                    
                    code = task.get('code')
                    program_io = io.StringIO()
                    
                    try:
                        # 记录执行前的图像数量
                        pre_figures_count = len(runtime.captured_figures)
                        
                        with redirect_stdout(program_io):
                            runtime.exec_code(code)
                        
                        program_io.seek(0)
                        stdout_output = program_io.read()
                        
                        # 只获取新生成的图像
                        if pre_figures_count == len(runtime.captured_figures):
                            self.output_queue.put({
                                'status': 'success',
                                'result': {
                                    'text': stdout_output.strip(),
                                    'total_figures': len(runtime.captured_figures)
                                }
                            })

                        else:
                            new_figures = runtime.captured_figures[pre_figures_count:]
                            
                            self.output_queue.put({
                                'status': 'success',
                                'result': {
                                    'text': stdout_output.strip(),
                                    'images': new_figures,
                                    'total_figures': len(runtime.captured_figures)
                                }
                            })
                    
                    except Exception as e:
                        self.output_queue.put({
                            'status': 'error',
                            'error': str(e),
                            'traceback': traceback.format_exc()
                        })
                
                elif task_type == 'reset':
                    # 重置Runtime
                    messages = task.get('messages', [])
                    runtime = SafeImageRuntime(messages)
                    self.output_queue.put({
                        'status': 'success',
                        'result': 'Reset'
                    })
                    
            except Exception as e:
                self.output_queue.put({
                    'status': 'error',
                    'error': f'Worker error: {str(e)}',
                    'traceback': traceback.format_exc()
                })
    
    def execute(self, code: str, messages: list = None, timeout: int = 30):
        """执行代码"""
        self.input_queue.put({
            'type': 'execute',
            'code': code,
            'messages': messages
        })
        
        try:
            result = self.output_queue.get(timeout=timeout)
            return result
        except queue.Empty:
            return {
                'status': 'error',
                'error': 'Execution timeout'
            }
    
    def init_runtime(self, messages: list):
        """初始化Runtime"""
        self.input_queue.put({
            'type': 'init',
            'messages': messages
        })
        return self.output_queue.get()
    
    def reset_runtime(self, messages: list = None):
        """重置Runtime"""
        self.input_queue.put({
            'type': 'reset',
            'messages': messages
        })
        return self.output_queue.get()
    
    def terminate(self):
        """终止工作进程"""
        if self.process and self.process.is_alive():
            self.input_queue.put(None)
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.terminate()

class SafeImageRuntime:
    """安全版本的Runtime，强制使用隔离环境"""

    HEADERS = [
        "import matplotlib",
        "matplotlib.use('Agg')",  # Use non-interactive backend
        "import matplotlib.pyplot as plt",
        "from PIL import Image",
        "import io",
        "import base64",
        "import numpy as np",
        "_captured_figures = []",  # Initialize image capture list
        # 添加 plt.show() 的替代函数
        """def _internal_capture_plt_figure():
    '''Capture current matplotlib figure and save to _captured_figures'''
    fig = plt.gcf()
    width_px = fig.get_figwidth() * fig.dpi
    height_px = fig.get_figheight() * fig.dpi
    aspect_ratio = max(width_px, height_px) / min(width_px, height_px)
    if aspect_ratio >= 200:
        raise RuntimeError(f"Image aspect ratio too extreme: {aspect_ratio:.2f} (must be < 200)")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    _captured_image = base64.b64encode(buf.read()).decode('utf-8')
    _captured_figures.append(_captured_image)
    plt.close()
""",
    ]

    def __init__(self, messages=None):
        import matplotlib
        matplotlib.use('Agg')
        
        self._global_vars = {
            "_captured_figures": [],
        }

        for c in self.HEADERS:
            exec(c, self._global_vars)

        if messages:
            image_var_dict = {}
            image_var_idx = 0
            init_captured_figures = []
            for i, message in enumerate(messages):
                for item in message.get('content', []):
                    if item.get('type') == "image_url":
                        img = base64_to_image(item['image_url']['url'])
                        if img:
                            image_var_dict[f"image_clue_{image_var_idx}"] = img
                            init_captured_figures.append(img)
                            image_var_idx += 1

                    elif item.get('type') == "image_hint_base64":
                        item_image_url = item['url']
                        image = base64_to_image(item_image_url)
                        if image:
                            image_var_dict[f"image_hint_{image_var_idx}"] = image
                            # init_captured_figures.append(base64.b64encode(
                            #     BytesIO(image.tobytes()).getvalue()).decode('utf-8'))
                            image_var_idx += 1

            image_var_dict[f"_captured_figures"] = init_captured_figures
            self._global_vars.update(image_var_dict)

    def exec_code(self, code: str) -> None:
        """执行代码并捕获图形"""
        if regex.search(r"(\s|^)?(input|os\.system|subprocess)\(", code):
            raise RuntimeError("Forbidden function calls detected")

        modified_code = code.replace("plt.show()", "_internal_capture_plt_figure()")
        
        try:
            exec(modified_code, self._global_vars)
        except Exception as e:
            plt.close('all')
            raise e
    
    @property
    def captured_figures(self):
        return self._global_vars.get("_captured_figures", [])

class MultiModalPythonTool_wo_Image_Hint(ToolBase):
    name = "pyvision_gym_wo_image_hint"
    description = "Tool for executing Python code with multimodal capabilities"
    
    def __init__(self, _name=None, _desc=None, _params=None, **kwargs):
        super().__init__(name=self.name)
        self.chatml_history = []
        self.multi_modal_data = None
        self.use_process_isolation = True
        self.persistent_worker = None  # 持久化的工作进程
        self._figures_count = 0  # 跟踪图像数量
    
    def _ensure_worker(self):
        """确保工作进程存在"""
        if self.persistent_worker is None:
            self.persistent_worker = PersistentWorker()
    
    def extract_answer(self, action_string: str) -> str:
        """Extract answer from action string"""
        answer = regex.findall(r'<answer>(.*?)</answer>', action_string, regex.DOTALL)
        return answer[-1] if answer else None
        
    def extract_code(self, action_string: str) -> str:
        """Extract Python code from action string"""
        code_blocks = regex.findall(r'<code>\s*```python\s*(.*?)\s*```\s*</code>', action_string, regex.DOTALL)
        if code_blocks:
            return code_blocks[-1]
        
        code_blocks = regex.findall(r'```python\s*(.*?)\s*```', action_string, regex.DOTALL)
        return code_blocks[-1] if code_blocks else None
    
    def execute(self, action_string: str, **kwargs) -> tuple:
        """Execute Python code safely"""
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {"final_answer": answer}
        
        code = self.extract_code(action_string)
        if not code:
            error_msg = "No Python code or final answer found. There is something wrong with the format."
            obs = f"\n<|im_start|>observation\n<interpreter>Error: {error_msg}</interpreter>\n<|im_end|>\n<|im_start|>assistant\n"
            return obs, 0.0, False, {"error": error_msg}
        
        try:
            messages = self._convert_to_messages(self.multi_modal_data)
            
            if self.use_process_isolation:
                # 确保工作进程存在
                self._ensure_worker()
                
                # 执行代码
                result = self.persistent_worker.execute(code, messages)
                
                if result['status'] == 'success':
                    exec_result = result['result']
                    obs_content = exec_result.get('text', 'None')
                    
                    if 'images' in exec_result and exec_result['images']:
                        images = [self._base64_to_image(img) for img in exec_result['images']]
                        if None in images:
                            error_msg = "Something wrong with processed images."
                            obs = f"\n<|im_start|>observation\n<interpreter>Error: {error_msg}</interpreter>\n<|im_end|>\n<|im_start|>assistant\n"
                            return obs, 0.0, False, {"error": error_msg}
                        image_content = []
                        image_clue_idx = self._figures_count
                        
                        for _ in range(len(images)):
                            interpreter_message_images = [
                                {"type": "text", "text": f"<image_clue_{image_clue_idx}>"},
                                {"type": "text", "text": "<image>"},
                                {"type": "text", "text": f"</image_clue_{image_clue_idx}>"}
                            ]
                            image_content += interpreter_message_images
                            image_clue_idx += 1
                        
                        # 更新图像计数
                        self._figures_count = exec_result.get('total_figures', self._figures_count)

                        if obs_content is None:
                            content_prefix = [
                                {"type": "text", "text": "<interpreter>"},
                                {"type": "text", "text": "Image Result:\n"},
                            ]
                        else:
                            content_prefix = [
                                {"type": "text", "text": "<interpreter>"},
                                {"type": "text", "text": f"Text Result:\n{obs_content}\n"},
                                {"type": "text", "text": "Image Result:\n"},
                            ]

                        content_suffix = [
                            {"type": "text", "text": "</interpreter>\n"}
                        ]
                        
                        obs_chat = [{
                            "role": "observation",
                            "content": content_prefix + image_content + content_suffix
                        }]
                        
                        obs = {
                            "prompt": "",
                            "chat": obs_chat,
                            "multi_modal_data": {"image": [img for img in images if img]}
                        }
                    else:
                        obs = f"\n<|im_start|>observation\n<interpreter>Text Result:\n{obs_content}</interpreter>\n<|im_end|>\n<|im_start|>assistant\n"
                    
                    return obs, 0.1, False, {"status": "success"}
                else:
                    error_msg = f"Execution error: {result.get('error', 'Unknown error')}"
                    obs = f"\n<|im_start|>observation\n<interpreter>{error_msg}</interpreter>\n<|im_end|>\n<|im_start|>assistant\n"
                    return obs, 0.1, False, {"error": error_msg}
            else:
                # 非隔离模式（调试用）
                runtime = SafeImageRuntime(messages)
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    runtime.exec_code(code)
                program_io.seek(0)
                result = {
                    'text': program_io.read().strip(),
                    'images': runtime.captured_figures
                }
                obs = f"\n<|im_start|>observation\n<interpreter>Text Result:\n{result['text']}</interpreter><|im_end|>\n<|im_start|>assistant\n"
                return obs, 0.1, False, {"status": "success"}
                
        except Exception as e:
            error_msg = f"Tool error: {str(e)}"
            obs = f"\n<|im_start|>observation\n<interpreter>{error_msg}</interpreter><|im_end|>\n<|im_start|>assistant\n"
            return obs, 0.0, False, {"error": str(e)}
    
    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        """Reset tool state"""
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        self._figures_count = 1
        
        # 重置持久化工作进程的状态
        if self.persistent_worker:
            messages = self._convert_to_messages_wo_image_hint(origin_multi_modal_data)
            self.persistent_worker.reset_runtime(messages)
    
    def __del__(self):
        """清理资源"""
        if self.persistent_worker:
            self.persistent_worker.terminate()
    
    def _convert_to_messages(self, multi_modal_data):
        """Convert multi_modal_data to messages format"""
        if not multi_modal_data or 'image' not in multi_modal_data:
            return []
        
        messages = [{
            "role": "user",
            "content": []
        }]
        
        for i, img in enumerate(multi_modal_data['image']):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })
        
        return messages

    def _convert_to_messages_wo_image_hint(self, multi_modal_data):
        """Convert multi_modal_data to messages format"""
        if not multi_modal_data or 'image' not in multi_modal_data:
            return []
        
        messages = [{
            "role": "user",
            "content": []
        }]
        
        for i, img in enumerate(multi_modal_data['image']):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            messages[0]["content"].append({
                "type": "image_hint_base64",
                "url": f"data:image/png;base64,{img_base64}"
            })
        
        return messages
    
    def _base64_to_image(self, base64_str):
        """Convert base64 string to PIL Image"""
        try:
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            return Image.open(io.BytesIO(base64.b64decode(base64_str)))
        except Exception:
            return None

def base64_to_image(base64_str: str, remove_prefix: bool = True) -> Union[Image.Image, None]:
    """Convert Base64 string to PIL Image"""
    try:
        if remove_prefix and "," in base64_str:
            base64_str = base64_str.split(",")[1]
        return Image.open(io.BytesIO(base64.b64decode(base64_str)))
    except Exception as e:
        print(f"Base64 decode error: {str(e)}")
        return None