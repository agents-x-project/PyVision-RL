import base64
import copy
import io
import regex
import pickle
import traceback
import multiprocessing
from multiprocessing import Queue
from typing import Any, Dict, Optional, Tuple, List, Union
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from verl.workers.agent.tool_envs import ToolBase
from contextlib import redirect_stdout

class SafeImageRuntime:
    """安全版本的Runtime，强制使用隔离环境"""
    def __init__(self, messages=None):
        # 必须在所有plt导入前设置后端
        import matplotlib
        matplotlib.use('Agg')
        
        self._global_vars = {
            "__builtins__": copy.copy(__builtins__),
            "_captured_figures": [],
            "plt": plt,
            "np": np,
            "Image": Image,
            "io": io,
            "base64": base64
        }
        
        # 初始化图像变量
        if messages:
            image_var_dict = {}
            for i, message in enumerate(messages):
                for item in message.get('content', []):
                    if item.get('type') == "image_url":
                        img = base64_to_image(item['image_url']['url'])
                        if img:
                            image_var_dict[f"image_clue_{i}"] = img
            self._global_vars.update(image_var_dict)
    
    def exec_code(self, code: str) -> None:
        """执行代码并捕获图形"""
        # 安全检查
        if regex.search(r"(\s|^)?(input|os\.system|subprocess)\(", code):
            raise RuntimeError("Forbidden function calls detected")
        
        # 替换plt.show()调用
        modified_code = code.replace("plt.show()", """
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
_captured_image = base64.b64encode(buf.read()).decode('utf-8')
_captured_figures.append(_captured_image)
plt.close()
""")
        try:
            exec(modified_code, self._global_vars)
        except Exception as e:
            plt.close('all')
            raise e
    
    @property
    def captured_figures(self):
        return self._global_vars.get("_captured_figures", [])

class MultiModalPythonTool(ToolBase):
    name = "multi_modal_python_tool"
    description = "Tool for executing Python code with multimodal capabilities"
    
    def __init__(self, _name=None, _desc=None, _params=None, **kwargs):
        super().__init__(name=self.name)
        self.chatml_history = []
        self.multi_modal_data = None
        self.use_process_isolation = True  # 默认启用进程隔离
    
    def _execute_in_process(self, code: str, messages: list) -> Tuple[Any, str]:
        """在子进程中安全执行代码"""
        def worker(code: str, messages: list, queue: Queue):
            try:
                runtime = SafeImageRuntime(messages)
                program_io = io.StringIO()
                
                with redirect_stdout(program_io):
                    runtime.exec_code(code)
                
                program_io.seek(0)
                stdout_output = program_io.read()
                
                result = {
                    'text': stdout_output.strip(),
                    'images': runtime.captured_figures
                }
                queue.put((result, "Done"))
            except Exception as e:
                queue.put((
                    {
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    },
                    f"Error: {str(e)}"
                ))
        
        queue = Queue()
        p = multiprocessing.Process(target=worker, args=(code, messages, queue))
        p.start()
        result, report = queue.get()
        p.join(timeout=30)  # 30秒超时
        
        if p.is_alive():
            p.terminate()
            return {"error": "Timeout"}, "Timeout Error"
        return result, report

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
            error_msg = "No Python code found."
            obs = f"\n<|im_start|>user\n<interpreter>Error: {error_msg}</interpreter><|im_end|>\n<|im_start|>assistant\n"
            return obs, 0.0, False, {"error": error_msg}
        
        try:
            messages = self._convert_to_messages(self.multi_modal_data)
            
            if self.use_process_isolation:
                result, report = self._execute_in_process(code, messages)
            else:
                # 调试模式（不推荐生产使用）
                runtime = SafeImageRuntime(messages)
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    runtime.exec_code(code)
                program_io.seek(0)
                result = {
                    'text': program_io.read().strip(),
                    'images': runtime.captured_figures
                }
                report = "Done"
            
            if report == "Done":
                obs_content = result.get('text', '')
                
                if 'images' in result and result['images']:
                    images = [self._base64_to_image(img) for img in result['images']]
                    obs = {
                        "prompt": f"\n<|im_start|>user\n<interpreter>{obs_content}</interpreter><|im_end|>\n<|im_start|>assistant\n",
                        "multi_modal_data": {"image": [img for img in images if img]}
                else:
                    obs = f"\n<|im_start|>user\n<interpreter>{obs_content}</interpreter><|im_end|>\n<|im_start|>assistant\n"
                
                return obs, 0.1, False, {"status": "success"}
            else:
                error_msg = f"Execution error: {report}"
                obs = f"\n<|im_start|>user\n<interpreter>{error_msg}</interpreter><|im_end|>\n<|im_start|>assistant\n"
                return obs, 0.0, False, {"error": error_msg}
                
        except Exception as e:
            error_msg = f"Tool error: {str(e)}"
            obs = f"\n<|im_start|>user\n<interpreter>{error_msg}</interpreter><|im_end|>\n<|im_start|>assistant\n"
            return obs, 0.0, False, {"error": str(e)}
    
    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        """Reset tool state"""
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
    
    def _convert_to_messages(self, multi_modal_data):
        """Convert multi_modal_data to messages format"""
        if not multi_modal_data or 'image' not in multi_modal_data:
            return []
        
        messages = [{
            "role": "user",
            "content": []
        }]
        
        # 添加图像
        for i, img in enumerate(multi_modal_data['image']):
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
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