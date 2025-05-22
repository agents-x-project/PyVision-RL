import os
import io
import regex
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
import multiprocess
from multiprocess import Pool
from typing import Any, Dict, Optional, Tuple, List, Union
from pebble import ProcessPool
from tqdm import tqdm
from concurrent.futures import TimeoutError
from functools import partial
from timeout_decorator import timeout
from contextlib import redirect_stdout
import base64
from io import BytesIO
from PIL import Image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def base64_to_image(
    base64_str: str, 
    remove_prefix: bool = True, 
    convert_mode: Optional[str] = "RGB"
) -> Union[Image.Image, None]:
    """
    将Base64编码的图片字符串转换为PIL Image对象
    
    Args:
        base64_str: Base64编码的图片字符串（可带data:前缀）
        remove_prefix: 是否自动去除"data:image/..."前缀（默认True）
        convert_mode: 转换为指定模式（如"RGB"/"RGBA"，None表示不转换）
    
    Returns:
        PIL.Image.Image 对象，解码失败时返回None
        
    Examples:
        >>> img = base64_to_image("data:image/png;base64,iVBORw0KGg...")
        >>> img = base64_to_image("iVBORw0KGg...", remove_prefix=False)
    """
    try:
        # 1. 处理Base64前缀
        if remove_prefix and "," in base64_str:
            base64_str = base64_str.split(",")[1]

        # 2. 解码Base64
        image_data = base64.b64decode(base64_str)
        
        # 3. 转换为PIL Image
        image = Image.open(BytesIO(image_data))
        
        # 4. 可选模式转换
        if convert_mode:
            image = image.convert(convert_mode)
            
        return image
    
    except (base64.binascii.Error, OSError, Exception) as e:
        print(f"Base64解码失败: {str(e)}")
        return None


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None
        self._captured_figures = []

        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        if regex.search(r"(\s|^)?input\(", code_piece) or regex.search(
            r"(\s|^)?os.system\(", code_piece
        ):
            raise RuntimeError("Forbidden function calls detected")

        
        
        # 检测并修改plt.show()调用
        if "plt.show()" in code_piece:
            modified_code = code_piece.replace("plt.show()", """
# 捕获当前图像
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
_captured_image = base64.b64encode(buf.read()).decode('utf-8')
_captured_figures.append(_captured_image)
plt.close()
""")
            # 确保_captured_figures变量存在
            if "_captured_figures" not in self._global_vars:
                self._global_vars["_captured_figures"] = []
            
            exec(modified_code, self._global_vars)
        else:
            exec(code_piece, self._global_vars)

    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)

    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v

    @property
    def answer(self):
        return self._global_vars.get("answer", None)
    
    @property
    def captured_figures(self):
        return self._global_vars.get("_captured_figures", [])


class ImageRuntime(GenericRuntime):
    # """支持图像处理的运行时环境"""
    # GLOBAL_DICT = {}  # 不预加载模块，避免序列化问题
    # LOCAL_DICT = None
    
    HEADERS = [
        "import matplotlib",
        "matplotlib.use('Agg')",  # 使用非交互式后端
        "import matplotlib.pyplot as plt",
        "from PIL import Image",
        "import io",
        "import base64",
        "import numpy as np",
        "_captured_figures = []",  # 初始化图像捕获列表
    ]

    def __init__(self, messages):
        super().__init__()

        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None
        self._captured_figures = []

        for c in self.HEADERS:
            self.exec_code(c)

        image_var_dict = {}
        image_var_idx = 0
        for message_item in messages:
            content = message_item['content']  # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            for item in content:
                item_type = item['type']
                if item_type == "image_url":
                    item_image_url = item['image_url']['url']
                    image = base64_to_image(item_image_url)
                    image_var_dict[f"image_clue_{image_var_idx}"] = image
                    image_var_idx += 1

        self.inject(image_var_dict)
                    

class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {}
    HEADERS = [
        "import datetime",
        "from dateutil.relativedelta import relativedelta",
        "timedelta = relativedelta"
    ]


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()


class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {"dict": CustomDict}


class PythonExecutor:
    def __init__(
        self,
        runtime_class=None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = True,
        timeout_length: int = 20,
    ) -> None:
        self.runtime_class = runtime_class if runtime_class else ImageRuntime
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.timeout_length = timeout_length

        # Create a persistent runtime instance if messages are provided
        self.persistent_runtime = None

    def process_generation_to_code(self, gens: str):
        return [g.split("\n") for g in gens]

    # def execute(
    #     self,
    #     code,
    #     messages,
    #     get_answer_from_stdout=True,
    #     runtime_class=None,
    #     # run_time_instance=None,
    #     answer_symbol=None,
    #     answer_expr=None,
    #     # 移除 timeout_length 参数
    # ) -> Tuple[Union[str, Dict[str, Any]], str]:
    #     # print("dome")
    #     # try:
    #     # 在每个进程中创建新的运行时实例
    #     # runtime = runtime_class(messages)
    #     runtime = self.persistent_runtime
        
    #     if get_answer_from_stdout:
    #         program_io = io.StringIO()
    #         with redirect_stdout(program_io):
    #             # 移除 timeout 调用
    #             runtime.exec_code("\n".join(code))
    #         program_io.seek(0)
    #         result = program_io.read()
    #     elif answer_symbol:
    #         # 移除 timeout 调用
    #         runtime.exec_code("\n".join(code))
    #         result = runtime._global_vars.get(answer_symbol, "")
    #     elif answer_expr:
    #         # 移除 timeout 调用
    #         runtime.exec_code("\n".join(code))
    #         # 移除 timeout 调用
    #         result = runtime.eval_code(answer_expr)
    #     else:
    #         if len(code) > 1:
    #             # 移除 timeout 调用
    #             runtime.exec_code("\n".join(code[:-1]))
    #             # 移除 timeout 调用
    #             result = runtime.eval_code(code[-1])
    #         else:
    #             # 移除 timeout 调用
    #             runtime.exec_code("\n".join(code))
    #             result = ""
        
    #     # 检查是否有捕获的图像
    #     captured_figures = runtime._global_vars.get("_captured_figures", [])
    #     if captured_figures:
    #         # 如果有文本输出和图像，将它们组合
    #         if result:
    #             result = {
    #                 'text': result,
    #                 'images': captured_figures
    #             }
    #         else:
    #             result = {'images': captured_figures}
    #     else:
    #         if result:
    #             result = {
    #                 'text': result,
    #             }
        
    #     report = "Done"
        
    #     # 确保结果可序列化
    #     try:
    #         pickle.dumps(result)
    #     except Exception as e:
    #         result = f"Result serialization error: {str(e)}"
    #         report = f"Serialization Error: {str(e)}"
            
    #     return result, report

    def execute(
        self,
        code,
        messages,
        get_answer_from_stdout=True,
        runtime_class=None,
        answer_symbol=None,
        answer_expr=None,
    ) -> Tuple[Union[str, Dict[str, Any]], str]:
        print(runtime_class)
        runtime = self.persistent_runtime
    
        try:
            if get_answer_from_stdout:
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    runtime.exec_code("\n".join(code))
                program_io.seek(0)
                result = program_io.read()
            elif answer_symbol:
                runtime.exec_code("\n".join(code))
                result = runtime._global_vars.get(answer_symbol, "")
            elif answer_expr:
                runtime.exec_code("\n".join(code))
                result = runtime.eval_code(answer_expr)
            else:
                if len(code) > 1:
                    runtime.exec_code("\n".join(code[:-1]))
                    result = runtime.eval_code(code[-1])
                else:
                    runtime.exec_code("\n".join(code))
                    result = ""
    
            # Check for captured figures
            captured_figures = runtime._global_vars.get("_captured_figures", [])
            if captured_figures:
                result = {
                    'text': result,
                    'images': captured_figures
                } if result else {'images': captured_figures}
            else:
                result = {'text': result} if result else {}
    
            report = "Done"
    
        except Exception as e:
            result = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            # report = f"Error: {str(e)}\n{traceback.format_exc()}"
            report = f"Error: {str(e)}"
    
        # Ensure result is serializable
        try:
            pickle.dumps(result)
        except Exception as e:
            result = f"Result serialization error: {str(e)}"
            report = f"Serialization Error: {str(e)}"
    
        return result, report


    def apply(self, code, messages):
        return self.batch_apply([code], messages)[0]

    @staticmethod
    def truncate(s, max_length=400):
        if isinstance(s, dict):
            # 如果是字典（包含图像），只截断文本部分
            if 'text' in s:
                half = max_length // 2
                if len(s['text']) > max_length:
                    s['text'] = s['text'][:half] + "..." + s['text'][-half:]
            return s
        else:
            half = max_length // 2
            if isinstance(s, str) and len(s) > max_length:
                s = s[:half] + "..." + s[-half:]
            return s

    def batch_apply(self, batch_code, messages):
        if not self.persistent_runtime and messages:
            self.persistent_runtime = self.runtime_class(messages)
        all_code_snippets = self.process_generation_to_code(batch_code)

        timeout_cnt = 0
        all_exec_results = []

        # 去掉 ProcessPool，改为单进程顺序执行
        if len(all_code_snippets) > 100:
            progress_bar = tqdm(total=len(all_code_snippets), desc="Execute")
        else:
            progress_bar = None

        for code in all_code_snippets:
            try:
                # 直接调用 self.execute，而不是用 ProcessPool
                result = self.execute(
                    code,
                    messages=messages,
                    get_answer_from_stdout=self.get_answer_from_stdout,
                    runtime_class=self.runtime_class,
                    answer_symbol=self.answer_symbol,
                    answer_expr=self.answer_expr,
                    # timeout_length=self.timeout_length,
                )
                all_exec_results.append(result)
            except TimeoutError as error:
                print(error)
                all_exec_results.append(("", "Timeout Error"))
                timeout_cnt += 1
            except Exception as error:
                print(f"Error in batch_apply: {error}")
                all_exec_results.append(("", f"Error: {str(error)}"))
            
            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            # 处理结果
            if isinstance(res, dict):
                # 如果结果包含图像，特殊处理
                if 'text' in res:
                    res['text'] = str(res['text']).strip()
                    res['text'] = self.truncate(res['text'])
                report = str(report).strip()
                report = self.truncate(report)
            else:
                # 普通文本结果
                res = str(res).strip()
                res = self.truncate(res)
                report = str(report).strip()
                report = self.truncate(report)
            batch_results.append((res, report))
        return batch_results

    # def release_runtime(self):
    #     """
    #     释放持久化运行时环境，清理相关资源。
    #     在不再需要当前运行时环境时调用此方法，可以释放内存并重置状态。
        
    #     Returns:
    #         bool: 如果成功释放返回True，如果没有运行时环境需要释放返回False
    #     """
    #     if self.persistent_runtime is not None:
    #         # 清理可能的大型对象引用
    #         if hasattr(self.persistent_runtime, '_global_vars'):
    #             # 清理全局变量字典中的大型对象
    #             keys_to_clear = []
    #             for key, value in self.persistent_runtime._global_vars.items():
    #                 # 避免清理内置模块和函数
    #                 if not key.startswith('__') and not callable(value):
    #                     keys_to_clear.append(key)
                
    #             for key in keys_to_clear:
    #                 self.persistent_runtime._global_vars[key] = None
            
    #         # 清理捕获的图像
    #         if hasattr(self.persistent_runtime, '_captured_figures'):
    #             self.persistent_runtime._captured_figures.clear()
            
    #         # 将运行时实例设置为None
    #         self.persistent_runtime = None
    #         return True
    #     return False



def _test():
    image_path = "/mnt/petrelfs/zhaoshitian/vis_tool_inference_engine/test_data/0.JPG"
    image_base64 = encode_image(image_path)
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "From the information on that advertising board, what is the type of this shop?"}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "image_clue_0"}] + [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]
        }
    ]
    # 测试普通计算
    math_code ="""
a = 1
b = 2
c = a + b
print(c)
"""

    batch_code = [math_code]

    executor = PythonExecutor()
    predictions = executor.apply(batch_code[0], messages)
    print("数学计算结果:", predictions)
    
    # 测试图像显示
    image_code = """
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

# 创建一个简单的图像
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'r-', linewidth=2)
plt.title('Sine Wave')
plt.grid(True)
plt.show()

# 也可以显示一个简单的图像
# 创建一个彩色渐变图像
arr = np.zeros((100, 100, 3), dtype=np.uint8)
for i in range(100):
    for j in range(100):
        arr[i, j, 0] = i  # 红色通道
        arr[i, j, 1] = j  # 绿色通道
        arr[i, j, 2] = 100  # 蓝色通道

img = Image.fromarray(arr)
plt.figure()
plt.imshow(img)
plt.title('Gradient Image')
plt.show()

print("图像生成完成")
    """

    image_code = """
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io

plt.imshow(image_clue_0)
plt.title("Original Image - Locate Advertising Board")
plt.show()
    """
    
    image_result = executor.apply(image_code, messages)
    print("\n图像结果类型:", type(image_result[0]))
    if isinstance(image_result[0], dict) and 'images' in image_result[0]:
        print(f"捕获到 {len(image_result[0]['images'])} 个图像")
        print("第一个图像的base64编码前20个字符:", image_result[0]['images'][0][:20])
        
        # 可选：保存图像到文件
        for i, img_data in enumerate(image_result[0]['images']):
            img_bytes = base64.b64decode(img_data)
            with open(f"captured_image_{i}.png", "wb") as f:
                f.write(img_bytes)
            print(f"图像已保存为 captured_image_{i}.png")
            
        if 'text' in image_result[0]:
            print("文本输出:", image_result[0]['text'])
    else:
        print("未捕获到图像")
        print("结果:", image_result[0])
    
    print("\n执行状态:", image_result[1])


if __name__ == "__main__":
    _test()
