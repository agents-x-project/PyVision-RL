import base64
import copy
import io
import regex
import pickle
import traceback
from typing import Any, Dict, Optional, Tuple, List, Union
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from verl.workers.agent.tool_envs import ToolBase
from contextlib import redirect_stdout

class MultiModalPythonTool(ToolBase):
    name = "multi_modal_python_tool"
    description = "Tool for executing Python code with multimodal capabilities"
    
    def __init__(self, _name=None, _desc=None, _params=None, **kwargs):
        super().__init__(name=self.name)
        self.chatml_history = []
        self.multi_modal_data = None
        self.executor = PythonExecutor()
        
    def extract_answer(self, action_string: str) -> str:
        """Extract answer from action string"""
        answer = regex.findall(r'<answer>(.*?)</answer>', action_string, regex.DOTALL)
        return answer[-1] if answer else None
        
    def extract_code(self, action_string: str) -> str:
        """Extract Python code from action string with <code> tags"""
        # First try to extract from <code> tags
        code_blocks = regex.findall(r'<code>\s*```python\s*(.*?)\s*```\s*</code>', action_string, regex.DOTALL)
        if code_blocks:
            return code_blocks[-1]
        
        # Fallback to regular python code blocks
        code_blocks = regex.findall(r'```python\s*(.*?)\s*```', action_string, regex.DOTALL)
        return code_blocks[-1] if code_blocks else None
    
    def execute(self, action_string: str, **kwargs) -> tuple:
        """Execute Python code and return the results"""
        # Check if there's a final answer
        answer = self.extract_answer(action_string)
        if answer:
            return "", 0.0, True, {"final_answer": answer}
        
        code = self.extract_code(action_string)
        if not code:
            error_msg = "No Python code found. Please provide code between <code>```python and ```</code> tags."
            obs = f"\n<|im_start|>user\n<interpreter>Error: {error_msg}</interpreter><|im_end|>\n<|im_start|>assistant\n"
            return obs, 0.0, False, {"error": error_msg}
        
        try:
            # Convert the multi_modal_data to messages format expected by the executor
            messages = self._convert_to_messages(self.multi_modal_data)
            
            # Execute the code
            result, report = self.executor.execute(code.split('\n'), messages)
            
            if report == "Done":
                # Build response based on execution result
                interpreter_content = ""
                
                # Add text output if present
                if isinstance(result, dict):
                    if 'text' in result and result['text']:
                        interpreter_content += f"Text Result:\n{result['text']}\n"
                    
                    if 'images' in result and result['images']:
                        interpreter_content += f"Image Result:\n"
                        
                        # Create multi_modal_data with generated images
                        images = []
                        for img_base64 in result['images']:
                            img = self._base64_to_image(img_base64)
                            if img:
                                images.append(img)
                        
                        if images:
                            # Prepare observation with both text and images
                            obs = {
                                "prompt": f"\n<|im_start|>user\n<interpreter>{interpreter_content}</interpreter><|im_end|>\n<|im_start|>assistant\n",
                                "multi_modal_data": {"image": images}
                            }
                        else:
                            interpreter_content += "Generated images (displayed above)\n"
                            obs = f"\n<|im_start|>user\n<interpreter>{interpreter_content}</interpreter><|im_end|>\n<|im_start|>assistant\n"
                    else:
                        # Only text result
                        obs = f"\n<|im_start|>user\n<interpreter>{interpreter_content}</interpreter><|im_end|>\n<|im_start|>assistant\n"
                else:
                    # Handle simple string result
                    interpreter_content = str(result) if result else "Code executed successfully."
                    obs = f"\n<|im_start|>user\n<interpreter>{interpreter_content}</interpreter><|im_end|>\n<|im_start|>assistant\n"
                
                return obs, 0.1, False, {"status": "success"}
            else:
                # Return error message
                error_msg = f"Code execution error: {report}"
                obs = f"\n<|im_start|>user\n<interpreter>{error_msg}</interpreter><|im_end|>\n<|im_start|>assistant\n"
                return obs, 0.0, False, {"error": error_msg, "status": "failed"}
                
        except Exception as e:
            error_msg = f"Tool execution error: {str(e)}"
            obs = f"\n<|im_start|>user\n<interpreter>{error_msg}</interpreter><|im_end|>\n<|im_start|>assistant\n"
            return obs, 0.0, False, {"error": str(e), "status": "failed"}
    
    def reset(self, raw_prompt, multi_modal_data, origin_multi_modal_data, **kwargs):
        """Reset tool state for a new conversation"""
        self.chatml_history = raw_prompt
        self.multi_modal_data = origin_multi_modal_data
        
        # Reset the executor's persistent runtime
        if hasattr(self.executor, 'persistent_runtime'):
            self.executor.persistent_runtime = None
    
    def _convert_to_messages(self, multi_modal_data):
        """Convert multi_modal_data to messages format expected by PythonExecutor"""
        messages = []
        
        # Add a placeholder user message
        user_message = {
            "role": "user",
            "content": []
        }
        
        # Add text content
        user_message["content"].append({"type": "text", "text": "image_clue_0"})
        
        # Add image content if available
        if multi_modal_data and 'image' in multi_modal_data and multi_modal_data['image']:
            for i, img in enumerate(multi_modal_data['image']):
                # Convert PIL image to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                user_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })
        
        messages.append(user_message)
        return messages
    
    def _base64_to_image(self, base64_str, convert_mode="RGB"):
        """Convert base64 string to PIL Image"""
        try:
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            if convert_mode:
                image = image.convert(convert_mode)
            return image
        except Exception as e:
            print(f"Base64 decode error: {str(e)}")
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
        
        # Check and modify plt.show() calls
        if "plt.show()" in code_piece:
            modified_code = code_piece.replace("plt.show()", """
# Capture current figure
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
_captured_image = base64.b64encode(buf.read()).decode('utf-8')
_captured_figures.append(_captured_image)
plt.close()
""")
            # Ensure _captured_figures variable exists
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
    HEADERS = [
        "import matplotlib",
        "matplotlib.use('Agg')",  # Use non-interactive backend
        "import matplotlib.pyplot as plt",
        "from PIL import Image",
        "import io",
        "import base64",
        "import numpy as np",
        "_captured_figures = []",  # Initialize image capture list
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
            content = message_item['content']
            for item in content:
                item_type = item.get('type')
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


def base64_to_image(base64_str: str, remove_prefix: bool = True, convert_mode: Optional[str] = "RGB") -> Union[Image.Image, None]:
    """
    Convert Base64 encoded image string to PIL Image object
    
    Args:
        base64_str: Base64 encoded image string (may include data: prefix)
        remove_prefix: Whether to automatically remove "data:image/..." prefix (default True)
        convert_mode: Convert to specified mode (e.g. "RGB"/"RGBA", None means no conversion)
    
    Returns:
        PIL.Image.Image object, or None if decoding fails
    """
    try:
        # 1. Handle Base64 prefix
        if remove_prefix and "," in base64_str:
            base64_str = base64_str.split(",")[1]

        # 2. Decode Base64
        image_data = base64.b64decode(base64_str)
        
        # 3. Convert to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # 4. Optional mode conversion
        if convert_mode:
            image = image.convert(convert_mode)
            
        return image
    
    except (base64.binascii.Error, OSError, Exception) as e:
        print(f"Base64 decoding failed: {str(e)}")
        return None


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
        self.persistent_runtime = None

    def process_generation_to_code(self, gens: str):
        return [g.split("\n") for g in gens]

    def execute(
        self,
        code,
        messages,
        get_answer_from_stdout=True,
        runtime_class=None,
        answer_symbol=None,
        answer_expr=None,
    ) -> Tuple[Union[str, Dict[str, Any]], str]:
        # Create runtime instance if needed
        if not self.persistent_runtime:
            self.persistent_runtime = self.runtime_class(messages)
        
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
            report = f"Error: {str(e)}"
    
        # Ensure result is serializable
        try:
            pickle.dumps(result)
        except Exception as e:
            result = {'text': f"Result serialization error: {str(e)}"}
            report = f"Serialization Error: {str(e)}"
    
        return result, report

    def apply(self, code, messages):
        return self.batch_apply([code], messages)[0]

    @staticmethod
    def truncate(s, max_length=400):
        if isinstance(s, dict):
            # If it's a dictionary (with images), only truncate text part
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

        all_exec_results = []

        for code in all_code_snippets:
            try:
                result = self.execute(
                    code,
                    messages=messages,
                    get_answer_from_stdout=self.get_answer_from_stdout,
                    runtime_class=self.runtime_class,
                    answer_symbol=self.answer_symbol,
                    answer_expr=self.answer_expr,
                )
                all_exec_results.append(result)
            except Exception as error:
                print(f"Error in batch_apply: {error}")
                all_exec_results.append(("", f"Error: {str(error)}"))

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            # Process results
            if isinstance(res, dict):
                # Special handling for results with images
                if 'text' in res:
                    res['text'] = str(res['text']).strip()
                    res['text'] = self.truncate(res['text'])
                report = str(report).strip()
                report = self.truncate(report)
            else:
                # Regular text results
                res = str(res).strip()
                res = self.truncate(res)
                report = str(report).strip()
                report = self.truncate(report)
            batch_results.append((res, report))
        return batch_results


if __name__ == "__main__":
    # Example usage (for testing)
    from PIL import Image
    import numpy as np
    
    # Create a test tool instance
    tool = MultiModalPythonTool("multi_modal_python_tool", "Tool for executing Python code with multimodal capabilities", {})
    
    # Create a test image for testing
    test_image = Image.new('RGB', (100, 100), color='red')
    test_multi_modal_data = {"image": [test_image]}
    
    # Reset the tool with test data
    tool.reset("test prompt", test_multi_modal_data, test_multi_modal_data)
    
    # Test 1: Simple math calculation with <code> tags (should return reward=0.1)
    math_action = """
    Let me solve this step by step:
    
    <code>
    ```python
    a = 5
    b = 3
    result = a + b
    print(f"The sum of {a} and {b} is {result}")
    ```
    </code>
    """
    obs, reward, done, info = tool.execute(math_action)
    print(f"Math calculation result - Reward: {reward}, Done: {done}, Info: {info}")
    print(f"Observation type: {type(obs)}")
    if isinstance(obs, dict):
        print(f"Has multi_modal_data: {'multi_modal_data' in obs}")
    print()
    
    # Test 2: Image processing with matplotlib and <code> tags (should return reward=0.1 and generate images)
    image_plot_action = """
    Let me analyze the input image and create a visualization:
    
    <code>
    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    
    # First, let's examine the input image
    if 'image_clue_0' in locals():
        img = image_clue_0
        print(f"Input image size: {img.size}")
        print(f"Input image mode: {img.mode}")
        
        # Display the original image
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        # Create a simple analysis
        img_array = np.array(img)
        plt.subplot(1, 3, 2)
        plt.hist(img_array.flatten(), bins=50, alpha=0.7)
        plt.title('Color Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        # Create a simple mathematical plot
        plt.subplot(1, 3, 3)
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        plt.plot(x, y, 'b-', linewidth=2)
        plt.title('Sine Wave')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("Analysis complete!")
    else:
        print("No input image found")
    ```
    </code>
    """
    obs, reward, done, info = tool.execute(image_plot_action)
    print(f"Image analysis result - Reward: {reward}, Done: {done}, Info: {info}")
    print(f"Observation type: {type(obs)}")
    if isinstance(obs, dict) and 'multi_modal_data' in obs:
        images = obs['multi_modal_data']['image']
        print(f"Generated {len(images)} image(s)")
        for i, img in enumerate(images):
            print(f"Image {i}: {img.size} pixels, mode: {img.mode}")
    print()
    
    # Test 3: Final answer format (should return done=True)
    answer_action = """
    Based on my analysis, I can now provide the final answer:
    
    <answer>
    \\boxed{The image is a 100x100 red square}
    </answer>
    """
    obs, reward, done, info = tool.execute(answer_action)
    print(f"Final answer result - Reward: {reward}, Done: {done}, Info: {info}")
    print()
    
    # Test 4: Code with syntax error (should return reward=0.0)
    error_action = """
    This code has an error:
    
    <code>
    ```python
    x = 5
    y = x + 
    print(y)
    ```
    </code>
    """
    obs, reward, done, info = tool.execute(error_action)
    print(f"Error code result - Reward: {reward}, Done: {done}, Info: {info}")
    print()
    
    # Test 5: No code provided (should return reward=0.0)
    no_code_action = """
    This message doesn't contain any Python code blocks.
    Just some regular text without <code> tags.
    """
    obs, reward, done, info = tool.execute(no_code_action)
    print(f"No code result - Reward: {reward}, Done: {done}, Info: {info}")
    print()
    
    # Test 6: Multiple code blocks (should execute the last one)
    multiple_code_action = """
    Here are multiple code blocks:
    
    <code>
    ```python
    print("First block")
    ```
    </code>
    
    Some text in between.
    
    <code>
    ```python
    print("Second block - this should be executed")
    result = 42
    print(f"The answer is {result}")
    ```
    </code>
    """
    obs, reward, done, info = tool.execute(multiple_code_action)
    print(f"Multiple code blocks result - Reward: {reward}, Done: {done}, Info: {info}")
    print()
    
    # Test 7: Code that generates both text and image output
    mixed_output_action = """
    Let me create a comprehensive analysis:
    
    <code>
    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Print some information
    print("Generating a comprehensive mathematical visualization...")
    
    # Create data
    x = np.linspace(-2*np.pi, 2*np.pi, 1000)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)
    
    # Create plot
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, y1, 'r-', label='sin(x)', linewidth=2)
    plt.title('Sine Function')
    plt.xlabel('x (radians)')
    plt.ylabel('sin(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(x, y2, 'b-', label='cos(x)', linewidth=2)
    plt.title('Cosine Function')
    plt.xlabel('x (radians)')
    plt.ylabel('cos(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(2, 2, 3)
    # Limit tan function to avoid infinity
    y3_limited = np.where(np.abs(y3) > 10, np.nan, y3)
    plt.plot(x, y3_limited, 'g-', label='tan(x)', linewidth=2)
    plt.title('Tangent Function (limited)')
    plt.xlabel('x (radians)')
    plt.ylabel('tan(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-10, 10)
    
    plt.subplot(2, 2, 4)
    plt.plot(x, y1, 'r-', label='sin(x)', linewidth=2)
    plt.plot(x, y2, 'b-', label='cos(x)', linewidth=2)
    plt.title('Sin and Cos Comparison')
    plt.xlabel('x (radians)')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization complete!")
    print("All trigonometric functions have been displayed.")
    print("Key observations:")
    print("- Sin and cos are periodic with period 2π")
    print("- Sin and cos are bounded between -1 and 1")
    print("- Tan has vertical asymptotes at odd multiples of π/2")
    ```
    </code>
    """
    obs, reward, done, info = tool.execute(mixed_output_action)
    print(f"Mixed output result - Reward: {reward}, Done: {done}, Info: {info}")
    print(f"Observation type: {type(obs)}")
    if isinstance(obs, dict) and 'multi_modal_data' in obs:
        images = obs['multi_modal_data']['image']
        print(f"Generated {len(images)} image(s)")
    print()
