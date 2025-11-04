from openai import OpenAI
import requests
import random
import re
import os

from math_verify import parse, verify

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def extract_answer(text):
    """
    从给定的文本中提取<answer>\boxed{answer content}</answer>标签内部的内容。
    
    参数:
        text (str): 包含<answer>标签的文本
        
    返回:
        str or None: 标签内部\boxed{}中的内容，如果未找到则返回None。
    """
    # 匹配 <answer>\boxed{answer content}</answer> 格式
    # 使用非贪婪模式匹配\boxed{和}之间的内容
    pattern = r'<answer>\n\\boxed\{(.*?)\}\n</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def grade_with_llm(client, model_name, question: str, correct_answer: str, response: str) -> bool:
    """
    使用OpenAI API判断预测答案是否正确
    """
    grader_prompt = GRADER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )
    try:
        completion = client.chat.completions.create(
            model=model_name,
            max_tokens=1024,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are an expert grader that evaluates answer correctness."},
                {"role": "user", "content": grader_prompt}
            ],
            timeout=30
        )
        grading_response = completion.choices[0].message.content
        match = re.search(r"correct: (yes|no)", grading_response, re.IGNORECASE)
        return match.group(1).lower() == "yes" if match else False
    except Exception as e:
        print(f"Error grading with OpenAI: {e}")
        return False


def compute_score(predict_str: str, ground_truth: str, extra_info=None, llm_as_a_judge_config=None):
    is_format_error = False

    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split('</think>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = extract_answer(predict_str)

    if answer_text is None:
        reward = {}
        reward['score'] = 0.0
        reward['is_answer_right'] = False
        return reward
    elif "<interpreter>" in answer_text or "<answer>" in answer_text or "\\boxed" in answer_text:
        reward = {}
        reward['score'] = 0.0
        reward['is_answer_right'] = False
        return reward

    question_text = extra_info['question']

    client = OpenAI(
        api_key=llm_as_a_judge_config['api_key'],
        base_url=llm_as_a_judge_config['base_url'],
    )
    model_name = llm_as_a_judge_config['model_name']

    is_correct = grade_with_llm(client=client, model_name=model_name, question=question_text, correct_answer=ground_truth, response=answer_text)

    is_answer_right = is_correct

    if is_answer_right:
        acc_reward = 1.0
    else:
        acc_reward = 0.0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    #### pyvision reward ####
    reward = {}
    reward['score'] = 1.0 * acc_reward
    reward['is_answer_right'] = is_answer_right

    return reward

if __name__ == '__main__':
    predict_str = "The answer is <think> 2 + 2 = 4 </think> <answer> right </answer> <answer> left </answer>"
    ground_truth = "left"
    extra_info = {'answer': 'The woman is to the left of the man who is holding the camera.', 'id': 0, 'image': '/cpfs/user/honglingyi/DATA/LLM/Vstar/gqa/images/713270.jpg', 'pred_ans': 'The woman is to the right of the man who is holding the camera.', 'question': 'Is the woman to the left or to the right of the man who is holding the camera?'}

    score = compute_score(predict_str, ground_truth, extra_info)
    print(f"Score: {score}")