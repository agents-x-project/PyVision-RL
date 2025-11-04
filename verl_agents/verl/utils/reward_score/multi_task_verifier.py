from openai import OpenAI
import requests
import random
import re
import os

from math_verify import parse, verify

def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template

def get_gpt4_score_ICE_visual_search():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
""" # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
""" # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
""" # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
""" # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
""" # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
""" # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
""" # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


def get_gpt4_score_ICE_math():
    example_1 = """
[Question]: The histogram below shows the distribution of Mike's 8 shots. What is Mike's average score ( )?
[Standard Answer]: 8
[Model_answer]: 8
Judgement: 1
"""  # noqa

    example_2 = """
[Question]: A factory produces products. A product is randomly selected, and the random variable X is defined as follows: X=1 indicates the product is qualified, X=0 indicates the product is unqualified. The probability that a product is qualified is 0.7. What is the probability that both of two randomly selected products are qualified?
[Standard Answer]: 0.49
[Model_answer]: 0.7 × 0.7 = 0.49
Judgement: 1
"""  # noqa

    example_3 = """
[Question]: From a box containing 10 balls, 3 red and 7 blue, balls are drawn with replacement. Let S₄ be the total number of red balls drawn in 4 draws. Find P(S₄ ≥ 2).
[Standard Answer]: 0.348
[Model_answer]: 0.348
Judgement: 1
"""  # noqa

    example_4 = """
[Question]: The lifetime X (in hours) of a component follows an exponential distribution with parameter λ=0.3. Find P(X>10).
[Standard Answer]: 0.05
[Model_answer]: 0.394
Judgement: 0
"""  # noqa

    example_5 = """
[Question]: The random variable X has a probability density function as shown in the figure. What should the coefficient k be?
[Standard Answer]: 3/8
[Model_answer]: 0.375
Judgement: 1
"""  # noqa

    example_6 = """
[Question]: As shown in the figure, the curve is the probability density function of a normal distribution. Calculate the variance.
[Standard Answer]: 100
[Model_answer]: 50
Judgement: 0
"""  # noqa

    example_7 = """
[Question]: As shown in the figure, the bar chart represents the probability distribution of a discrete random variable X. Calculate the standard deviation.
[Standard Answer]: 1.4
[Model_answer]: 1.4
Judgement: 1
"""  # noqa

    example_8 = """
[Question]: Given that X ~ N(80, 4²), what is P(X < 76)?
[Standard Answer]: 0.1587
[Model_answer]: 0.84
Judgement: 0
"""  # noqa

    example_9 = """
[Question]: The lifetime X of a component follows an exponential distribution with parameter λ=0.1. Calculate the probability that the lifetime exceeds 10 hours, P(X>10).
[Standard Answer]: 0.394
[Model_answer]: 0.394
Judgement: 1
"""  # noqa

    example_10 = """
[Question]: The bar chart shows the distribution of scores from Mike’s 8 attempts. What is the mean score?
[Standard Answer]: 8.125
[Model_answer]: 8.12
Judgement: 1
"""  # noqa

    return [
        example_1,
        example_2,
        example_3,
        example_4,
        example_5,
        example_6,
        example_7,
        example_8,
        example_9,
        example_10,
    ]
    
def get_gpt4_score_ICE_chart():
    example_1 = """
[Question]: What is the title of the chart?
[Standard Answer]: The title of the chart is 'Comparison of Literacy Rates by Country'.
[Model_answer]: Comparison of Literacy Rates by Country
Judgement: 1
"""  # descriptive

    example_2 = """
[Question]: How many subplots are present in the figure?
[Standard Answer]: There are 4 subplots in the figure.
[Model_answer]: Four subplots.
Judgement: 1
"""  # descriptive

    example_3 = """
[Question]: What is the color used for the background of the overall chart?
[Standard Answer]: The color used for the background of the overall chart is off-white.
[Model_answer]: white
Judgement: 0
"""  # descriptive (model slightly incorrect)

    example_4 = """
[Question]: What is the percentage increase in the number of 'Prototypes Developed' compared to 'Ideas Proposed'?
[Standard Answer]: The values for 'Prototypes Developed' and 'Ideas Proposed' are 800 and 500 respectively. The percentage increase is [(800 - 500) / 500] * 100 = 60%. Answer: 60%.
[Model_answer]: 60%
Judgement: 1
"""  # reasoning

    example_5 = """
[Question]: Which metric has the smallest value across all quartiles, and what is the corresponding quartile and value?
[Standard Answer]: The smallest value is 65.0, corresponding to the metric 'Skewness' and quartile 'Q1'. Answer: Skewness, Q1, 65.0
[Model_answer]: Q1 of Skewness, value 65.0
Judgement: 1
"""  # reasoning

    example_6 = """
[Question]: What is the percentage difference between 'Digital Media' and 'Broadcast Journalism' funding allocations?
[Standard Answer]: The allocation for 'Digital Media' is 30.0% and for 'Broadcast Journalism' is 18.0%. The percentage difference is 12.0%.
[Model_answer]: 18%
Judgement: 0
"""  # reasoning (incorrect numerical answer)

    return [example_1, example_2, example_3, example_4, example_5, example_6]

def get_gpt4_score_ICE_longvila():
    example_1 = """
[Question]: What is the primary goal of the video creator, based on the sequence of actions and contextual clues across all timestamps?
A. Demonstrate the steps to thoroughly clean a dusty computer tower
B. Promote a specific branded computer fan product for RGB customization
C. Educate viewers on general computer hardware assembly principles
D. Showcase a custom-built computer setup for gaming or professional use
[Standard Answer]: B
[Model_answer]: Educate viewers on general computer hardware assembly principles
Judgement: 0
"""  # noqa

    example_2 = """
[Question]: What is the woman's most likely overarching intention throughout the video, based on the progression of scenes and contextual clues?  
A. Promoting a commercial baby product brand through demonstrations  
B. Sharing parenting tips for toddlers through casual storytelling  
C. Documenting her pregnancy journey for a personal blog audience  
D. Advertising home organization strategies using her living space  
[Standard Answer]: C
[Model_answer]: A. Promoting a commercial baby product brand through demonstrations
Judgement: 0
"""  # noqa

    example_3 = """
[Question]: Based on the video, what is the most likely primary intention of the driver (man in glasses and gray t-shirt) interacting with multiple Mazda MX-5 vehicles in different contexts?  
A. He is a car reviewer test-driving different trims of the same model.  
B. He is evaluating a specific used car's condition to purchase it.  
C. He is commuting to work using different cars he owns.  
D. He is practicing parallel parking in various locations.  
[Standard Answer]: B
[Model_answer]: A. He is a car reviewer test-driving different trims of the same model.
Judgement: 0
"""  # noqa

    example_4 = """
[Question]: What is the creators' primary intention behind including the scenes featuring the Tomcat mouse trap (0:00:20-0:00:30) and the "Horn" product (0:02:50-0:03:00 and 0:03:10-0:03:20) in the video?  
A. To demonstrate the proper use of common garage tools like drills and pliers  
B. To highlight a repair process for a malfunctioning wooden box labeled "Top Secret"  
C. To contrast commercial pest control products with homemade workshop solutions  
D. To promote a DIY project tutorial available on "THEKINGOFRANDOM.COM"  
[Standard Answer]: D
[Model_answer]: C. To contrast commercial pest control products with homemade workshop solutions
Judgement: 0
"""  # noqa

    example_5 = """
[Question]: Based on the video, which overarching goal is Albus Dumbledore most likely pursuing across the depicted scenes in "Harry Potter and the Goblet of Fire" and later films?  
A. To publicly expose Lord Voldemort's return using the Triwizard Tournament  
B. To manipulate the Golden Snitch to ensure Harry’s survival via the Resurrection Stone  
C. To prevent the Goblet of Fire from being corrupted by Death Eaters  
D. To restore Hogwarts’ safety after the events of the Second Wizarding War  
[Standard Answer]: B
[Model_answer]: To restore Hogwarts’ safety after the events of the Second Wizarding War
Judgement: 0
"""  # noqa

    example_6 = """
[Question]: What was the creator’s primary intention in structuring the video to include roller coasters from Schwarzkopf Industries, Gerstlauer Amusement Rides, and others, alongside diverse elements like castles, dinosaurs, and varying track colors?  
A. To chronologically document the historical development of roller coasters.  
B. To critique the safety standards of different amusement park manufacturers.  
C. To highlight how thematic storytelling and engineering diversity enhance rider experiences.  
D. Safety critiques (B) and historical documentation (A) are never implied; focus is on experiential diversity (C).
[Standard Answer]: C
[Model_answer]: D
Judgement: 0
"""  # noqa

    example_7 = """
[Question]: What is the primary intention behind the man in the green shirt’s activities across the video segments, based on contextual clues and recurring themes?  
A. Promoting a local stationery business (Karachi Stationery Mart)  
B. Documenting a personal travel experience for a YouTube series  
C. Conducting academic research on urban infrastructure in Delhi  
D. Advocating for affordable street food pricing (e.g., Rs.20/Rs.70 signs)  
[Standard Answer]: B
[Model_answer]: D. Advocating for affordable street food pricing (e.g., Rs.20/Rs.70 signs)
Judgement: 0
"""  # noqa

    example_8 = """
[Question]: What is the man's primary motivation for opposing Phaserbeam Computers, as inferred from the interconnected scenes and dialogue?  
A. To gain personal fame as a technological innovator surpassing Phaserbeam.  
B. To protect his family from threats indirectly linked to Phaserbeam's influence.  
C. However, the revolution is likely a *means* to protect his family, not an end goal. The familial emphasis (0:00:30–0:00:40) anchors his actions in personal rather than purely ideological motives.  
D. To resolve a professional rivalry stemming from workplace competition.  
[Standard Answer]: B
[Model_answer]: C. However, the revolution is likely a *means* to protect his family, not an end goal. The familial emphasis (0:00:30–0:00:40) anchors his actions in personal rather than purely ideological motives.
Judgement: 0
"""  # noqa

    # 尽管所有例子目前都是错误回答（Judgement: 0），但为了泛化性，我们加入一个正确示例
    example_9 = """
[Question]: What is the primary intention behind the man in the green shirt’s activities across the video segments, based on contextual clues and recurring themes?  
A. Promoting a local stationery business (Karachi Stationery Mart)  
B. Documenting a personal travel experience for a YouTube series  
C. Conducting academic research on urban infrastructure in Delhi  
D. Advocating for affordable street food pricing (e.g., Rs.20/Rs.70 signs)  
[Standard Answer]: B
[Model_answer]: B
Judgement: 1
"""  # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7, example_8, example_9]

def get_gpt4_score_ICE_vsi():
    example_1 = """
[Question]: If I am standing by the book and facing the box, is the suitcase to the left or the right of the box?
A. left
B. right
[Standard Answer]: B
[Model_answer] : B
Judgement: 1
"""  # noqa

    example_2 = """
[Question]: If I am standing by the whiteboard and facing the doorframe, is the cabinets to the left or the right of the doorframe?
A. right
B. left
[Standard Answer]: A
[Model_answer] : right
Judgement: 1
"""  # noqa

    example_3 = """
[Question]: If I am standing by the refrigerator and facing the coffee kettle, is the cabinet to my left, right, or back?
An object is to my back if I would have to turn at least 135 degrees in order to face it.
A. left
B. back
C. right
[Standard Answer]: C
[Model_answer] : C
Judgement: 1
"""  # noqa

    example_4 = """
[Question]: What will be the first-time appearance order of the following categories in the video: radiator, shelf, shower curtain rod, soap dispenser?
A. soap dispenser, radiator, shelf, shower curtain rod
B. soap dispenser, shelf, radiator, shower curtain rod
C. radiator, soap dispenser, shelf, shower curtain rod
D. radiator, shelf, shower curtain rod, soap dispenser
[Standard Answer]: C
[Model_answer] : C
Judgement: 1
"""  # noqa

    example_5 = """
[Question]: How many cabinet(s) are in this room?
[Standard Answer]: 2
[Model_answer] : 2
Judgement: 1
"""  # noqa

    example_6 = """
[Question]: If I am standing by the desk and facing the door, is the keyboard to the left or the right of the door?
A. right
B. left
[Standard Answer]: A
[Model_answer] : left
Judgement: 0
"""  # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6]


def get_prompt(predict_str, ground_truth, question, data_source):
    if data_source == "deepeyes" or data_source == "minio3":
        examples = get_gpt4_score_ICE_visual_search()
    elif data_source == "ecd":
        examples = get_gpt4_score_ICE_chart()
    elif data_source == "wemath-standard" or data_source == "wemath-pro":
        examples = get_gpt4_score_ICE_math()
    elif data_source == "vsi":
        examples = get_gpt4_score_ICE_vsi()
    elif data_source == "longvila":
        examples = get_gpt4_score_ICE_longvila()
        
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + '\n\n'
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f'{demo_prompt}{test_prompt}'


    return full_prompt

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


def compute_score(predict_str: str, ground_truth: str, extra_info=None, llm_as_a_judge_config=None, data_source=None):
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

    # answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
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
    full_prompt = get_prompt(answer_text, ground_truth, question_text, data_source)

    client = OpenAI(
        api_key=llm_as_a_judge_config['api_key'],
        base_url=llm_as_a_judge_config['base_url'],
    )
    model_name = llm_as_a_judge_config['model_name']

    chat_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        seed = random.randint(0, 1000000),
        temperature=0.3,
    )
    response = chat_response.choices[0].message.content.strip()
    # print(f"##################################### response: {response}")
    # print(response)
    is_answer_right = False
    if 'Judgement:' in response:
        response = response.split('Judgement:')[-1].strip()
        if '1' in response:
            acc_reward = 1.0
            is_answer_right = True
        elif '0' in response:
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
            acc_reward = 0.0
    else:
        if response == '1':
            acc_reward = 1.0
            is_answer_right = True
        elif response == '0':
            acc_reward = 0.0
        else:
            print(f' [WARNING] resp format error {response=}')
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