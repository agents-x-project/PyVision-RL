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
# from . import gsm8k, math, prime_math, prime_code
import torch

def _default_compute_score(data_source, solution_str, ground_truth, extra_info=None, llm_as_a_judge_config=None):
    # print(f"###################################### data_source: {data_source}")
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from . import prime_code

        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)

    elif data_source in ['rag_v2-train']:
        from . import agent
        res = agent.compute_score(solution_str, ground_truth)
    elif data_source in ['rag_v2-test']:
        from . import agent
        res = agent.compute_score_eval(solution_str, ground_truth)

    elif data_source in ['vstar', 'vl_agent', 'chart', 'zebra_cot', 'vigorl', 'deepeyes', 'math_8k_verified', 'barc', 'wemath-standard', 'wemath-pro', 'minio3', 'vsi', 'longvila', 'ecd']:
        from verl.utils.reward_score import vl_agent, general_verifier, multi_task_verifier
        # print(f"################################## import the vl_agent successfully.")
        res = multi_task_verifier.compute_score(solution_str, ground_truth, extra_info, llm_as_a_judge_config, data_source)
        # res = general_verifier.compute_score(solution_str, ground_truth, extra_info, llm_as_a_judge_config)
        # print(f"############################# reward result: {res}")

    elif data_source in ['vsi_numerical'] or data_source.endswith('_rule_verifiable'):
        from . import rule_match_verify
        res = rule_match_verify.compute_score_rule_mra(solution_str, ground_truth)

    elif data_source in ['geoguessr']:
        from . import vl_agent
        res = vl_agent.compute_common_reasoning(solution_str, ground_truth, extra_info)

    elif data_source in ['thinklite_eureka', 'xince']:
        from . import vl_agent
        res = vl_agent.compute_score_math(solution_str, ground_truth, extra_info)

    elif data_source in ["frozenlake"]:
        res = 0.0

    else:
        from verl.utils.reward_score import vl_agent, general_verifier, multi_task_verifier
        res = multi_task_verifier.compute_score(solution_str, ground_truth, extra_info, llm_as_a_judge_config, data_source)
        # res = general_verifier.compute_score(solution_str, ground_truth, extra_info, llm_as_a_judge_config)
        print(f"[WARNING] Reward function is not implemented for {data_source=}, fall back to general_verifier (llm as judge)")

        # raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
