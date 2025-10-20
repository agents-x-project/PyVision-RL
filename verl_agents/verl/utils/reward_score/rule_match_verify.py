from typing import Optional, Any, Union
import re

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+")

def _parse_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = _NUM_RE.search(value)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
    return None

def extract_answer_type(ground_truth: Union[int, str, float]) -> str:
    if isinstance(ground_truth, (int, float)):
        return "number"
    elif _parse_number(ground_truth) is not None:
        return "number"
    elif isinstance(ground_truth, str):
        assert re.fullmatch(r"[A-Z]", ground_truth.strip() or " "), "ground_truth should be a single uppercase letter or a int / float number"
        return "choice"
    else:
        raise ValueError(f"ground_truth should be a single uppercase letter or a int / float number, but got {ground_truth}")


def _compute_mra(y_hat: float, y_true: float, num_thresholds: int = 10) -> float:
    # Threshold set C = {0.5, 0.55, ..., 0.95}
    thresholds = [0.5 + 0.05 * i for i in range(num_thresholds)]
    # Handle y_true == 0 explicitly to avoid division by zero
    if y_true == 0:
        correct = 1.0 if y_hat == 0 else 0.0
        return correct  # identical across all thresholds
    rel_err = abs(y_hat - y_true) / abs(y_true)
    hits = sum(1 for theta in thresholds if rel_err < (1.0 - theta))
    return hits / len(thresholds)


def compute_score_rule(solution_str: str, 
                       ground_truth: Union[str, int, float],
                       apply_mra: bool = True) -> float:
    '''
    verify score by rule match
    ground_truth should be a single uppercase letter or a int / float number
    '''
    if str(solution_str).strip() == str(ground_truth).strip():
        return 1.0
    
    answer_type = extract_answer_type(ground_truth)

    if answer_type == "choice":
        m = re.search(r"[A-Z]", solution_str)
        if m:
            return 1.0 if m.group(0) == str(ground_truth) else 0.0
        else:
            return 0.0
    elif answer_type == "number":
        answer_number = _parse_number(solution_str)
        gt_number = _parse_number(ground_truth)
        if answer_number is not None:
            if apply_mra:
                return _compute_mra(answer_number, gt_number)
            else:
                return 1.0 if answer_number == gt_number else 0.0
        else:
            return 0.0

    else:
        raise ValueError(f"Undefined answer_type: {answer_type} with solution_str: '{solution_str}', ground_truth: {ground_truth}")


def compute_score_rule_mra(solution_str: str, 
                       ground_truth: Union[str, int, float],
                       apply_mra: bool = True) -> float:
    '''
    verify score by rule match
    ground_truth should be a single uppercase letter or a int / float number
    '''
    # if str(solution_str).strip() == str(ground_truth).strip():
    #     return 1.0

    answer_number = _parse_number(solution_str)
    gt_number = _parse_number(ground_truth)

    acc_reward = _compute_mra(answer_number, gt_number)
    if acc_reward >= 0.5:
        is_answer_right = True 
    else:
        is_answer_right = False
    reward = {}
    reward['score'] = 1.0 * acc_reward
    reward['is_answer_right'] = is_answer_right

    return reward


