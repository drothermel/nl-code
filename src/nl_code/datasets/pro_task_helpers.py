from typing import Any

from nl_code.code_parsing import (
    find_first_function_name,
    get_comments,
    merge_code_components,
)


def build_gt_solution(
    raw_problem: Any,
    raw_solution: Any,
    new_problem: Any,
    new_solution: Any,
) -> str:
    for val in (raw_problem, raw_solution, new_problem, new_solution):
        if not isinstance(val, str):
            raise ValueError("all solution components must be strings")
    base = merge_code_components(raw_problem, raw_solution)
    new = merge_code_components(new_problem, new_solution)
    return merge_code_components(base, new)


def extract_new_entry_point(new_problem: Any, new_solution: Any) -> str:
    if not isinstance(new_problem, str) or not isinstance(new_solution, str):
        raise ValueError("new_problem and new_solution must be strings")
    new_function = merge_code_components(new_problem, new_solution)
    return find_first_function_name(new_function)


def extract_new_description(new_problem: Any) -> str:
    if not isinstance(new_problem, str):
        raise ValueError("new_problem must be a string")
    return get_comments(new_problem, strip_hash=True) or ""
