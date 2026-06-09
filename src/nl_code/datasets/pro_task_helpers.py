import ast
from typing import Any

from nl_code.code_parsing import (
    extract_top_level_import_source,
    find_docstring_expr,
    find_first_function_name,
    find_first_function_node,
    get_comments,
    get_docstrings_and_comments,
    get_first_function_docstring,
    merge_code_components,
    remove_docstrings_preserving_comments,
    remove_docstrings_and_comments,
    remove_full_line_comments,
)
from nl_code.datasets.validation import require_string


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
    return build_two_part_code(base, new)


def _merge_nonempty_code_components(*components: str) -> str:
    present_components = [component for component in components if component.strip()]
    if not present_components:
        return ""
    return merge_code_components(*present_components)


def _merge_two_code_components_with_blank_line(first: str, second: str) -> str:
    first_stripped = first.rstrip()
    second_stripped = second.rstrip()
    if not first_stripped and not second_stripped:
        return ""
    if not first_stripped:
        return second_stripped + "\n"
    if not second_stripped:
        return first_stripped + "\n"
    return f"{first_stripped}\n\n\n{second_stripped}\n"


def build_two_part_code(first_part: Any, second_part: Any) -> str:
    first_part_str = require_string(first_part, name="first_part")
    second_part_str = require_string(second_part, name="second_part")
    return _merge_two_code_components_with_blank_line(first_part_str, second_part_str)


def build_original_official_prompt(raw_problem: Any) -> str:
    raw_problem_str = require_string(raw_problem, name="raw_problem")
    return (
        "You are an exceptionally intelligent coding assistant that "
        "consistently delivers accurate and reliable responses to user "
        "instructions. Write a solution of python file to the following problem\n"
        "@@ Instruction \n"
        f"{raw_problem_str.rstrip()}\n"
        "@@ Response\n"
    )


def build_new_official_prompt(raw_problem: Any, new_problem: Any) -> str:
    raw_problem_str = require_string(raw_problem, name="raw_problem")
    new_problem_str = require_string(new_problem, name="new_problem")
    return (
        "You are an exceptionally intelligent coding assistant that "
        "consistently delivers accurate and reliable responses to user "
        "instructions. Write a solution of python file to the following "
        "problems, the solution of the second problem requires single or "
        "multiple calls to the first\n"
        "@@ Instruction \n"
        "```python\n"
        f"{raw_problem_str.rstrip()}\n"
        f"{new_problem_str.rstrip()}\n"
        "```\n"
        "@@ Response\n"
    )


def _join_nonempty_text_parts(*parts: str) -> str:
    present_parts = [part.strip() for part in parts if part.strip()]
    return "\n\n".join(present_parts)


def build_new_function_source(new_problem: Any, new_solution: Any) -> str:
    new_problem_str = require_string(new_problem, name="new_problem")
    new_solution_str = require_string(new_solution, name="new_solution")
    return merge_code_components(new_problem_str, new_solution_str)


def build_original_function_source(raw_problem: Any, raw_solution: Any) -> str:
    raw_problem_str = require_string(raw_problem, name="raw_problem")
    raw_solution_str = require_string(raw_solution, name="raw_solution")
    return merge_code_components(raw_problem_str, raw_solution_str)


def build_original_function_without_docstrings_and_comments(
    raw_problem: Any, raw_solution: Any
) -> str:
    return remove_docstrings_and_comments(
        build_original_function_source(raw_problem, raw_solution)
    )


def build_new_function_without_docstrings_and_comments(
    new_problem: Any, new_solution: Any
) -> str:
    return remove_docstrings_and_comments(
        build_new_function_source(new_problem, new_solution)
    )


def extract_source_imports(source: Any, *, field_name: str) -> str:
    source_str = require_string(source, name=field_name)
    return extract_top_level_import_source(source_str, allow_stub_body=True)


def extract_function_docstring(source: Any, *, field_name: str) -> str:
    source_str = require_string(source, name=field_name)
    return get_first_function_docstring(source_str, allow_stub_body=True)


def extract_docstrings_and_comments(source: Any, *, field_name: str) -> str:
    source_str = require_string(source, name=field_name)
    return get_docstrings_and_comments(source_str, allow_stub_body=True)


def extract_verified_new_docstring(new_function: Any, new_solution: Any) -> str:
    new_function_str = require_string(new_function, name="new_function")
    new_solution_str = require_string(new_solution, name="new_solution")
    function_node = find_first_function_node(new_function_str, allow_stub_body=True)
    docstring_expr = find_docstring_expr(function_node)
    if docstring_expr is None:
        return ""

    docstring_source = ast.get_source_segment(new_function_str, docstring_expr)
    if docstring_source is None or docstring_source not in new_solution_str:
        raise ValueError("new function docstring must be present in new_solution")
    return get_first_function_docstring(new_function_str, allow_stub_body=True)


def extract_verified_new_docstrings_and_comments(
    new_problem: Any, new_solution: Any
) -> str:
    new_problem_str = require_string(new_problem, name="new_problem")
    new_solution_str = require_string(new_solution, name="new_solution")
    new_function_str = build_new_function_source(new_problem_str, new_solution_str)
    comments = get_comments(new_function_str, strip_hash=True) or ""
    docstring = extract_verified_new_docstring(new_function_str, new_solution_str)
    return _join_nonempty_text_parts(comments, docstring)


def extract_problem_comments(problem: Any, *, field_name: str) -> str:
    problem_str = require_string(problem, name=field_name)
    return get_comments(problem_str, strip_hash=True) or ""


def build_function_stub_without_docstrings(source: Any, *, field_name: str) -> str:
    source_str = require_string(source, name=field_name)
    return remove_docstrings_preserving_comments(source_str)


def build_function_stub_without_docstrings_and_comments(
    source: Any, *, field_name: str
) -> str:
    source_str = require_string(source, name=field_name)
    without_docstrings = remove_docstrings_preserving_comments(source_str)
    return remove_full_line_comments(without_docstrings)


def build_problem_stub_without_docstrings_and_comments(
    problem: Any, *, field_name: str
) -> str:
    problem_str = require_string(problem, name=field_name)
    without_docstrings = remove_docstrings_preserving_comments(problem_str)
    return remove_full_line_comments(without_docstrings)


def build_new_function_stub(raw_problem_imports: Any, new_problem_stub: Any) -> str:
    raw_problem_imports_str = require_string(
        raw_problem_imports, name="raw_problem_imports"
    )
    new_problem_stub_str = require_string(new_problem_stub, name="new_problem_stub")
    return _merge_nonempty_code_components(
        raw_problem_imports_str, new_problem_stub_str
    )


def build_new_two_part_function_stub(raw_problem: Any, new_problem_stub: Any) -> str:
    raw_problem_str = require_string(raw_problem, name="raw_problem")
    new_problem_stub_str = require_string(new_problem_stub, name="new_problem_stub")
    return build_two_part_code(raw_problem_str, new_problem_stub_str)


def build_two_part_prompt(first_part: Any, second_part: Any) -> str:
    return build_two_part_code(first_part, second_part)


def extract_new_entry_point(new_problem: Any, new_solution: Any) -> str:
    if not isinstance(new_problem, str) or not isinstance(new_solution, str):
        raise ValueError("new_problem and new_solution must be strings")
    new_function = merge_code_components(new_problem, new_solution)
    return find_first_function_name(new_function)


def extract_new_description(new_problem: Any) -> str:
    if not isinstance(new_problem, str):
        raise ValueError("new_problem must be a string")
    return get_comments(new_problem, strip_hash=True) or ""
