import pytest
from pydantic import ValidationError

from nl_code.datasets.humaneval_task import (
    RawHumanEvalTask,
    extract_prompt_comments,
    extract_prompt_docstring,
    merge_prompt_and_solution,
)

from conftest import make_humaneval_row


class TestHelperFunctions:
    def test_merge_prompt_and_solution(self) -> None:
        result = merge_prompt_and_solution("def foo():", "    pass")
        assert result == "def foo():\n    pass\n"

    def test_merge_rejects_non_strings(self) -> None:
        with pytest.raises(ValueError, match="must be strings"):
            merge_prompt_and_solution(123, "x")

    def test_extract_prompt_docstring(self) -> None:
        prompt = 'def foo():\n    """Hello world."""\n'
        assert extract_prompt_docstring(prompt, "foo") == "Hello world."

    def test_extract_prompt_comments(self) -> None:
        prompt = "# a comment\ndef foo():\n    pass\n"
        assert extract_prompt_comments(prompt) == "a comment"

    def test_extract_prompt_comments_none(self) -> None:
        assert extract_prompt_comments("def foo():\n    pass\n") is None


class TestRawHumanEvalTask:
    def test_construction(self, valid_raw_task: RawHumanEvalTask) -> None:
        assert valid_raw_task.task_id == "HumanEval/0"
        assert valid_raw_task.entry_point == "add"
        assert valid_raw_task.validated is True

    def test_computed_gt_solution(self, valid_raw_task: RawHumanEvalTask) -> None:
        assert "def add" in valid_raw_task.gt_solution
        assert "return a + b" in valid_raw_task.gt_solution

    def test_computed_gt_solution_without_comments(
        self,
        valid_raw_task: RawHumanEvalTask,
    ) -> None:
        assert '"""' not in valid_raw_task.gt_solution_without_comments
        assert "add" in valid_raw_task.gt_solution_without_comments

    def test_computed_docstring(self, valid_raw_task: RawHumanEvalTask) -> None:
        assert (
            valid_raw_task.prompt_docstring == "Add two integers and return the result."
        )

    def test_computed_test_inputs(self, valid_raw_task: RawHumanEvalTask) -> None:
        assert valid_raw_task.test_inputs == [[1, 2], [0, 0], [-1, 1]]
        assert valid_raw_task.test_results == [3, 0, 0]

    def test_run_test_passes_gt(self, valid_raw_task: RawHumanEvalTask) -> None:
        assert valid_raw_task.run_test_on_gt_solution() is True

    def test_run_test_fails_bad_code(self, valid_raw_task: RawHumanEvalTask) -> None:
        bad_code = "def add(a, b):\n    return a - b\n"
        assert valid_raw_task.run_test(bad_code) is False

    def test_validation_rejects_failing_solution(self) -> None:
        row = make_humaneval_row(canonical_solution="    return a - b\n")
        with pytest.raises(ValidationError):
            RawHumanEvalTask.model_validate(row)

    def test_validated_flag_skips_validation(self) -> None:
        row = make_humaneval_row(canonical_solution="    return a - b\n")
        row["validated"] = True
        task = RawHumanEvalTask.model_validate(row)
        assert task.validated is True
