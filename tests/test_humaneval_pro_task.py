import pytest

from nl_code.datasets.humaneval_pro_task import RawHumanEvalProTask

from conftest import make_humaneval_pro_row

pytestmark = pytest.mark.docker


class TestRawHumanEvalProTask:
    def test_construction(self) -> None:
        row = make_humaneval_pro_row()
        row["task_id"] = "HumanEvalPro/0"
        task = RawHumanEvalProTask.model_validate(row)
        assert task.task_id == "HumanEvalPro/0"
        assert task.source__raw_problem == row["raw_problem"]
        assert task.source__raw_solution == row["raw_solution"]
        assert task.source__new_problem == row["new_problem"]
        assert task.source__new_solution == row["new_solution"]
        assert task.source__test_code == row["test_code"]
        assert task.validated is False

    def test_gt_solution_contains_both_functions(self) -> None:
        row = make_humaneval_pro_row()
        row["task_id"] = "HumanEvalPro/0"
        task = RawHumanEvalProTask.model_validate(row)
        assert "def add" in task.gt_solution
        assert "def add_pairs" in task.gt_solution

    def test_gt_solution_base_before_new(self) -> None:
        row = make_humaneval_pro_row()
        row["task_id"] = "HumanEvalPro/0"
        task = RawHumanEvalProTask.model_validate(row)
        add_pos = task.gt_solution.index("def add(")
        add_pairs_pos = task.gt_solution.index("def add_pairs(")
        assert add_pos < add_pairs_pos

    def test_gt_solution_without_comments(self) -> None:
        row = make_humaneval_pro_row()
        row["task_id"] = "HumanEvalPro/0"
        task = RawHumanEvalProTask.model_validate(row)
        assert '"""' not in task.gt_solution_without_comments
        assert "#" not in task.gt_solution_without_comments
        assert "def add" in task.gt_solution_without_comments

    def test_new_entry_point(self) -> None:
        row = make_humaneval_pro_row()
        row["task_id"] = "HumanEvalPro/0"
        task = RawHumanEvalProTask.model_validate(row)
        assert task.new_entry_point == "add_pairs"

    def test_new_description(self) -> None:
        row = make_humaneval_pro_row()
        row["task_id"] = "HumanEvalPro/0"
        task = RawHumanEvalProTask.model_validate(row)
        assert "list of pairs" in task.new_description

    def test_run_test_passes_gt(self) -> None:
        row = make_humaneval_pro_row()
        row["task_id"] = "HumanEvalPro/0"
        task = RawHumanEvalProTask.model_validate(row)
        assert task.run_test_on_gt_solution() is True

    def test_run_test_fails_bad_code(self) -> None:
        row = make_humaneval_pro_row()
        row["task_id"] = "HumanEvalPro/0"
        task = RawHumanEvalProTask.model_validate(row)
        bad_code = "def add_pairs(pairs):\n    return []\n"
        assert task.run_test(bad_code) is False

    def test_construction_allows_failing_solution(self) -> None:
        row = make_humaneval_pro_row(
            new_solution="    return []\n",
        )
        row["task_id"] = "HumanEvalPro/0"
        task = RawHumanEvalProTask.model_validate(row)
        assert task.validated is False
        assert task.run_test_on_gt_solution() is False

    def test_validated_flag_skips_validation(self) -> None:
        row = make_humaneval_pro_row(
            new_solution="    return []\n",
        )
        row["task_id"] = "HumanEvalPro/0"
        row["validated"] = True
        task = RawHumanEvalProTask.model_validate(row)
        assert task.validated is True
