import pytest
from pydantic import ValidationError

from nl_code.datasets.bigcodebench_lite_pro_task import RawBigCodeBenchLiteProTask

from conftest import make_bigcodebench_lite_pro_row


class TestRawBigCodeBenchLiteProTask:
    def test_construction(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert task.task_id == "BigCodeBenchLitePro/23"
        assert task.validated is True

    def test_gt_solution_contains_both_functions(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert "def multiply" in task.gt_solution
        assert "def multiply_pairs" in task.gt_solution

    def test_gt_solution_base_before_new(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        mul_pos = task.gt_solution.index("def multiply(")
        mul_pairs_pos = task.gt_solution.index("def multiply_pairs(")
        assert mul_pos < mul_pairs_pos

    def test_gt_solution_without_comments(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert '"""' not in task.gt_solution_without_comments
        assert "#" not in task.gt_solution_without_comments
        assert "def multiply" in task.gt_solution_without_comments

    def test_new_entry_point(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert task.new_entry_point == "multiply_pairs"

    def test_new_description(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert "list of pairs" in task.new_description

    def test_run_test_passes_gt(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert task.run_test_on_gt_solution() is True

    def test_run_test_fails_bad_code(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        bad_code = "def multiply_pairs(pairs):\n    return []\n"
        assert task.run_test(bad_code) is False

    def test_validation_rejects_failing_solution(self) -> None:
        row = make_bigcodebench_lite_pro_row(
            new_solution="    return []\n",
        )
        row["task_id"] = "BigCodeBenchLitePro/23"
        with pytest.raises(ValidationError):
            RawBigCodeBenchLiteProTask.model_validate(row)

    def test_validated_flag_skips_validation(self) -> None:
        row = make_bigcodebench_lite_pro_row(
            new_solution="    return []\n",
        )
        row["task_id"] = "BigCodeBenchLitePro/23"
        row["validated"] = True
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert task.validated is True
