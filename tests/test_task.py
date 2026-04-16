import pytest

from nl_code.datasets.task import CodeDataset, Task


def test_code_dataset_humaneval_plus() -> None:
    assert CodeDataset.HUMANEVAL_PLUS == "evalplus/humanevalplus"


def test_code_dataset_humaneval_pro() -> None:
    assert CodeDataset.HUMANEVAL_PRO == "CodeEval-Pro/humaneval-pro"


def test_code_dataset_mbpp_pro() -> None:
    assert CodeDataset.MBPP_PRO == "CodeEval-Pro/mbpp-pro"


def test_code_dataset_bigcodebench_lite_pro() -> None:
    assert CodeDataset.BIGCODEBENCH_LITE_PRO == "CodeEval-Pro/bigcodebench-lite-pro"


def test_task_construction() -> None:
    task = Task(
        dataset=CodeDataset.HUMANEVAL_PLUS,
        task_id="HumanEval/0",
        entry_point_name="add",
        description="Add two integers and return the result.",
        gt_solution="def add(a, b):\n    return a + b\n",
    )
    assert task.task_id == "HumanEval/0"
    assert task.entry_point_name == "add"
    assert task.description == "Add two integers and return the result."
    assert task.dataset == CodeDataset.HUMANEVAL_PLUS
    assert task.version == "v2"


def test_task_validate_raw_task_version_match() -> None:
    task = Task(
        dataset=CodeDataset.HUMANEVAL_PLUS,
        task_id="HumanEval/0",
        entry_point_name="add",
        description="Add two integers and return the result.",
        gt_solution="def add(a, b):\n    return a + b\n",
    )

    class _Raw:
        version = "v2"

    task.validate_raw_task_version(_Raw())


def test_task_validate_raw_task_version_mismatch() -> None:
    task = Task(
        dataset=CodeDataset.HUMANEVAL_PLUS,
        task_id="HumanEval/0",
        entry_point_name="add",
        description="Add two integers and return the result.",
        gt_solution="def add(a, b):\n    return a + b\n",
    )

    class _Raw:
        version = "v1"

    with pytest.raises(ValueError, match="does not match raw task version"):
        task.validate_raw_task_version(_Raw())
