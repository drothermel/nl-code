import pytest

from nl_code.datasets.task import CodeDataset, Task, TaskSource, TaskTarget


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
        target=TaskTarget(name="add"),
        source=TaskSource(code="def add(a, b):\n    return a + b\n"),
    )
    assert task.task_id == "HumanEval/0"
    assert task.target.name == "add"
    assert task.source.code == "def add(a, b):\n    return a + b\n"
    assert task.dataset == CodeDataset.HUMANEVAL_PLUS
    assert task.version == "v3"


def test_task_validate_raw_task_version_match() -> None:
    task = Task(
        dataset=CodeDataset.HUMANEVAL_PLUS,
        task_id="HumanEval/0",
        target=TaskTarget(name="add"),
        source=TaskSource(code="def add(a, b):\n    return a + b\n"),
    )

    class _Raw:
        version = "v3"

    task.validate_raw_task_version(_Raw())


def test_task_validate_raw_task_version_mismatch() -> None:
    task = Task(
        dataset=CodeDataset.HUMANEVAL_PLUS,
        task_id="HumanEval/0",
        target=TaskTarget(name="add"),
        source=TaskSource(code="def add(a, b):\n    return a + b\n"),
    )

    class _Raw:
        version = "v2"

    with pytest.raises(ValueError, match="does not match raw task version"):
        task.validate_raw_task_version(_Raw())
