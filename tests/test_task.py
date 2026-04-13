from nl_code.datasets.task import CompressibleDataset, Task


def test_compressible_dataset_value() -> None:
    assert CompressibleDataset.HUMAN_EVAL_PLUS == "evalplus/humanevalplus"


def test_task_construction() -> None:
    task = Task(
        dataset=CompressibleDataset.HUMAN_EVAL_PLUS,
        task_id="HumanEval/0",
        entry_point_name="add",
        gt_solution="def add(a, b):\n    return a + b\n",
    )
    assert task.task_id == "HumanEval/0"
    assert task.entry_point_name == "add"
    assert task.dataset == CompressibleDataset.HUMAN_EVAL_PLUS
