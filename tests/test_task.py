from nl_code.datasets.task import CodeDataset, Task


def test_code_dataset_humaneval_plus() -> None:
    assert CodeDataset.HUMAN_EVAL_PLUS == "evalplus/humanevalplus"


def test_code_dataset_humaneval_pro() -> None:
    assert CodeDataset.HUMAN_EVAL_PRO == "CodeEval-Pro/humaneval-pro"


def test_code_dataset_mbpp_pro() -> None:
    assert CodeDataset.MBPP_PRO == "CodeEval-Pro/mbpp-pro"


def test_code_dataset_bigcodebench_lite_pro() -> None:
    assert CodeDataset.BIGCODEBENCH_LITE_PRO == "CodeEval-Pro/bigcodebench-lite-pro"


def test_task_construction() -> None:
    task = Task(
        dataset=CodeDataset.HUMAN_EVAL_PLUS,
        task_id="HumanEval/0",
        entry_point_name="add",
        description="Add two integers and return the result.",
        gt_solution="def add(a, b):\n    return a + b\n",
    )
    assert task.task_id == "HumanEval/0"
    assert task.entry_point_name == "add"
    assert task.description == "Add two integers and return the result."
    assert task.dataset == CodeDataset.HUMAN_EVAL_PLUS
