from nl_code.datasets import (
    BigCodeBenchLiteProDataset,
    ClassEvalDataset,
    Dataset,
    DatasetSlice,
    HumanEvalDataset,
    HumanEvalProDataset,
    MbppProDataset,
    RawBigCodeBenchLiteProTask,
    RawClassEvalTask,
    RawHumanEvalProTask,
    RawHumanEvalTask,
    RawMbppProTask,
    Task,
)


def test_dataset_package_re_exports_task_and_dataset_classes() -> None:
    assert Dataset.__name__ == "Dataset"
    assert DatasetSlice.__name__ == "DatasetSlice"
    assert Task.__name__ == "Task"

    assert HumanEvalDataset.__name__ == "HumanEvalDataset"
    assert HumanEvalProDataset.__name__ == "HumanEvalProDataset"
    assert MbppProDataset.__name__ == "MbppProDataset"
    assert BigCodeBenchLiteProDataset.__name__ == "BigCodeBenchLiteProDataset"
    assert ClassEvalDataset.__name__ == "ClassEvalDataset"

    assert RawHumanEvalTask.__name__ == "RawHumanEvalTask"
    assert RawHumanEvalProTask.__name__ == "RawHumanEvalProTask"
    assert RawMbppProTask.__name__ == "RawMbppProTask"
    assert RawBigCodeBenchLiteProTask.__name__ == "RawBigCodeBenchLiteProTask"
    assert RawClassEvalTask.__name__ == "RawClassEvalTask"
