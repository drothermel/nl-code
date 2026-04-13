from nl_code.datasets.bigcodebench_lite_pro_dataset import BigCodeBenchLiteProDataset
from nl_code.datasets.classeval_dataset import ClassEvalDataset
from nl_code.datasets.dataset import Dataset
from nl_code.datasets.humaneval_dataset import HumanEvalDataset
from nl_code.datasets.humaneval_pro_dataset import HumanEvalProDataset
from nl_code.datasets.mbpp_pro_dataset import MbppProDataset

DATASET_TYPES_BY_KEY: dict[str, type[Dataset]] = {
    HumanEvalDataset.dataset_key: HumanEvalDataset,
    HumanEvalProDataset.dataset_key: HumanEvalProDataset,
    MbppProDataset.dataset_key: MbppProDataset,
    BigCodeBenchLiteProDataset.dataset_key: BigCodeBenchLiteProDataset,
    ClassEvalDataset.dataset_key: ClassEvalDataset,
}


def get_dataset_type(dataset_key: str) -> type[Dataset]:
    try:
        return DATASET_TYPES_BY_KEY[dataset_key]
    except KeyError as exc:
        raise KeyError(dataset_key) from exc
