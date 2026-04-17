from typing import Any, ClassVar

from pydantic import BaseModel

from nl_code.code_execution.models import AssertionBatchItem
from nl_code.code_execution.runner import batch_run_assertion_tests
from nl_code.datasets.dataset import Dataset
from nl_code.datasets.gt_verification import run_batched_with_infra_isolation
from nl_code.datasets.humaneval_pro_task import RawHumanEvalProTask
from nl_code.datasets.task import CodeDataset, Task


class HumanEvalProDataset(Dataset):
    dataset_key: ClassVar[str] = "humaneval-pro"
    raw_model_type: ClassVar[type[BaseModel]] = RawHumanEvalProTask
    source_revision: ClassVar[str] = "cd078f93d57d1902b5c3e4ae330166b2ca0e0e80"
    dataset_id: CodeDataset = CodeDataset.HUMANEVAL_PRO
    split: str = "train"

    def _parse_row(self, row: dict[str, Any]) -> RawHumanEvalProTask:
        row["task_id"] = f"HumanEvalPro/{row['id']}"
        return RawHumanEvalProTask.model_validate(row)

    def _extract_task_id(self, row: dict[str, Any]) -> str:
        return f"HumanEvalPro/{row['id']}"

    def _verify_ground_truth_samples(
        self,
        raw_samples: dict[str, BaseModel],
        raw_inputs: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, BaseModel], dict[str, Any]]:
        items_by_task_id = [
            (
                task_id,
                AssertionBatchItem(
                    code=raw.gt_solution_with_comments,
                    test_code=raw.source__test_code,
                ),
            )
            for task_id, raw_base in raw_samples.items()
            for raw in [raw_base]
            if isinstance(raw, RawHumanEvalProTask)
        ]

        def run_batch(chunk: list[AssertionBatchItem]) -> list[Any]:
            return batch_run_assertion_tests(chunk, chunk_size=len(chunk))

        results, infra_failures = run_batched_with_infra_isolation(
            items_by_task_id,
            run_batch,
        )

        verified_raw_samples: dict[str, BaseModel] = {}
        flawed_samples: dict[str, Any] = {}
        for task_id, raw_base in raw_samples.items():
            assert isinstance(raw_base, RawHumanEvalProTask)
            raw_input = raw_inputs[task_id]
            infra_error = infra_failures.get(task_id)
            if infra_error is not None:
                flawed_samples[task_id] = self._docker_failure(
                    detail=str(infra_error),
                    raw_input=raw_input,
                )
                continue

            result = results[task_id]
            if not result.passed:
                flawed_samples[task_id] = self._dataset_failure(
                    detail=result.error
                    or "ground-truth solution failed assertion tests",
                    raw_input=raw_input,
                )
                continue

            verified_raw_samples[task_id] = self._mark_raw_validated(raw_base)

        return verified_raw_samples, flawed_samples

    def _to_task(self, task_id: str, raw: BaseModel) -> Task:
        assert isinstance(raw, RawHumanEvalProTask)
        return Task(
            dataset=self.dataset_id,
            task_id=task_id,
            entry_point_name=raw.new_entry_point,
            description=raw.new_description,
            gt_solution=raw.gt_solution,
            version=raw.version,
        )
