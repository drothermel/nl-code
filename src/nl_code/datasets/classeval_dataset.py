from typing import Any, ClassVar

from pydantic import BaseModel

from nl_code.code_execution.models import UnittestBatchItem
from nl_code.code_execution.runner import batch_run_unittest_tests
from nl_code.datasets.classeval_task import RawClassEvalTask
from nl_code.datasets.dataset import Dataset
from nl_code.datasets.gt_verification import run_batched_with_infra_isolation
from nl_code.datasets.task import CodeDataset, Task


class ClassEvalDataset(Dataset):
    dataset_key: ClassVar[str] = "class-eval"
    raw_model_type: ClassVar[type[BaseModel]] = RawClassEvalTask
    source_revision: ClassVar[str] = "fef204b34e221f207f47904ee660bb920d4c5d1d"
    dataset_id: CodeDataset = CodeDataset.CLASS_EVAL

    def _parse_row(self, row: dict[str, Any]) -> RawClassEvalTask:
        return RawClassEvalTask.model_validate(row)

    def _extract_task_id(self, row: dict[str, Any]) -> str:
        return str(row["task_id"])

    def _verify_ground_truth_samples(
        self,
        raw_samples: dict[str, BaseModel],
        raw_inputs: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, BaseModel], dict[str, Any]]:
        items_by_task_id = [
            (
                task_id,
                UnittestBatchItem(
                    code=raw.gt_code_with_comments,
                    test_code=raw.test,
                    test_class_names=raw.test_classes,
                ),
            )
            for task_id, raw_base in raw_samples.items()
            for raw in [raw_base]
            if isinstance(raw, RawClassEvalTask) and raw.auto_fail_reason is None
        ]

        def run_batch(chunk: list[UnittestBatchItem]) -> list[Any]:
            return batch_run_unittest_tests(chunk, chunk_size=len(chunk))

        results, infra_failures = run_batched_with_infra_isolation(
            items_by_task_id,
            run_batch,
        )

        verified_raw_samples: dict[str, BaseModel] = {}
        flawed_samples: dict[str, Any] = {}
        for task_id, raw_base in raw_samples.items():
            assert isinstance(raw_base, RawClassEvalTask)
            raw_input = raw_inputs[task_id]
            if raw_base.auto_fail_reason is not None:
                flawed_samples[task_id] = self._dataset_failure(
                    detail=f"known dataset issue: {raw_base.auto_fail_reason}",
                    raw_input=raw_input,
                )
                continue

            infra_error = infra_failures.get(task_id)
            if infra_error is not None:
                flawed_samples[task_id] = self._docker_failure(
                    detail=str(infra_error),
                    raw_input=raw_input,
                )
                continue

            result = results[task_id]
            if not result.all_passed:
                flawed_samples[task_id] = self._dataset_failure(
                    detail=result.error
                    or (
                        "ground-truth solution failed unittest tests "
                        f"({result.total_tests_failed} failed, "
                        f"{result.total_tests_errored} errored)"
                    ),
                    raw_input=raw_input,
                )
                continue

            verified_raw_samples[task_id] = self._mark_raw_validated(raw_base)

        return verified_raw_samples, flawed_samples

    def _to_task(self, task_id: str, raw: BaseModel) -> Task:
        assert isinstance(raw, RawClassEvalTask)
        return Task(
            dataset=self.dataset_id,
            task_id=task_id,
            entry_point_name=raw.class_name,
            description=raw.class_description,
            gt_solution=raw.gt_code,
        )
