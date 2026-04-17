import logging
import os

from contextlib import contextmanager
from typing import Any, ClassVar, Generator, Self

from datasets import load_dataset
from pydantic import BaseModel, Field, ValidationError

from nl_code.datasets.cache import (
    ParsedDatasetSnapshot,
    build_manifest,
    read_snapshot,
    write_snapshot,
)
from nl_code.datasets.task import CodeDataset, Task

logger = logging.getLogger(__name__)


class FlawedSample(BaseModel):
    """A raw sample that failed validation."""

    error: str
    raw_input: dict[str, Any]


class DatasetCacheMissError(FileNotFoundError):
    """Raised when a parsed dataset cache entry is unavailable."""


@contextmanager
def _hf_offline_mode(enabled: bool) -> Generator[None, None, None]:
    if not enabled:
        yield
        return

    original_values = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for key, value in original_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class Dataset(BaseModel):
    dataset_key: ClassVar[str]
    raw_model_type: ClassVar[type[BaseModel]]
    source_revision: ClassVar[str | None] = None

    dataset_id: CodeDataset
    split: str = "test"
    raw_samples: dict[str, BaseModel] = Field(default_factory=dict)
    flawed_raw_samples: dict[str, FlawedSample] = Field(default_factory=dict)
    tasks: dict[str, Task] = Field(default_factory=dict)

    def _parse_row(self, row: dict[str, Any]) -> BaseModel:
        raise NotImplementedError

    def _extract_task_id(self, row: dict[str, Any]) -> str:
        raise NotImplementedError

    def _to_task(self, task_id: str, raw: BaseModel) -> Task:
        raise NotImplementedError

    def _verify_ground_truth_samples(
        self,
        raw_samples: dict[str, BaseModel],
        raw_inputs: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, BaseModel], dict[str, FlawedSample]]:
        return raw_samples, {}

    def _dataset_failure(
        self,
        *,
        detail: str,
        raw_input: dict[str, Any],
    ) -> FlawedSample:
        return FlawedSample(error=f"dataset_failure: {detail}", raw_input=raw_input)

    def _docker_failure(
        self,
        *,
        detail: str,
        raw_input: dict[str, Any],
    ) -> FlawedSample:
        return FlawedSample(error=f"docker_failure: {detail}", raw_input=raw_input)

    def _mark_raw_validated(self, raw: BaseModel) -> BaseModel:
        if "validated" not in type(raw).model_fields:
            return raw
        return raw.model_copy(update={"validated": True})

    def _normalize_index(self, index: int, size: int, *, collection_name: str) -> int:
        normalized_index = index + size if index < 0 else index
        if normalized_index < 0 or normalized_index >= size:
            raise IndexError(
                f"{collection_name} index {index} out of range for {size} items"
            )
        return normalized_index

    def get_task_at_index(self, index: int) -> Task:
        normalized_index = self._normalize_index(
            index, len(self.tasks), collection_name="task"
        )
        task_id = list(self.tasks)[normalized_index]
        return self.tasks[task_id]

    def get_raw_sample_at_index(self, index: int) -> BaseModel:
        normalized_index = self._normalize_index(
            index, len(self.raw_samples), collection_name="raw sample"
        )
        task_id = list(self.raw_samples)[normalized_index]
        return self.raw_samples[task_id]

    def load(self, hf_id: str | None = None, *, force_reparse: bool = False) -> Self:
        if not force_reparse:
            if hf_id is not None:
                raise ValueError("hf_id can only be used when force_reparse=True")
            snapshot = read_snapshot(self.dataset_id, self.split)
            if snapshot is not None:
                self._restore_from_snapshot(snapshot)
                return self

            logger.info(
                "Parsed dataset cache not found for %s (split=%s); rebuilding",
                self.dataset_id.value,
                self.split,
            )
            return self.rebuild_cache()

        return self.rebuild_cache(hf_id=hf_id)

    def rebuild_cache(self, *, hf_id: str | None = None, offline: bool = False) -> Self:
        effective_id = hf_id or self.dataset_id.value
        revision = self._source_revision_for(hf_id)
        with _hf_offline_mode(offline):
            rows = load_dataset(effective_id, split=self.split, revision=revision)
        self._build_from_rows(rows, effective_id=effective_id)
        snapshot = ParsedDatasetSnapshot(
            manifest=build_manifest(
                dataset_id=self.dataset_id,
                split=self.split,
                source_revision=revision,
                raw_sample_count=len(self.raw_samples),
                flawed_count=len(self.flawed_raw_samples),
                task_count=len(self.tasks),
            ),
            raw_samples={
                task_id: raw.model_dump(mode="json")
                for task_id, raw in self.raw_samples.items()
            },
            flawed_raw_samples={
                task_id: flawed.model_dump(mode="json")
                for task_id, flawed in self.flawed_raw_samples.items()
            },
            tasks={
                task_id: task.model_dump(mode="json")
                for task_id, task in self.tasks.items()
            },
        )
        cache_dir = write_snapshot(snapshot)
        logger.info(
            "Wrote parsed dataset cache for %s (split=%s) to %s",
            self.dataset_id.value,
            self.split,
            cache_dir,
        )
        return self

    def _build_from_rows(self, rows: Any, *, effective_id: str) -> None:
        total_rows = len(rows)
        next_progress_pct = 10

        logger.info(
            "Loading tasks from %s (split=%s, total=%d)",
            effective_id,
            self.split,
            total_rows,
        )

        raw_samples: dict[str, BaseModel] = {}
        flawed_raw_samples: dict[str, FlawedSample] = {}
        raw_inputs: dict[str, dict[str, Any]] = {}

        for index, row in enumerate(rows, start=1):
            row_dict = dict(row)
            try:
                task_id = self._extract_task_id(row_dict)
            except (KeyError, TypeError):
                task_id = f"row-{index}"

            try:
                raw_samples[task_id] = self._parse_row(row_dict)
                raw_inputs[task_id] = row_dict
            except (
                ValidationError,
                KeyError,
                TypeError,
                ValueError,
                SyntaxError,
            ) as exc:
                flawed_raw_samples[task_id] = FlawedSample(
                    error=str(exc),
                    raw_input=row_dict,
                )
                logger.warning("Skipping flawed task %s: %s", task_id, exc)

            if total_rows > 0:
                progress_pct = index * 100 // total_rows
                while progress_pct >= next_progress_pct and next_progress_pct <= 100:
                    logger.info(
                        "Loaded %d/%d tasks (%d%%)",
                        index,
                        total_rows,
                        next_progress_pct,
                    )
                    next_progress_pct += 10

        verified_raw_samples, gt_flawed_samples = self._verify_ground_truth_samples(
            raw_samples,
            raw_inputs,
        )
        flawed_raw_samples.update(gt_flawed_samples)

        self.raw_samples = verified_raw_samples
        self.flawed_raw_samples = flawed_raw_samples

        tasks: dict[str, Task] = {}
        for task_id, raw in verified_raw_samples.items():
            try:
                task = self._to_task(task_id, raw)
                task.validate_raw_task_version(raw)
                tasks[task_id] = task
            except Exception as exc:
                flawed_raw_samples[task_id] = FlawedSample(
                    error=f"_to_task failed: {exc}",
                    raw_input=raw_inputs.get(task_id, {"task_id": task_id}),
                )
                logger.warning("Failed to convert task %s: %s", task_id, exc)
        self.tasks = tasks

        logger.info(
            "Finished loading from %s: %d valid, %d flawed, %d tasks",
            effective_id,
            len(self.raw_samples),
            len(self.flawed_raw_samples),
            len(self.tasks),
        )

    def _restore_from_snapshot(self, snapshot: ParsedDatasetSnapshot) -> None:
        self.raw_samples = {
            task_id: self.raw_model_type.model_validate(raw_input)
            for task_id, raw_input in snapshot.raw_samples.items()
        }
        self.flawed_raw_samples = {
            task_id: FlawedSample.model_validate(flawed)
            for task_id, flawed in snapshot.flawed_raw_samples.items()
        }
        self.tasks = {
            task_id: Task.model_validate(task)
            for task_id, task in snapshot.tasks.items()
        }
        logger.info(
            "Loaded parsed dataset cache for %s (split=%s): %d valid, %d flawed, %d tasks",
            self.dataset_id.value,
            self.split,
            len(self.raw_samples),
            len(self.flawed_raw_samples),
            len(self.tasks),
        )

    def _source_revision_for(self, hf_id: str | None) -> str | None:
        if hf_id is None or hf_id == self.dataset_id.value:
            return self.source_revision
        return None
