import logging
from typing import Any, Self

from datasets import load_dataset
from pydantic import BaseModel, Field, ValidationError

from nl_code.datasets.task import CodeDataset, Task

logger = logging.getLogger(__name__)


class FlawedSample(BaseModel):
    """A raw sample that failed validation."""

    error: str
    raw_input: dict[str, Any]


class Dataset(BaseModel):
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

    def load(self, hf_id: str | None = None) -> Self:
        effective_id = hf_id or self.dataset_id.value
        rows = load_dataset(effective_id, split=self.split)
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

        for index, row in enumerate(rows, start=1):
            try:
                task_id = self._extract_task_id(dict(row))
            except (KeyError, TypeError):
                task_id = f"row-{index}"

            try:
                raw_samples[task_id] = self._parse_row(dict(row))
            except (ValidationError, KeyError, TypeError, ValueError) as exc:
                flawed_raw_samples[task_id] = FlawedSample(
                    error=str(exc),
                    raw_input=dict(row),
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

        self.raw_samples = raw_samples
        self.flawed_raw_samples = flawed_raw_samples

        tasks: dict[str, Task] = {}
        for task_id, raw in raw_samples.items():
            try:
                tasks[task_id] = self._to_task(task_id, raw)
            except Exception as exc:
                flawed_raw_samples[task_id] = FlawedSample(
                    error=f"_to_task failed: {exc}",
                    raw_input={"task_id": task_id},
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

        return self
