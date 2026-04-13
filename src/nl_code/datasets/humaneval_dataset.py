import logging
from typing import Any

from datasets import load_dataset
from pydantic import BaseModel, Field, ValidationError

from nl_code.datasets.humaneval_task import HumanEvalTask, RawHumanEvalTask
from nl_code.datasets.task import CompressibleDataset

logger = logging.getLogger(__name__)


class FlawedSample(BaseModel):
    """A raw sample that failed validation."""

    error: str
    raw_input: dict[str, Any]


class HumanEvalDataset(BaseModel):
    dataset_id: str = "evalplus/humanevalplus"
    raw_samples: dict[str, RawHumanEvalTask] = Field(default_factory=dict)
    flawed_raw_samples: dict[str, FlawedSample] = Field(default_factory=dict)
    tasks: dict[str, HumanEvalTask] = Field(default_factory=dict)

    def load_raw_samples(self, dataset_id: str | None = None) -> None:
        if dataset_id is not None:
            self.dataset_id = dataset_id

        rows = load_dataset(self.dataset_id, split="test")
        total_rows = len(rows)
        next_progress_pct = 10

        logger.info(
            "Loading raw HumanEval tasks from %s (split=test, total=%d)",
            self.dataset_id,
            total_rows,
        )

        raw_samples: dict[str, RawHumanEvalTask] = {}
        flawed_raw_samples: dict[str, FlawedSample] = {}
        for index, row in enumerate(rows, start=1):
            try:
                task_id = str(row["task_id"])
                raw_samples[task_id] = RawHumanEvalTask.model_validate(row)
            except (ValidationError, KeyError, TypeError) as exc:
                fallback_id = str(row.get("task_id", f"row-{index}"))
                flawed_raw_samples[fallback_id] = FlawedSample(
                    error=str(exc),
                    raw_input=dict(row),
                )
                logger.warning(
                    "Skipping flawed HumanEval task %s: %s",
                    fallback_id,
                    exc,
                )

            if total_rows > 0:
                progress_pct = index * 100 // total_rows
                while progress_pct >= next_progress_pct and next_progress_pct <= 100:
                    logger.info(
                        "Loaded %d/%d raw HumanEval tasks (%d%%)",
                        index,
                        total_rows,
                        next_progress_pct,
                    )
                    next_progress_pct += 10

        self.raw_samples = raw_samples
        self.flawed_raw_samples = flawed_raw_samples
        self.tasks = {
            task_id: HumanEvalTask(
                dataset=CompressibleDataset.HUMAN_EVAL_PLUS,
                task_id=raw_task.task_id,
                entry_point_name=raw_task.entry_point,
                gt_solution=raw_task.gt_solution_without_comments,
            )
            for task_id, raw_task in self.raw_samples.items()
        }
        logger.info(
            (
                "Finished loading raw HumanEval tasks from %s: "
                "%d valid raw tasks, %d flawed raw tasks, %d derived tasks"
            ),
            self.dataset_id,
            len(self.raw_samples),
            len(self.flawed_raw_samples),
            len(self.tasks),
        )
