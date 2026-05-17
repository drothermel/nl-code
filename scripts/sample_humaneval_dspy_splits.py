from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_DIRECT_LOG = Path("logs/human_eval_dspy_direct_eval_20260515T051718Z.json")
DEFAULT_ENCDEC_LOG = Path("logs/human_eval_dspy_encdec_eval_20260515T052743Z.json")


class SplitTargets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    both_fail: int
    direct_only_fail: int
    encdec_only_fail: int
    passed_both: int

    @model_validator(mode="after")
    def validate_counts(self) -> SplitTargets:
        for name, value in self.model_dump().items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        return self


class TaskResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_index: int
    task_id: str
    failed: bool


class SplitSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: list[str] = Field(default_factory=list)
    dev: list[str] = Field(default_factory=list)
    eval: list[str] = Field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return sort_task_ids(set(self.train) | set(self.dev) | set(self.eval))


SPLIT_TARGETS = {
    "train": SplitTargets(
        both_fail=4,
        direct_only_fail=6,
        encdec_only_fail=2,
        passed_both=4,
    ),
    "dev": SplitTargets(
        both_fail=2,
        direct_only_fail=4,
        encdec_only_fail=2,
        passed_both=4,
    ),
    "eval": SplitTargets(
        both_fail=3,
        direct_only_fail=6,
        encdec_only_fail=2,
        passed_both=30,
    ),
}


def main(
    seed: int = typer.Option(42, "--seed", help="Random seed for split sampling."),
    direct_log: Path = typer.Option(
        DEFAULT_DIRECT_LOG,
        "--direct-log",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Direct-generation full eval JSON log.",
    ),
    encdec_log: Path = typer.Option(
        DEFAULT_ENCDEC_LOG,
        "--encdec-log",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Encoder-decoder full eval JSON log.",
    ),
    eval_passed_both: int = typer.Option(
        SPLIT_TARGETS["eval"].passed_both,
        "--eval-passed-both",
        min=0,
        help="Number of passed-by-both tasks to sample into the eval split.",
    ),
) -> None:
    direct_results = load_task_results(direct_log)
    encdec_results = load_task_results(encdec_log)
    common_task_ids = set(direct_results) & set(encdec_results)

    buckets = build_buckets(
        common_task_ids=common_task_ids,
        direct_results=direct_results,
        encdec_results=encdec_results,
    )
    split_targets = SPLIT_TARGETS | {
        "eval": SPLIT_TARGETS["eval"].model_copy(
            update={"passed_both": eval_passed_both}
        )
    }
    sample = sample_splits(buckets, split_targets, seed=seed)

    print_summary(buckets, sample)
    print_csv_lists(sample)


def load_task_results(path: Path) -> dict[str, TaskResult]:
    data = json.loads(path.read_text())
    results = {}
    for item in data["outputs"]:
        output = item["output"]
        task_id = output["task_id"]
        results[task_id] = TaskResult(
            dataset_index=item["dataset_index"],
            task_id=task_id,
            failed=output.get("skipped") is True or output["pass_rate"] < 1,
        )
    return results


def build_buckets(
    *,
    common_task_ids: set[str],
    direct_results: dict[str, TaskResult],
    encdec_results: dict[str, TaskResult],
) -> dict[str, list[str]]:
    direct_failed = {
        task_id for task_id in common_task_ids if direct_results[task_id].failed
    }
    encdec_failed = {
        task_id for task_id in common_task_ids if encdec_results[task_id].failed
    }
    either_failed = direct_failed | encdec_failed
    return {
        "both_fail": sort_task_ids(direct_failed & encdec_failed),
        "direct_only_fail": sort_task_ids(direct_failed - encdec_failed),
        "encdec_only_fail": sort_task_ids(encdec_failed - direct_failed),
        "passed_both": sort_task_ids(common_task_ids - either_failed),
    }


def sample_splits(
    buckets: dict[str, list[str]],
    split_targets: dict[str, SplitTargets],
    *,
    seed: int,
) -> SplitSample:
    rng = random.Random(seed)
    sampled_by_bucket = {
        bucket_name: shuffled(task_ids, rng)
        for bucket_name, task_ids in buckets.items()
    }
    validate_targets(sampled_by_bucket, split_targets)

    split_values: dict[str, list[str]] = {"train": [], "dev": [], "eval": []}
    for bucket_name, shuffled_task_ids in sampled_by_bucket.items():
        offset = 0
        for split_name in ("train", "dev", "eval"):
            count = getattr(split_targets[split_name], bucket_name)
            split_values[split_name].extend(shuffled_task_ids[offset : offset + count])
            offset += count

    return SplitSample(
        train=sort_task_ids(split_values["train"]),
        dev=sort_task_ids(split_values["dev"]),
        eval=sort_task_ids(split_values["eval"]),
    )


def validate_targets(
    buckets: dict[str, list[str]],
    split_targets: dict[str, SplitTargets],
) -> None:
    for bucket_name, task_ids in buckets.items():
        requested = sum(
            getattr(split_target, bucket_name)
            for split_target in split_targets.values()
        )
        available = len(task_ids)
        if requested > available:
            raise typer.BadParameter(
                f"requested {requested} {bucket_name} tasks, "
                f"but only {available} are available"
            )


def shuffled(values: list[str], rng: random.Random) -> list[str]:
    copied = list(values)
    rng.shuffle(copied)
    return copied


def print_summary(buckets: dict[str, list[str]], sample: SplitSample) -> None:
    typer.echo("Bucket counts:")
    for bucket_name, task_ids in buckets.items():
        typer.echo(f"  {bucket_name}: {len(task_ids)}")
    typer.echo("Split counts:")
    for split_name in ("train", "dev", "eval", "all"):
        task_ids = getattr(sample, split_name)
        typer.echo(f"  {split_name}: {len(task_ids)}")
    typer.echo("Split bucket counts:")
    for split_name in ("train", "dev", "eval"):
        counts = split_bucket_counts(getattr(sample, split_name), buckets)
        typer.echo(
            f"  {split_name}: "
            f"both_fail={counts['both_fail']}, "
            f"direct_only_fail={counts['direct_only_fail']}, "
            f"encdec_only_fail={counts['encdec_only_fail']}, "
            f"passed_both={counts['passed_both']}"
        )


def split_bucket_counts(
    split_task_ids: list[str],
    buckets: dict[str, list[str]],
) -> dict[str, int]:
    split_task_id_set = set(split_task_ids)
    return {
        bucket_name: len(split_task_id_set & set(bucket_task_ids))
        for bucket_name, bucket_task_ids in buckets.items()
    }


def print_csv_lists(sample: SplitSample) -> None:
    typer.echo("CSV task-id lists:")
    for split_name in ("all", "train", "dev", "eval"):
        task_ids = getattr(sample, split_name)
        typer.echo(f"{split_name}={','.join(task_ids)}")


def sort_task_ids(task_ids: set[str] | list[str]) -> list[str]:
    return sorted(task_ids, key=task_number)


def task_number(task_id: str) -> int:
    return int(task_id.rsplit("/", maxsplit=1)[-1])


if __name__ == "__main__":
    typer.run(main)
