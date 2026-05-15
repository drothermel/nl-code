from __future__ import annotations

import os
from pathlib import Path

import typer

from nl_code.optim.humaneval_dspy_eval import (
    GenerationType,
    HumanEvalDspyEvalConfig,
    HumanEvalDspyEvalSummary,
    run_humaneval_dspy_eval,
)
from nl_code.optim.dspy_generators import (
    DEFAULT_DSPY_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_REASONING_EFFORT,
)


def main(
    generation_type: GenerationType = typer.Option(
        GenerationType.DIRECT,
        "--generation-type",
        case_sensitive=False,
        help="Generation mode to evaluate.",
    ),
    n_samples: int = typer.Option(
        1,
        "--n-samples",
        min=1,
        help="Number of seeded random evaluable samples to run.",
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for sample selection."),
    sample_indices: list[int] | None = typer.Option(
        None,
        "--sample-index",
        help="Dataset index to evaluate. Can be passed multiple times.",
    ),
    task_ids: list[str] | None = typer.Option(
        None,
        "--task-id",
        help=(
            "HumanEval task id to evaluate. Can be passed multiple times or "
            "as comma-separated values."
        ),
    ),
    num_repeats: int = typer.Option(
        1,
        "--num-repeats",
        min=1,
        help="Number of fresh generations per selected sample and generation type.",
    ),
    model: str = typer.Option(
        DEFAULT_DSPY_MODEL,
        "--model",
        help="DSPy/LiteLLM model name.",
    ),
    reasoning_effort: str | None = typer.Option(
        DEFAULT_REASONING_EFFORT,
        "--reasoning-effort",
        help="Reasoning effort value, or 'none' to omit reasoning settings.",
    ),
    api_base: str = typer.Option(
        os.getenv("OPENROUTER_API_BASE", DEFAULT_OPENROUTER_BASE_URL),
        "--api-base",
        help="OpenRouter-compatible API base URL.",
    ),
    output_dir: Path = typer.Option(
        Path("logs"),
        "--output-dir",
        help="Directory for generation and run logs.",
    ),
    timeout_seconds: float = typer.Option(
        30.0,
        "--timeout-seconds",
        min=0.001,
        help="Timeout for each generated-code test run.",
    ),
    docker_image: str | None = typer.Option(
        None,
        "--docker-image",
        help="Optional code execution Docker image override.",
    ),
    log_every: int = typer.Option(
        0,
        "--log-every",
        min=0,
        help="Log progress every N completed attempts. Use 0 to disable.",
    ),
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise typer.BadParameter("OPENROUTER_API_KEY must be set")

    config = HumanEvalDspyEvalConfig(
        generation_type=generation_type,
        n_samples=n_samples,
        seed=seed,
        sample_indices=sample_indices or [],
        task_ids=_parse_task_ids(task_ids),
        num_repeats=num_repeats,
        model=model,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        log_every=log_every,
    )
    run = run_humaneval_dspy_eval(config, api_key=api_key)

    typer.echo(f"Selected dataset indices: {run.selected_dataset_indices}")
    for generation_name, summary in run.summaries.items():
        typer.echo(_format_summary(generation_name, summary))
    typer.echo(f"Run log: {run.run_log_file}")


def _parse_task_ids(task_ids: list[str] | None) -> list[str]:
    if not task_ids:
        return []
    return [
        task_id.strip()
        for task_id_arg in task_ids
        for task_id in task_id_arg.split(",")
        if task_id.strip()
    ]


def _format_summary(
    generation_name: str,
    summary: HumanEvalDspyEvalSummary,
) -> str:
    return (
        f"{generation_name}: attempts={summary.evaluated_attempts}/"
        f"{summary.total_attempts}, "
        f"attempt_pass_rate={summary.attempt_pass_rate:.1%}, "
        f"best_of_n_sample_pass_rate={summary.sample_best_pass_rate:.1%}, "
        f"average_test_pass_rate={summary.average_test_pass_rate:.3%}"
    )


if __name__ == "__main__":
    typer.run(main)
