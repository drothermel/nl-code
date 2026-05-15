from __future__ import annotations

import os
from pathlib import Path

import typer

from nl_code.optim.dspy_generators import (
    DEFAULT_DSPY_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_REASONING_EFFORT,
)
from nl_code.optim.humaneval_dspy_optimize import (
    EncDecOptimizationTarget,
    SplitTaskIds,
    api_key_from_env,
    normalize_auto,
    optimize_encoder_decoder_generation,
    parse_task_ids,
    require_task_ids,
)


def main(
    optimize_target: EncDecOptimizationTarget = typer.Option(
        EncDecOptimizationTarget.BOTH,
        "--optimize-target",
        case_sensitive=False,
        help="Encoder-decoder prompt target to optimize.",
    ),
    train_task_ids: list[str] | None = typer.Option(
        None,
        "--train-task-ids",
        help="Train split task IDs. Can be repeated or comma-separated.",
    ),
    dev_task_ids: list[str] | None = typer.Option(
        None,
        "--dev-task-ids",
        help="Dev split task IDs. Can be repeated or comma-separated.",
    ),
    eval_task_ids: list[str] | None = typer.Option(
        None,
        "--eval-task-ids",
        help="Eval split task IDs. Can be repeated or comma-separated.",
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
        Path("logs/dspy_optimized"),
        "--output-dir",
        help="Directory for optimized program and summary logs.",
    ),
    auto: str | None = typer.Option(
        "medium",
        "--auto",
        help="MIPROv2 auto budget: light, medium, heavy, or none.",
    ),
    num_threads: int | None = typer.Option(
        8,
        "--num-threads",
        min=1,
        help="Number of parallel MIPRO/evaluation workers.",
    ),
    seed: int = typer.Option(42, "--seed", help="MIPRO and sampling seed."),
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
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet",
        help="Print detailed progress for each phase and metric call.",
    ),
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    task_ids = SplitTaskIds(
        train=parse_task_ids(train_task_ids),
        dev=parse_task_ids(dev_task_ids),
        eval=parse_task_ids(eval_task_ids),
    )
    try:
        require_task_ids(task_ids)
        api_key = api_key_from_env()
        auto_mode = normalize_auto(auto)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    run = optimize_encoder_decoder_generation(
        task_ids=task_ids,
        target=optimize_target,
        model=model,
        api_key=api_key,
        api_base=api_base,
        reasoning_effort=reasoning_effort,
        output_dir=output_dir,
        auto=auto_mode,
        num_threads=num_threads,
        seed=seed,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        verbose=verbose,
    )
    typer.echo(f"Optimized program: {run.summary.optimized_program_path}")
    typer.echo(f"Summary: {run.summary.summary_path}")


if __name__ == "__main__":
    typer.run(main)
