from __future__ import annotations

import os
from pathlib import Path

import typer

from nl_code.optim.dspy_generators import (
    DEFAULT_DSPY_MODEL,
    DEFAULT_OPENROUTER_BASE_URL,
    DEFAULT_REASONING_EFFORT,
    resolve_openrouter_llm_config,
)
from nl_code.optim.humaneval_dspy_gepa import optimize_direct_generation_gepa
from nl_code.optim.humaneval_dspy_optimize import (
    SplitTaskIds,
    api_key_from_env,
    normalize_auto,
    optimization_artifact_paths,
    optimization_log_context,
    parse_task_ids,
    require_task_ids,
)


def main(
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
        help="DSPy/LiteLLM model name used for task and reflection calls. Overridden by --llm-config-id.",
    ),
    llm_config_id: str | None = typer.Option(
        None,
        "--llm-config-id",
        help="Supported OpenRouter catalog config id. Overrides --model and --reasoning-effort.",
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
    run_log_path: Path | None = typer.Option(
        None,
        "--run-log-path",
        help="Stdout/stderr log path. Defaults to the run artifact namespace.",
    ),
    event_log_path: Path | None = typer.Option(
        None,
        "--event-log-path",
        help="Structured JSONL event log path. Defaults to the run artifact namespace.",
    ),
    auto: str | None = typer.Option(
        "light",
        "--auto",
        help="GEPA auto budget: light, medium, or heavy. Ignored when --max-metric-calls is set.",
    ),
    max_metric_calls: int | None = typer.Option(
        None,
        "--max-metric-calls",
        min=1,
        help="Explicit GEPA metric-call budget. Useful for smoke tests; overrides --auto.",
    ),
    num_threads: int | None = typer.Option(
        8,
        "--num-threads",
        min=1,
        help="Number of parallel GEPA/evaluation workers.",
    ),
    seed: int = typer.Option(42, "--seed", help="GEPA seed."),
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
        auto_mode = None if max_metric_calls is not None else normalize_auto(auto)
        reasoning_config = None
        if llm_config_id:
            lm_catalog_config = resolve_openrouter_llm_config(llm_config_id)
            model = lm_catalog_config.model
            reasoning_effort = None
            reasoning_config = lm_catalog_config.reasoning
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if auto_mode is None and max_metric_calls is None:
        raise typer.BadParameter(
            "Either --auto or --max-metric-calls is required for GEPA"
        )

    artifacts = optimization_artifact_paths(
        output_dir=output_dir,
        generation_type="direct_gepa",
        optimization_target=None,
    )
    run_log_path = run_log_path or artifacts.run_log_path
    event_log_path = event_log_path or artifacts.event_log_path

    with optimization_log_context(
        run_log_path=run_log_path,
        event_log_path=event_log_path,
    ):
        run = optimize_direct_generation_gepa(
            task_ids=task_ids,
            model=model,
            api_key=api_key,
            api_base=api_base,
            reasoning_effort=reasoning_effort,
            reasoning_config=reasoning_config,
            llm_config_id=llm_config_id,
            output_dir=output_dir,
            auto=auto_mode,
            max_metric_calls=max_metric_calls,
            num_threads=num_threads,
            seed=seed,
            timeout_seconds=timeout_seconds,
            docker_image=docker_image,
            verbose=verbose,
            artifact_stem=artifacts.stem,
            run_log_path=run_log_path,
            event_log_path=event_log_path,
        )
        typer.echo(f"Optimized program: {run.summary.optimized_program_path}")
        typer.echo(f"Summary: {run.summary.summary_path}")
        typer.echo(f"Run log: {run_log_path}")
        typer.echo(f"Event log: {event_log_path}")


if __name__ == "__main__":
    typer.run(main)
