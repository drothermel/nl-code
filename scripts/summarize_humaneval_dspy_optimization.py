from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel, ConfigDict, Field


SUMMARY_SUFFIX = "_summary.json"
TIMESTAMP_RE = re.compile(r"(?P<timestamp>\d{8}T\d{6}Z)")
PROGRAM_RE = re.compile(r"program_(?P<trial>-?\d+)(?:_(?P<note>[^.]+))?\.json$")
TRIAL_RE = re.compile(
    r"(?:==|=====) Trial (?P<trial>\d+) / .*?(?P<kind>Minibatch|Full Evaluation)?"
)
DEFAULT_SCORE_RE = re.compile(r"Default program score: (?P<score>-?\d+(?:\.\d+)?)")
SCORE_RE = re.compile(r"Score: (?P<score>-?\d+(?:\.\d+)?)")


class SplitScore(BaseModel):
    model_config = ConfigDict(extra="ignore")

    task_count: int
    average_pass_rate: float
    full_pass_count: int
    full_pass_rate: float


class OptimizationSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    timestamp: str
    generation_type: str
    optimization_target: str | None = None
    model: str
    auto: str | None = None
    num_threads: int | None = None
    seed: int
    baseline_scores: dict[str, SplitScore]
    optimized_scores: dict[str, SplitScore]
    optimized_program_path: Path
    summary_path: Path | None = None
    run_log_path: Path | None = None
    event_log_path: Path | None = None


class PromptCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trial_id: int
    path: Path
    prompts: dict[str, str] = Field(default_factory=dict)
    score: float | None = None
    score_label: str | None = None


def main(
    log_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Optimization summary JSON or optimized-program JSON.",
    ),
    mipro_log_dir: Path | None = typer.Option(
        None,
        "--mipro-log-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="DSPy MIPRO log directory containing evaluated_programs/.",
    ),
    run_log: Path | None = typer.Option(
        None,
        "--run-log",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Optional captured stdout/stderr log to recover candidate scores.",
    ),
    prompt_chars: int = typer.Option(
        500,
        "--prompt-chars",
        min=0,
        help="Maximum prompt characters in the candidate table. Use 0 for full prompts.",
    ),
) -> None:
    summary_path = resolve_summary_path(log_path)
    summary = load_summary(summary_path)
    optimized_program_path = resolve_existing_path(
        summary.optimized_program_path,
        fallback_dir=summary_path.parent,
    )

    resolved_run_log = run_log or resolve_optional_path(
        summary.run_log_path,
        fallback_dir=summary_path.parent,
    )
    score_lookup = parse_run_log_scores(resolved_run_log) if resolved_run_log else {}
    inferred_mipro_log_dir = mipro_log_dir or infer_mipro_log_dir(
        summary_path=summary_path,
        optimized_program_path=optimized_program_path,
    )
    candidates = load_prompt_candidates(
        inferred_mipro_log_dir,
        score_lookup=score_lookup,
    )
    final_prompts = load_program_prompts(optimized_program_path)
    initial_prompts = candidates[0].prompts if candidates else {}

    print_summary_header(
        summary, summary_path, optimized_program_path, resolved_run_log
    )
    print_score_table(summary)
    print_candidate_table(candidates, inferred_mipro_log_dir, prompt_chars)
    print_prompt_block("Initial prompt", initial_prompts)
    print_prompt_block("Final prompt", final_prompts)

    if candidates and all(candidate.score is None for candidate in candidates):
        typer.echo(
            "\nNote: candidate-level scores were not found in these JSON logs. "
            "Pass a captured terminal log with --run-log if you have one."
        )


def resolve_summary_path(log_path: Path) -> Path:
    if log_path.name.endswith(SUMMARY_SUFFIX):
        return log_path
    candidate = log_path.with_name(f"{log_path.stem}{SUMMARY_SUFFIX}")
    if candidate.exists():
        return candidate
    raise typer.BadParameter(
        f"{log_path} is not a summary file and {candidate} does not exist"
    )


def load_summary(path: Path) -> OptimizationSummary:
    data = load_json(path)
    if not isinstance(data, dict):
        raise typer.BadParameter(f"{path} does not contain a JSON object")
    try:
        return OptimizationSummary.model_validate(data)
    except ValueError as exc:
        raise typer.BadParameter(
            f"{path} is not an optimization summary: {exc}"
        ) from exc


def resolve_existing_path(path: Path, *, fallback_dir: Path) -> Path:
    if path.exists():
        return path
    fallback = fallback_dir / path.name
    if fallback.exists():
        return fallback
    return path


def resolve_optional_path(path: Path | None, *, fallback_dir: Path) -> Path | None:
    if path is None:
        return None
    resolved = resolve_existing_path(path, fallback_dir=fallback_dir)
    return resolved if resolved.exists() else None


def infer_mipro_log_dir(
    *,
    summary_path: Path,
    optimized_program_path: Path,
) -> Path | None:
    base_dir = optimized_program_path.parent
    root = base_dir / "mipro_logs"
    if not root.exists():
        return None

    summary_timestamp = timestamp_from_path(summary_path)
    final_prompts = canonical_prompts(load_program_prompts(optimized_program_path))
    matching_dirs = []
    fallback_dirs = []
    for candidate_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        timestamp = timestamp_from_path(candidate_dir)
        if summary_timestamp and timestamp and timestamp > summary_timestamp:
            continue
        if (candidate_dir / "evaluated_programs").exists():
            fallback_dirs.append(candidate_dir)
        if directory_contains_prompts(candidate_dir, final_prompts):
            matching_dirs.append(candidate_dir)

    if matching_dirs:
        return matching_dirs[-1]
    if fallback_dirs:
        return fallback_dirs[-1]
    return None


def timestamp_from_path(path: Path) -> str | None:
    match = TIMESTAMP_RE.search(str(path))
    return match.group("timestamp") if match else None


def directory_contains_prompts(
    directory: Path, final_prompts: tuple[tuple[str, str], ...]
) -> bool:
    if not final_prompts:
        return False
    for program_path in evaluated_program_paths(directory):
        if canonical_prompts(load_program_prompts(program_path)) == final_prompts:
            return True
    return False


def load_prompt_candidates(
    mipro_log_dir: Path | None,
    *,
    score_lookup: dict[str, tuple[float, str]],
) -> list[PromptCandidate]:
    if mipro_log_dir is None:
        return []
    candidates = []
    for program_path in evaluated_program_paths(mipro_log_dir):
        trial_id = trial_id_from_program_path(program_path)
        score, score_label = score_lookup.get(
            program_path.stem,
            score_lookup.get(f"program_{trial_id}", (None, None)),
        )
        candidates.append(
            PromptCandidate(
                trial_id=trial_id,
                path=program_path,
                prompts=load_program_prompts(program_path),
                score=score,
                score_label=score_label,
            )
        )
    return candidates


def evaluated_program_paths(mipro_log_dir: Path) -> list[Path]:
    evaluated_dir = mipro_log_dir / "evaluated_programs"
    if not evaluated_dir.exists():
        return []
    return sorted(
        evaluated_dir.glob("program_*.json"),
        key=lambda path: (trial_id_from_program_path(path), path.name),
    )


def trial_id_from_program_path(path: Path) -> int:
    match = PROGRAM_RE.match(path.name)
    if not match:
        return 0
    return int(match.group("trial"))


def load_program_prompts(path: Path) -> dict[str, str]:
    data = load_json(path)
    if not isinstance(data, dict):
        return {}
    prompts = {}
    for name, value in data.items():
        if name == "metadata" or not isinstance(value, dict):
            continue
        signature = value.get("signature")
        if isinstance(signature, dict) and isinstance(
            signature.get("instructions"), str
        ):
            prompts[name] = signature["instructions"]
    return prompts


def canonical_prompts(prompts: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(prompts.items()))


def parse_run_log_scores(path: Path) -> dict[str, tuple[float, str]]:
    lookup: dict[str, tuple[float, str]] = {}
    current_trial: int | None = None
    current_kind = "trial"
    for line in path.read_text(encoding="utf-8").splitlines():
        default_match = DEFAULT_SCORE_RE.search(line)
        if default_match:
            lookup["program_-1"] = (float(default_match.group("score")), "full")
            continue

        trial_match = TRIAL_RE.search(line)
        if trial_match:
            current_trial = int(trial_match.group("trial"))
            current_kind = "mb" if trial_match.group("kind") == "Minibatch" else "full"
            continue

        score_match = SCORE_RE.search(line)
        if score_match and current_trial is not None:
            lookup[f"program_{current_trial}"] = (
                float(score_match.group("score")),
                current_kind,
            )
    return lookup


def print_summary_header(
    summary: OptimizationSummary,
    summary_path: Path,
    optimized_program_path: Path,
    run_log_path: Path | None,
) -> None:
    target = (
        f" target={summary.optimization_target}" if summary.optimization_target else ""
    )
    typer.echo("Optimization summary")
    typer.echo(f"  Summary: {summary_path}")
    typer.echo(f"  Program: {optimized_program_path}")
    if run_log_path is not None:
        typer.echo(f"  Run log: {run_log_path}")
    if summary.event_log_path is not None:
        typer.echo(f"  Event log: {summary.event_log_path}")
    typer.echo(
        f"  Run: {summary.generation_type}{target} model={summary.model} "
        f"auto={summary.auto} seed={summary.seed} threads={summary.num_threads}"
    )


def print_score_table(summary: OptimizationSummary) -> None:
    typer.echo("\nInitial vs final performance")
    typer.echo(
        "  split   tasks  initial avg  final avg  delta avg  initial full  final full  delta full"
    )
    for split_name in ("train", "dev", "eval"):
        baseline = summary.baseline_scores.get(split_name)
        optimized = summary.optimized_scores.get(split_name)
        if baseline is None or optimized is None:
            continue
        typer.echo(
            "  "
            f"{split_name:<6} "
            f"{baseline.task_count:>5}  "
            f"{format_percent(baseline.average_pass_rate):>11}  "
            f"{format_percent(optimized.average_pass_rate):>9}  "
            f"{format_delta(optimized.average_pass_rate - baseline.average_pass_rate):>9}  "
            f"{format_full_score(baseline):>12}  "
            f"{format_full_score(optimized):>10}  "
            f"{format_delta(optimized.full_pass_rate - baseline.full_pass_rate):>10}"
        )


def print_candidate_table(
    candidates: list[PromptCandidate],
    mipro_log_dir: Path | None,
    prompt_chars: int,
) -> None:
    typer.echo("\nPrompt candidates")
    if not candidates:
        typer.echo("  No MIPRO evaluated_programs directory found.")
        return
    typer.echo(f"  MIPRO log: {mipro_log_dir}")
    typer.echo("  trial  score        module           prompt")
    for candidate in candidates:
        score = format_score(candidate)
        for index, (module, prompt) in enumerate(candidate.prompts.items()):
            trial = trial_label(candidate.trial_id) if index == 0 else ""
            score_text = score if index == 0 else ""
            typer.echo(
                f"  {trial:<5}  {score_text:<11}  {module:<15}  "
                f"{one_line(prompt, prompt_chars)}"
            )


def print_prompt_block(title: str, prompts: dict[str, str]) -> None:
    typer.echo(f"\n{title}")
    if not prompts:
        typer.echo("  n/a")
        return
    for module, prompt in prompts.items():
        typer.echo(f"\n[{module}]")
        typer.echo(prompt)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f}pp"


def format_full_score(score: SplitScore) -> str:
    return f"{score.full_pass_count}/{score.task_count} ({format_percent(score.full_pass_rate)})"


def format_score(candidate: PromptCandidate) -> str:
    if candidate.score is None:
        return "n/a"
    label = f" {candidate.score_label}" if candidate.score_label else ""
    return f"{candidate.score:.4f}{label}"


def trial_label(trial_id: int) -> str:
    return "init" if trial_id < 0 else str(trial_id)


def one_line(value: str, max_chars: int) -> str:
    collapsed = " ".join(value.split())
    if max_chars == 0 or len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3]}..."


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    typer.run(main)
