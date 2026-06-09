from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

from nl_code.optim.humaneval_dspy_logs import (
    parse_humaneval_dspy_logs,
    write_humaneval_dspy_log_snapshot,
)


def main(
    logs_dir: Path = typer.Option(
        Path("logs"),
        "--logs-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing HumanEval DSPy eval and generation logs.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output-path",
        help="Snapshot JSON path. Defaults to logs/human_eval_dspy_snapshot_<timestamp>.json.",
    ),
) -> None:
    snapshot = parse_humaneval_dspy_logs(logs_dir)
    if output_path is None:
        timestamp = datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = logs_dir / f"human_eval_dspy_snapshot_{timestamp}.json"
    written_path = write_humaneval_dspy_log_snapshot(snapshot, output_path)

    typer.echo(f"Snapshot: {written_path}")
    typer.echo(f"Log files: {len(snapshot.log_files)}")
    typer.echo(f"Pipelines: {len(snapshot.pipelines)}")
    typer.echo(f"Runs: {len(snapshot.runs)}")
    typer.echo(f"Attempts: {len(snapshot.attempts)}")
    typer.echo(f"Failed attempts: {len(snapshot.failed_attempts)}")
    typer.echo(f"Generation calls: {len(snapshot.generation_calls)}")


if __name__ == "__main__":
    typer.run(main)
