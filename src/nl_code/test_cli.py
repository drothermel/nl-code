from __future__ import annotations

import subprocess
import sys

import typer

app = typer.Typer(help="Test runner helpers.")


@app.command(
    "docker",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def docker_tests(ctx: typer.Context) -> None:
    """Run the Docker-marked integration tests."""

    pytest_args = list(ctx.args)
    if pytest_args[:1] == ["docker"]:
        pytest_args = pytest_args[1:]
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-m",
        "docker",
        *pytest_args,
    ]
    completed = subprocess.run(cmd, check=False)  # noqa: S603
    raise typer.Exit(completed.returncode)


if __name__ == "__main__":
    app()
