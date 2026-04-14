from __future__ import annotations

import subprocess

from typer.testing import CliRunner

from nl_code.test_cli import app

runner = CliRunner()


def test_docker_command_runs_pytest_with_marker(monkeypatch) -> None:
    captured_cmd: list[str] = []

    def fake_run(cmd: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        nonlocal captured_cmd
        captured_cmd = cmd
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr("nl_code.test_cli.subprocess.run", fake_run)

    result = runner.invoke(app, ["docker", "-q", "tests/test_execution_runner.py"])

    assert result.exit_code == 0
    assert captured_cmd[:7] == [
        captured_cmd[0],
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "-m",
        "docker",
    ]
    assert captured_cmd[7:] == ["-q", "tests/test_execution_runner.py"]
