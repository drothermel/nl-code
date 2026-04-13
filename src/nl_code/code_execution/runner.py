import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from nl_code.code_execution.models import ExecutionResult, TestCase, TestCaseResult


class ExecutionError(RuntimeError):
    """Raised when execution infrastructure fails (not code errors)."""

    def __init__(self, *, stage: str, detail: str) -> None:
        self.stage = stage
        self.detail = detail
        super().__init__(f"execution failure (stage={stage}): {detail}")


def _worker_script_path() -> Path:
    return Path(__file__).resolve().parent / "worker.py"


def _run_worker_subprocess(
    request: dict[str, Any],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    worker_path = _worker_script_path()
    if not worker_path.is_file():
        raise ExecutionError(
            stage="worker_missing",
            detail=f"worker script not found at {worker_path}",
        )
    cmd = [sys.executable, "-I", "-S", str(worker_path)]
    request_json = json.dumps(request)
    try:
        proc = subprocess.run(
            cmd,
            input=request_json,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExecutionError(
            stage="timeout",
            detail=f"worker timed out after {timeout_seconds}s",
        ) from exc
    return proc


def _parse_worker_response(proc: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if not proc.stdout.strip():
        raise ExecutionError(
            stage="empty_response",
            detail=f"worker returned no output (returncode={proc.returncode}, stderr={proc.stderr[:200]})",
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ExecutionError(
            stage="json_parse",
            detail=f"invalid JSON from worker: {exc} (stdout={proc.stdout[:200]})",
        ) from exc


def _values_equal(actual: Any, expected: Any, rel_tol: float = 1e-9) -> bool:
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=rel_tol)
    if isinstance(actual, float) and isinstance(expected, int):
        return math.isclose(actual, float(expected), rel_tol=rel_tol)
    if isinstance(actual, int) and isinstance(expected, float):
        return math.isclose(float(actual), expected, rel_tol=rel_tol)
    return actual == expected


def run_function_batch(
    code: str,
    function_name: str,
    input_values: list[Any],
    timeout_seconds: float = 10.0,
) -> list[ExecutionResult]:
    """Execute a function with multiple inputs in an isolated subprocess."""
    if not input_values:
        return []

    request = {
        "code": code,
        "function_name": function_name,
        "input_values": input_values,
    }
    proc = _run_worker_subprocess(request, timeout_seconds)
    response = _parse_worker_response(proc)

    if response.get("error"):
        return [
            ExecutionResult(input_value=iv, error=response["error"])
            for iv in input_values
        ]

    raw_results = response.get("results", [])
    if len(raw_results) != len(input_values):
        raise ExecutionError(
            stage="result_count_mismatch",
            detail=f"expected {len(input_values)} results, got {len(raw_results)}",
        )

    results: list[ExecutionResult] = []
    for iv, raw in zip(input_values, raw_results):
        results.append(
            ExecutionResult(
                input_value=iv,
                return_value=raw.get("return_value"),
                return_type=raw.get("return_type"),
                stdout=raw.get("stdout", ""),
                stdout_truncated=raw.get("stdout_truncated", False),
                error=raw.get("error"),
            )
        )
    return results


def run_test_cases(
    code: str,
    function_name: str,
    test_cases: list[TestCase],
    timeout_seconds: float = 10.0,
) -> tuple[list[TestCaseResult], float]:
    """Run test cases and return (results, pass_rate)."""
    if not test_cases:
        return [], 0.0

    input_values = [tc.input_value for tc in test_cases]
    exec_results = run_function_batch(
        code, function_name, input_values, timeout_seconds
    )

    tc_results: list[TestCaseResult] = []
    for tc, er in zip(test_cases, exec_results):
        passed = er.error is None and _values_equal(er.return_value, tc.expected_output)
        tc_results.append(
            TestCaseResult(
                input_value=tc.input_value,
                expected_output=tc.expected_output,
                actual_output=er.return_value,
                passed=passed,
                error=er.error,
            )
        )

    pass_rate = sum(1 for r in tc_results if r.passed) / len(tc_results)
    return tc_results, pass_rate


def check_compiles(code: str) -> tuple[bool, str | None]:
    """Check if Python code compiles. Uses compile() directly — no subprocess needed."""
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"
    return True, None
