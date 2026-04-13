"""Code execution runner with Docker (default) and local subprocess modes.

Docker mode requires the ``dr-docker`` package (install with
``pip install nl-code[docker]``). Local mode uses a subprocess with the
system Python interpreter.

Error contract:
  - Infrastructure problems RAISE ``CodeExecutionInfrastructureError``.
  - Code-level problems are RETURNED in result objects.
  These never overlap.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, model_validator

from nl_code.code_execution.models import (
    DEFAULT_CODE_EVAL_IMAGE,
    AssertionBatchItem,
    AssertionTestResult,
    CodeExecutionInfrastructureError,
    ExecutionResult,
    FunctionCallBatchItem,
    TestCase,
    TestCaseResult,
    UnittestBatchItem,
    UnittestResult,
    UnittestTestDetail,
)

logger = logging.getLogger(__name__)

# Backward compatibility alias
ExecutionError = CodeExecutionInfrastructureError

EXEC_MODE_DOCKER = "docker_worker"
EXEC_MODE_LOCAL = "local_worker"
_VALID_EXEC_MODES = {EXEC_MODE_DOCKER, EXEC_MODE_LOCAL}

_DEFAULT_STREAM_LIMIT = 1_048_576  # 1 MB


# ---------------------------------------------------------------------------
# Docker runtime configuration
# ---------------------------------------------------------------------------


class _WorkerRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    docker_image: str | None
    tmpfs_size: str
    pids_limit: int
    memory: str
    cpus: float
    tmpfs_exec: bool
    fsize_bytes: int
    nofile: int


_PYTHON_RUNTIME_CONFIG = _WorkerRuntimeConfig(
    docker_image=None,
    tmpfs_size="64m",
    pids_limit=256,
    memory="512m",
    cpus=1.0,
    tmpfs_exec=False,
    fsize_bytes=10_485_760,  # 10 MB
    nofile=1024,
)


# ---------------------------------------------------------------------------
# Worker request model
# ---------------------------------------------------------------------------


class WorkerRequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str = "function_call"
    code: str | None = None
    function_name: str | None = None
    input_value: Any | None = None
    input_values: list[Any] | None = None
    test_code: str | None = None
    test_class_names: list[str] | None = None
    # batch fields
    items: list[dict[str, Any]] | None = None
    timeout_per_item: int | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> WorkerRequestModel:
        if self.mode == "function_call":
            has_value = "input_value" in self.model_fields_set
            has_values = "input_values" in self.model_fields_set
            if has_value and has_values:
                raise ValueError("cannot include both input_value and input_values")
            if not has_value and not has_values:
                raise ValueError("must include input_value or input_values")
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Invalid integer for %s=%r, using default %d", name, raw, default
        )
        return default


def _worker_script_path() -> Path:
    return Path(__file__).resolve().parent / "worker.py"


def _require_worker_script(*, execution_mode: str) -> Path:
    worker = _worker_script_path()
    if not worker.is_file():
        raise CodeExecutionInfrastructureError(
            stage="worker_script_missing",
            execution_mode=execution_mode,
            detail=f"worker script not found at {worker}",
        )
    return worker


def _normalize_execution_mode(execution_mode: str) -> str:
    if execution_mode not in _VALID_EXEC_MODES:
        raise ValueError(
            f"Invalid execution_mode={execution_mode!r}. "
            f"Expected one of: {sorted(_VALID_EXEC_MODES)}"
        )
    return execution_mode


def _values_equal(actual: Any, expected: Any, rel_tol: float = 1e-9) -> bool:
    if isinstance(actual, float) and isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=rel_tol)
    if isinstance(actual, float) and isinstance(expected, int):
        return math.isclose(actual, float(expected), rel_tol=rel_tol)
    if isinstance(actual, int) and isinstance(expected, float):
        return math.isclose(float(actual), expected, rel_tol=rel_tol)
    return actual == expected


# ---------------------------------------------------------------------------
# Docker adapter (lazy import)
# ---------------------------------------------------------------------------


def _import_dr_docker() -> tuple[Any, ...]:
    """Lazy-import dr-docker types. Raises ImportError with install hint."""
    try:
        from dr_docker import (
            DockerMount,
            DockerRuntimeRequest,
            ErrorCode,
            ResourceLimits,
            SubprocessDockerAdapter,
            TmpfsMount,
        )
    except ImportError as exc:
        raise ImportError(
            "dr-docker is required for Docker execution mode. "
            "Install with: pip install nl-code[docker]"
        ) from exc
    return (
        SubprocessDockerAdapter,
        DockerRuntimeRequest,
        DockerMount,
        TmpfsMount,
        ResourceLimits,
        ErrorCode,
    )


def _docker_unavailable_stage(error_message: str) -> str:
    normalized = error_message.lower()
    if "not found on path" in normalized:
        return "docker_cli_missing"
    return "docker_unavailable"


def _make_adapter(*, stream_limit: int = _DEFAULT_STREAM_LIMIT) -> Any:
    SubprocessDockerAdapter = _import_dr_docker()[0]
    stdout_limit = _parse_int_env("NL_CODE_EVAL_WORKER_MAX_STDOUT_BYTES", stream_limit)
    stderr_limit = _parse_int_env("NL_CODE_EVAL_WORKER_MAX_STDERR_BYTES", stream_limit)
    return SubprocessDockerAdapter(
        max_stdout_bytes=stdout_limit,
        max_stderr_bytes=stderr_limit,
    )


def _build_docker_runtime_request(
    *,
    stdin_payload: bytes,
    timeout_seconds: float,
    runtime: _WorkerRuntimeConfig,
) -> Any:
    (
        _Adapter,
        DockerRuntimeRequest,
        DockerMount,
        TmpfsMount,
        ResourceLimits,
        _ErrorCode,
    ) = _import_dr_docker()

    worker = _require_worker_script(execution_mode=EXEC_MODE_DOCKER)
    worker_dir = worker.parent.resolve()
    if not worker_dir.is_dir():
        raise CodeExecutionInfrastructureError(
            stage="worker_bind_source_missing",
            execution_mode=EXEC_MODE_DOCKER,
            detail=f"worker bind directory does not exist: {worker_dir}",
        )

    image = runtime.docker_image or DEFAULT_CODE_EVAL_IMAGE
    return DockerRuntimeRequest(
        image=image,
        command=["-I", "-S", "/sandbox/code_eval_worker.py"],
        entrypoint="python3",
        timeout_seconds=max(1, math.ceil(timeout_seconds)),
        stdin_payload=stdin_payload,
        mounts=[
            DockerMount(
                source=str(worker_dir),
                target="/sandbox",
                read_only=True,
            ),
        ],
        tmpfs=[
            TmpfsMount(
                target="/tmp",
                size=runtime.tmpfs_size,
                exec_=runtime.tmpfs_exec,
            ),
        ],
        resources=ResourceLimits(
            memory=runtime.memory,
            cpus=runtime.cpus,
            pids_limit=runtime.pids_limit,
            fsize_bytes=runtime.fsize_bytes,
            nofile=runtime.nofile,
            nproc=runtime.pids_limit,
        ),
    )


# ---------------------------------------------------------------------------
# Worker invocation (Docker and local)
# ---------------------------------------------------------------------------


def _serialize_request(req: dict[str, Any], execution_mode: str) -> bytes:
    try:
        return json.dumps(req).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CodeExecutionInfrastructureError(
            stage="worker_request_serialization",
            execution_mode=execution_mode,
            detail=f"Failed to serialize worker request: {exc}",
        ) from exc


def _run_docker_worker(
    req_bytes: bytes,
    timeout_seconds: float,
    runtime: _WorkerRuntimeConfig,
    *,
    stream_limit: int = _DEFAULT_STREAM_LIMIT,
) -> subprocess.CompletedProcess[str]:
    _ErrorCode = _import_dr_docker()[5]

    docker_request = _build_docker_runtime_request(
        stdin_payload=req_bytes,
        timeout_seconds=timeout_seconds,
        runtime=runtime,
    )
    adapter = _make_adapter(stream_limit=stream_limit)
    try:
        result = adapter.execute_in_runtime(docker_request)
    except Exception as exc:  # noqa: BLE001
        raise CodeExecutionInfrastructureError(
            stage="worker_subprocess_error",
            execution_mode=EXEC_MODE_DOCKER,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    if result.error and result.error.code == _ErrorCode.UNAVAILABLE:
        raise CodeExecutionInfrastructureError(
            stage=_docker_unavailable_stage(result.error.message),
            execution_mode=EXEC_MODE_DOCKER,
            detail=result.error.message,
        )
    if result.error and result.error.code == _ErrorCode.TIMEOUT:
        raise CodeExecutionInfrastructureError(
            stage="docker_timeout",
            execution_mode=EXEC_MODE_DOCKER,
            detail=f"container timed out after {timeout_seconds}s",
        )
    if result.error and result.exit_code is None:
        raise CodeExecutionInfrastructureError(
            stage="docker_runtime_error",
            execution_mode=EXEC_MODE_DOCKER,
            detail=f"{result.error.code.value}: {result.error.message}",
        )

    return subprocess.CompletedProcess(
        args=["docker", "run"],
        returncode=result.exit_code or 0,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _run_local_worker(
    req_bytes: bytes,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    worker = _require_worker_script(execution_mode=EXEC_MODE_LOCAL)
    cmd = [sys.executable, "-I", "-S", str(worker)]
    try:
        result = subprocess.run(
            cmd,
            input=req_bytes,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise CodeExecutionInfrastructureError(
            stage="worker_timeout",
            execution_mode=EXEC_MODE_LOCAL,
            detail=f"worker timed out after {timeout_seconds}s",
        ) from exc
    return subprocess.CompletedProcess(
        args=cmd,
        returncode=result.returncode,
        stdout=result.stdout.decode("utf-8", errors="replace"),
        stderr=result.stderr.decode("utf-8", errors="replace"),
    )


def _run_worker(
    req: dict[str, Any],
    timeout_seconds: float,
    execution_mode: str,
    runtime: _WorkerRuntimeConfig,
    *,
    stream_limit: int = _DEFAULT_STREAM_LIMIT,
) -> subprocess.CompletedProcess[str]:
    req_bytes = _serialize_request(req, execution_mode)
    if execution_mode == EXEC_MODE_DOCKER:
        return _run_docker_worker(
            req_bytes, timeout_seconds, runtime, stream_limit=stream_limit
        )
    return _run_local_worker(req_bytes, timeout_seconds)


def _parse_worker_json(
    proc: subprocess.CompletedProcess[str],
    execution_mode: str,
) -> dict[str, Any]:
    if not proc.stdout.strip():
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_parse",
            execution_mode=execution_mode,
            detail=(
                f"worker returned no output "
                f"(rc={proc.returncode}, stderr={proc.stderr[:200]})"
            ),
        )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_parse",
            execution_mode=execution_mode,
            detail=f"invalid JSON from worker: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_parse",
            execution_mode=execution_mode,
            detail="worker returned non-object JSON",
        )
    if proc.returncode != 0:
        raise CodeExecutionInfrastructureError(
            stage="worker_nonzero_exit",
            execution_mode=execution_mode,
            detail=(
                f"worker exited with rc={proc.returncode}: {proc.stderr.strip()[:200]}"
            ),
        )
    return payload


# ---------------------------------------------------------------------------
# Result builders
# ---------------------------------------------------------------------------


def _execution_result_from_payload(
    input_value: Any,
    raw: dict[str, Any],
) -> ExecutionResult:
    compile_success = raw.get("compile_success")
    if not isinstance(compile_success, bool):
        compile_success = None
    compile_error = (
        str(raw["compile_error"]) if raw.get("compile_error") is not None else None
    )
    return ExecutionResult(
        input_value=input_value,
        return_value=raw.get("return_value"),
        return_type=raw.get("return_type"),
        stdout=raw.get("stdout", ""),
        stdout_truncated=raw.get("stdout_truncated", False),
        error=str(raw["error"]) if raw.get("error") else None,
        compile_success=compile_success,
        compile_error=compile_error,
    )


def _parse_function_call_results(
    payload: dict[str, Any],
    input_values: list[Any],
    execution_mode: str,
) -> list[ExecutionResult]:
    top_error = payload.get("error")
    if top_error:
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_error",
            execution_mode=execution_mode,
            detail=str(top_error),
        )
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_parse",
            execution_mode=execution_mode,
            detail="missing results list in worker response",
        )
    if len(raw_results) != len(input_values):
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_mismatched_batch_count",
            execution_mode=execution_mode,
            detail=(f"expected {len(input_values)} results, got {len(raw_results)}"),
        )
    results: list[ExecutionResult] = []
    for i, (iv, raw) in enumerate(zip(input_values, raw_results, strict=True)):
        if not isinstance(raw, dict):
            raise CodeExecutionInfrastructureError(
                stage="worker_payload_invalid_per_input",
                execution_mode=execution_mode,
                detail=f"result at index {i} is {type(raw).__name__}, expected dict",
            )
        results.append(
            _execution_result_from_payload(iv, cast(dict[str, Any], raw))
        )
    return results


# ---------------------------------------------------------------------------
# Public API — single-item functions
# ---------------------------------------------------------------------------


def check_compiles(code: str) -> tuple[bool, str | None]:
    """Check if Python code compiles. Uses compile() directly."""
    try:
        compile(code, "<generated>", "exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"
    return True, None


def run_function_batch(
    code: str,
    function_name: str,
    input_values: list[Any],
    timeout_seconds: float = 60.0,
    *,
    execution_mode: str = EXEC_MODE_DOCKER,
    docker_image: str | None = None,
) -> list[ExecutionResult]:
    """Execute a function with multiple inputs in an isolated environment."""
    execution_mode = _normalize_execution_mode(execution_mode)
    if not input_values:
        return []

    req: dict[str, Any] = {
        "mode": "function_call",
        "code": code,
        "function_name": function_name,
        "input_values": input_values,
    }
    runtime = _PYTHON_RUNTIME_CONFIG.model_copy(update={"docker_image": docker_image})
    proc = _run_worker(req, timeout_seconds, execution_mode, runtime)
    payload = _parse_worker_json(proc, execution_mode)
    return _parse_function_call_results(payload, input_values, execution_mode)


def run_test_cases(
    code: str,
    function_name: str,
    test_cases: list[TestCase],
    timeout_seconds: float = 30.0,
    *,
    execution_mode: str = EXEC_MODE_DOCKER,
    docker_image: str | None = None,
) -> tuple[list[TestCaseResult], float]:
    """Run test cases and return (results, pass_rate)."""
    if not test_cases:
        return [], 0.0

    input_values = [tc.input_value for tc in test_cases]
    exec_results = run_function_batch(
        code,
        function_name,
        input_values,
        timeout_seconds,
        execution_mode=execution_mode,
        docker_image=docker_image,
    )

    tc_results: list[TestCaseResult] = []
    for tc, er in zip(test_cases, exec_results, strict=True):
        if er.error:
            tc_results.append(
                TestCaseResult(
                    input_value=tc.input_value,
                    expected_output=tc.expected_output,
                    actual_output=None,
                    passed=False,
                    error=er.error,
                    compile_success=er.compile_success,
                    compile_error=er.compile_error or er.error,
                )
            )
        else:
            passed = _values_equal(er.return_value, tc.expected_output)
            tc_results.append(
                TestCaseResult(
                    input_value=tc.input_value,
                    expected_output=tc.expected_output,
                    actual_output=er.return_value,
                    passed=passed,
                    compile_success=er.compile_success,
                    compile_error=er.compile_error,
                )
            )

    pass_rate = sum(1 for r in tc_results if r.passed) / len(tc_results)
    return tc_results, pass_rate


def run_assertion_test(
    code: str,
    test_code: str,
    timeout_seconds: float = 30.0,
    *,
    execution_mode: str = EXEC_MODE_DOCKER,
    docker_image: str | None = None,
) -> AssertionTestResult:
    """Run code with assertion-based test code (Pro datasets)."""
    execution_mode = _normalize_execution_mode(execution_mode)
    req: dict[str, Any] = {
        "mode": "assertion",
        "code": code,
        "test_code": test_code,
    }
    runtime = _PYTHON_RUNTIME_CONFIG.model_copy(update={"docker_image": docker_image})
    proc = _run_worker(req, timeout_seconds, execution_mode, runtime)
    payload = _parse_worker_json(proc, execution_mode)
    return AssertionTestResult(
        passed=payload.get("passed", False),
        error=payload.get("error"),
        stdout=payload.get("stdout", ""),
        compile_success=payload.get("compile_success"),
        compile_error=payload.get("compile_error"),
    )


def run_unittest_test(
    code: str,
    test_code: str,
    test_class_names: list[str],
    timeout_seconds: float = 30.0,
    *,
    execution_mode: str = EXEC_MODE_DOCKER,
    docker_image: str | None = None,
) -> UnittestResult:
    """Run code with unittest test classes (ClassEval)."""
    execution_mode = _normalize_execution_mode(execution_mode)
    req: dict[str, Any] = {
        "mode": "unittest",
        "code": code,
        "test_code": test_code,
        "test_class_names": test_class_names,
    }
    runtime = _PYTHON_RUNTIME_CONFIG.model_copy(update={"docker_image": docker_image})
    proc = _run_worker(req, timeout_seconds, execution_mode, runtime)
    payload = _parse_worker_json(proc, execution_mode)

    if payload.get("error"):
        return UnittestResult(
            all_passed=False,
            total_tests_run=0,
            total_tests_passed=0,
            total_tests_failed=0,
            total_tests_errored=0,
            per_test_class=[],
            error=payload["error"],
        )

    per_test_class = [
        UnittestTestDetail(**tc) for tc in payload.get("per_test_class", [])
    ]
    return UnittestResult(
        all_passed=payload.get("all_passed", False),
        total_tests_run=payload.get("total_tests_run", 0),
        total_tests_passed=payload.get("total_tests_passed", 0),
        total_tests_failed=payload.get("total_tests_failed", 0),
        total_tests_errored=payload.get("total_tests_errored", 0),
        per_test_class=per_test_class,
        error=payload.get("error"),
    )


# ---------------------------------------------------------------------------
# Public API — batch functions (amortize Docker startup)
# ---------------------------------------------------------------------------


def _run_batch_chunk(
    items: list[dict[str, Any]],
    timeout_per_item: float,
    execution_mode: str,
    docker_image: str | None,
    chunk_mode: str,
) -> list[dict[str, Any]]:
    """Send a chunk of items to a single worker and return raw results."""
    execution_mode = _normalize_execution_mode(execution_mode)
    req: dict[str, Any] = {
        "mode": "batch",
        "timeout_per_item": int(timeout_per_item),
        "items": items,
    }
    runtime = _PYTHON_RUNTIME_CONFIG.model_copy(update={"docker_image": docker_image})
    # Docker timeout: generous budget for the full chunk
    docker_timeout = timeout_per_item * len(items) * 1.5 + 10
    # Scale stream limits for batch
    stream_limit = max(10_485_760, len(items) * 51_200)

    proc = _run_worker(
        req,
        docker_timeout,
        execution_mode,
        runtime,
        stream_limit=stream_limit,
    )
    payload = _parse_worker_json(proc, execution_mode)

    if payload.get("error"):
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_error",
            execution_mode=execution_mode,
            detail=str(payload["error"]),
        )

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_parse",
            execution_mode=execution_mode,
            detail="missing results list in batch response",
        )
    if len(raw_results) != len(items):
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_mismatched_batch_count",
            execution_mode=execution_mode,
            detail=(f"expected {len(items)} batch results, got {len(raw_results)}"),
        )
    for i, entry in enumerate(raw_results):
        if not isinstance(entry, dict):
            raise CodeExecutionInfrastructureError(
                stage="worker_payload_invalid_per_input",
                execution_mode=execution_mode,
                detail=f"batch result at index {i} is {type(entry).__name__}, expected dict",
            )
    return raw_results


def _chunk_list(lst: list[Any], size: int) -> list[list[Any]]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def batch_run_test_cases(
    items: list[FunctionCallBatchItem],
    timeout_per_item: float = 30.0,
    *,
    chunk_size: int = 200,
    execution_mode: str = EXEC_MODE_DOCKER,
    docker_image: str | None = None,
) -> list[tuple[list[TestCaseResult], float]]:
    """Run test cases for many code samples, batched into containers."""
    if not items:
        return []

    # Build worker items
    worker_items = []
    for item in items:
        worker_items.append(
            {
                "mode": "function_call",
                "code": item.code,
                "function_name": item.function_name,
                "input_values": [tc.input_value for tc in item.test_cases],
            }
        )

    all_results: list[tuple[list[TestCaseResult], float]] = []
    for chunk_idx, chunk in enumerate(_chunk_list(worker_items, chunk_size)):
        raw_results = _run_batch_chunk(
            chunk,
            timeout_per_item,
            execution_mode,
            docker_image,
            "function_call",
        )
        # Map chunk offset to original items for test case matching
        offset = chunk_idx * chunk_size
        for i, raw in enumerate(raw_results):
            item = items[offset + i]
            tc_list = item.test_cases

            # Handle item-level errors (timeout, crash)
            if raw.get("error") and not raw.get("results"):
                error_msg = str(raw["error"])
                tc_results = [
                    TestCaseResult(
                        input_value=tc.input_value,
                        expected_output=tc.expected_output,
                        passed=False,
                        error=error_msg,
                    )
                    for tc in tc_list
                ]
                all_results.append((tc_results, 0.0))
                continue

            exec_results_raw = raw.get("results", [])
            # Pad or truncate to match test case count
            tc_results: list[TestCaseResult] = []
            for j, tc in enumerate(tc_list):
                if j < len(exec_results_raw):
                    er = exec_results_raw[j]
                    if er.get("error"):
                        tc_results.append(
                            TestCaseResult(
                                input_value=tc.input_value,
                                expected_output=tc.expected_output,
                                passed=False,
                                error=str(er["error"]),
                                compile_success=er.get("compile_success"),
                                compile_error=(
                                    str(er["compile_error"])
                                    if er.get("compile_error")
                                    else None
                                ),
                            )
                        )
                    else:
                        passed = _values_equal(
                            er.get("return_value"), tc.expected_output
                        )
                        tc_results.append(
                            TestCaseResult(
                                input_value=tc.input_value,
                                expected_output=tc.expected_output,
                                actual_output=er.get("return_value"),
                                passed=passed,
                                compile_success=er.get("compile_success"),
                                compile_error=(
                                    str(er["compile_error"])
                                    if er.get("compile_error")
                                    else None
                                ),
                            )
                        )
                else:
                    tc_results.append(
                        TestCaseResult(
                            input_value=tc.input_value,
                            expected_output=tc.expected_output,
                            passed=False,
                            error="missing result from worker",
                        )
                    )
            pass_rate = (
                sum(1 for r in tc_results if r.passed) / len(tc_results)
                if tc_results
                else 0.0
            )
            all_results.append((tc_results, pass_rate))

    return all_results


def batch_run_assertion_tests(
    items: list[AssertionBatchItem],
    timeout_per_item: float = 30.0,
    *,
    chunk_size: int = 200,
    execution_mode: str = EXEC_MODE_DOCKER,
    docker_image: str | None = None,
) -> list[AssertionTestResult]:
    """Run assertion tests for many code samples, batched into containers."""
    if not items:
        return []

    worker_items = [
        {"mode": "assertion", "code": item.code, "test_code": item.test_code}
        for item in items
    ]

    all_results: list[AssertionTestResult] = []
    for chunk in _chunk_list(worker_items, chunk_size):
        raw_results = _run_batch_chunk(
            chunk, timeout_per_item, execution_mode, docker_image, "assertion"
        )
        for raw in raw_results:
            all_results.append(
                AssertionTestResult(
                    passed=raw.get("passed", False),
                    error=raw.get("error"),
                    stdout=raw.get("stdout", ""),
                    compile_success=raw.get("compile_success"),
                    compile_error=raw.get("compile_error"),
                )
            )

    return all_results


def batch_run_unittest_tests(
    items: list[UnittestBatchItem],
    timeout_per_item: float = 30.0,
    *,
    chunk_size: int = 200,
    execution_mode: str = EXEC_MODE_DOCKER,
    docker_image: str | None = None,
) -> list[UnittestResult]:
    """Run unittest tests for many code samples, batched into containers."""
    if not items:
        return []

    worker_items = [
        {
            "mode": "unittest",
            "code": item.code,
            "test_code": item.test_code,
            "test_class_names": item.test_class_names,
        }
        for item in items
    ]

    all_results: list[UnittestResult] = []
    for chunk in _chunk_list(worker_items, chunk_size):
        raw_results = _run_batch_chunk(
            chunk, timeout_per_item, execution_mode, docker_image, "unittest"
        )
        for raw in raw_results:
            if raw.get("error") and not raw.get("per_test_class"):
                all_results.append(
                    UnittestResult(
                        all_passed=False,
                        total_tests_run=0,
                        total_tests_passed=0,
                        total_tests_failed=0,
                        total_tests_errored=0,
                        per_test_class=[],
                        error=str(raw["error"]),
                    )
                )
            else:
                per_test_class = [
                    UnittestTestDetail(**tc) for tc in raw.get("per_test_class", [])
                ]
                all_results.append(
                    UnittestResult(
                        all_passed=raw.get("all_passed", False),
                        total_tests_run=raw.get("total_tests_run", 0),
                        total_tests_passed=raw.get("total_tests_passed", 0),
                        total_tests_failed=raw.get("total_tests_failed", 0),
                        total_tests_errored=raw.get("total_tests_errored", 0),
                        per_test_class=per_test_class,
                        error=raw.get("error"),
                    )
                )

    return all_results
