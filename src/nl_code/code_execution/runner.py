"""Code execution runner with Docker worker isolation.

Docker execution requires the ``dr-docker`` package (install with
``pip install nl-code[docker]``).

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
from pathlib import Path
from typing import Any, Never, cast

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
_DEFAULT_STREAM_LIMIT = 1_048_576  # 1 MB
_WORKER_MOUNT_DIR = "/sandbox"
_WORKER_WORKING_DIR = "/tmp"
_DEFAULT_DOCKER_MEMORY = "4g"
_DEFAULT_DOCKER_STDIN_LIMIT_BYTES = 52_428_800


# ---------------------------------------------------------------------------
# Docker runtime configuration
# ---------------------------------------------------------------------------


class _WorkerRuntimeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    docker_image: str | None
    worker_policy: Any


def _default_worker_policy() -> Any:
    WorkerRuntimePolicy = _import_dr_docker()[3]
    return WorkerRuntimePolicy.small_isolated().model_copy(
        update={
            "memory": _DEFAULT_DOCKER_MEMORY,
        }
    )


def _runtime_config(*, docker_image: str | None) -> _WorkerRuntimeConfig:
    try:
        worker_policy = _default_worker_policy().with_env_overrides()
    except ValueError as exc:
        raise CodeExecutionInfrastructureError(
            stage="worker_runtime_config",
            execution_mode=EXEC_MODE_DOCKER,
            detail=str(exc),
        ) from exc
    return _WorkerRuntimeConfig(
        docker_image=docker_image,
        worker_policy=worker_policy,
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
    timeout_per_item: float | None = None

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


def _require_worker_script() -> Path:
    worker = _worker_script_path()
    if not worker.is_file():
        raise CodeExecutionInfrastructureError(
            stage="worker_script_missing",
            execution_mode=EXEC_MODE_DOCKER,
            detail=f"worker script not found at {worker}",
        )
    return worker


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
            ErrorCode,
            JsonWorkerExecutionConfig,
            RuntimePrimitiveError,
            SubprocessDockerAdapter,
            WorkerRuntimePolicy,
            build_mounted_worker_request,
            execute_batch_in_container,
            execute_in_runtime_or_raise,
            mount_worker_file,
        )
    except ImportError as exc:
        raise ImportError(
            "dr-docker is required for Docker execution mode. "
            "Install with: pip install nl-code[docker]"
        ) from exc
    return (
        SubprocessDockerAdapter,
        ErrorCode,
        RuntimePrimitiveError,
        WorkerRuntimePolicy,
        build_mounted_worker_request,
        mount_worker_file,
        execute_batch_in_container,
        execute_in_runtime_or_raise,
        JsonWorkerExecutionConfig,
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


def _default_worker_execution_config(
    *, runtime_policy: Any, timeout_seconds: float
) -> Any:
    JsonWorkerExecutionConfig = _import_dr_docker()[8]
    return JsonWorkerExecutionConfig.from_runtime_policy(
        runtime_policy,
        timeout_seconds=timeout_seconds,
    ).model_copy(
        update={
            "stdin_limit_bytes": _DEFAULT_DOCKER_STDIN_LIMIT_BYTES,
        }
    )


def _build_docker_runtime_request(
    *,
    stdin_payload: bytes,
    timeout_seconds: float,
    runtime: _WorkerRuntimeConfig,
    docker_env: dict[str, str] | None = None,
) -> Any:
    (
        _Adapter,
        _ErrorCode,
        _RuntimePrimitiveError,
        _WorkerRuntimePolicy,
        build_mounted_worker_request,
        mount_worker_file,
        _execute_batch_in_container,
        _execute_in_runtime_or_raise,
        JsonWorkerExecutionConfig,
    ) = _import_dr_docker()

    worker = _require_worker_script()
    mounted_worker = mount_worker_file(
        worker,
        mount_target=_WORKER_MOUNT_DIR,
    ).with_path_command(
        entrypoint="python3",
        args_before_path=["-I"],
        working_dir=_WORKER_WORKING_DIR,
    )
    image = runtime.docker_image or DEFAULT_CODE_EVAL_IMAGE
    try:
        worker_execution_config = _default_worker_execution_config(
            runtime_policy=runtime.worker_policy,
            timeout_seconds=timeout_seconds,
        ).with_env_overrides()
    except ValueError as exc:
        raise CodeExecutionInfrastructureError(
            stage="worker_runtime_config",
            execution_mode=EXEC_MODE_DOCKER,
            detail=str(exc),
        ) from exc

    env = worker_execution_config.to_env()
    if docker_env:
        env.update(docker_env)

    return build_mounted_worker_request(
        image=image,
        worker=mounted_worker,
        timeout_seconds=max(1, math.ceil(timeout_seconds)),
        policy=runtime.worker_policy,
        stdin_payload=stdin_payload,
        env=env,
    )


# ---------------------------------------------------------------------------
# Worker invocation
# ---------------------------------------------------------------------------


def _serialize_request(req: dict[str, Any]) -> bytes:
    try:
        return json.dumps(req).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CodeExecutionInfrastructureError(
            stage="worker_request_serialization",
            execution_mode=EXEC_MODE_DOCKER,
            detail=f"Failed to serialize worker request: {exc}",
        ) from exc


def _run_docker_worker(
    req_bytes: bytes,
    timeout_seconds: float,
    runtime: _WorkerRuntimeConfig,
    *,
    stream_limit: int = _DEFAULT_STREAM_LIMIT,
    docker_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    _ErrorCode = _import_dr_docker()[1]
    RuntimePrimitiveError = _import_dr_docker()[2]
    execute_in_runtime_or_raise = _import_dr_docker()[7]

    docker_request = _build_docker_runtime_request(
        stdin_payload=req_bytes,
        timeout_seconds=timeout_seconds,
        runtime=runtime,
        docker_env=docker_env,
    )
    adapter = _make_adapter(stream_limit=stream_limit)
    try:
        result = execute_in_runtime_or_raise(adapter, docker_request)
    except RuntimePrimitiveError as exc:
        _raise_code_execution_infra_from_runtime_error(exc, error_code_type=_ErrorCode)
    except Exception as exc:  # noqa: BLE001
        raise CodeExecutionInfrastructureError(
            stage="worker_subprocess_error",
            execution_mode=EXEC_MODE_DOCKER,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    return subprocess.CompletedProcess(
        args=["docker", "run"],
        returncode=result.exit_code or 0,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def _run_worker(
    req: dict[str, Any],
    timeout_seconds: float,
    runtime: _WorkerRuntimeConfig,
    *,
    stream_limit: int = _DEFAULT_STREAM_LIMIT,
    docker_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    req_bytes = _serialize_request(req)
    return _run_docker_worker(
        req_bytes,
        timeout_seconds,
        runtime,
        stream_limit=stream_limit,
        docker_env=docker_env,
    )


def _parse_worker_json(
    proc: subprocess.CompletedProcess[str],
) -> dict[str, Any]:
    if not proc.stdout.strip():
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_parse",
            execution_mode=EXEC_MODE_DOCKER,
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
            execution_mode=EXEC_MODE_DOCKER,
            detail=f"invalid JSON from worker: {exc}",
        ) from exc
    if not isinstance(payload, dict):
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_parse",
            execution_mode=EXEC_MODE_DOCKER,
            detail="worker returned non-object JSON",
        )
    if proc.returncode != 0:
        raise CodeExecutionInfrastructureError(
            stage="worker_nonzero_exit",
            execution_mode=EXEC_MODE_DOCKER,
            detail=(
                f"worker exited with rc={proc.returncode}: {proc.stderr.strip()[:200]}"
            ),
        )
    return payload


def _completed_process_from_runtime_result(
    runtime_result: Any,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["docker", "run"],
        returncode=runtime_result.exit_code or 0,
        stdout=runtime_result.stdout,
        stderr=runtime_result.stderr,
    )


def _raise_code_execution_infra_from_runtime_error(
    exc: Any, *, error_code_type: Any
) -> Never:
    if exc.error.code == error_code_type.UNAVAILABLE:
        raise CodeExecutionInfrastructureError(
            stage=_docker_unavailable_stage(exc.error.message),
            execution_mode=EXEC_MODE_DOCKER,
            detail=exc.error.message,
        ) from exc
    if exc.error.code == error_code_type.TIMEOUT:
        raise CodeExecutionInfrastructureError(
            stage="docker_timeout",
            execution_mode=EXEC_MODE_DOCKER,
            detail=exc.error.message,
        ) from exc
    raise CodeExecutionInfrastructureError(
        stage="docker_runtime_error",
        execution_mode=EXEC_MODE_DOCKER,
        detail=f"{exc.error.code.value}: {exc.error.message}",
    ) from exc


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
) -> list[ExecutionResult]:
    top_error = payload.get("error")
    if top_error:
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_error",
            execution_mode=EXEC_MODE_DOCKER,
            detail=str(top_error),
        )
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_parse",
            execution_mode=EXEC_MODE_DOCKER,
            detail="missing results list in worker response",
        )
    if len(raw_results) != len(input_values):
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_mismatched_batch_count",
            execution_mode=EXEC_MODE_DOCKER,
            detail=(f"expected {len(input_values)} results, got {len(raw_results)}"),
        )
    results: list[ExecutionResult] = []
    for i, (iv, raw) in enumerate(zip(input_values, raw_results, strict=True)):
        if not isinstance(raw, dict):
            raise CodeExecutionInfrastructureError(
                stage="worker_payload_invalid_per_input",
                execution_mode=EXEC_MODE_DOCKER,
                detail=f"result at index {i} is {type(raw).__name__}, expected dict",
            )
        results.append(_execution_result_from_payload(iv, cast(dict[str, Any], raw)))
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
    docker_image: str | None = None,
    docker_env: dict[str, str] | None = None,
) -> list[ExecutionResult]:
    """Execute a function with multiple inputs in an isolated environment."""
    if not input_values:
        return []

    req: dict[str, Any] = {
        "mode": "function_call",
        "code": code,
        "function_name": function_name,
        "input_values": input_values,
    }
    runtime = _runtime_config(docker_image=docker_image)
    proc = _run_worker(req, timeout_seconds, runtime, docker_env=docker_env)
    payload = _parse_worker_json(proc)
    return _parse_function_call_results(payload, input_values)


def run_test_cases(
    code: str,
    function_name: str,
    test_cases: list[TestCase],
    timeout_seconds: float = 30.0,
    *,
    docker_image: str | None = None,
    docker_env: dict[str, str] | None = None,
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
        docker_image=docker_image,
        docker_env=docker_env,
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
                    compile_error=er.compile_error,
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
    docker_image: str | None = None,
    docker_env: dict[str, str] | None = None,
) -> AssertionTestResult:
    """Run code with assertion-based test code (Pro datasets)."""
    req: dict[str, Any] = {
        "mode": "assertion",
        "code": code,
        "test_code": test_code,
    }
    runtime = _runtime_config(docker_image=docker_image)
    proc = _run_worker(req, timeout_seconds, runtime, docker_env=docker_env)
    payload = _parse_worker_json(proc)
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
    docker_image: str | None = None,
    docker_env: dict[str, str] | None = None,
) -> UnittestResult:
    """Run code with unittest test classes (ClassEval)."""
    req: dict[str, Any] = {
        "mode": "unittest",
        "code": code,
        "test_code": test_code,
        "test_class_names": test_class_names,
    }
    runtime = _runtime_config(docker_image=docker_image)
    proc = _run_worker(req, timeout_seconds, runtime, docker_env=docker_env)
    payload = _parse_worker_json(proc)

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
    docker_image: str | None,
    docker_env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Send a chunk of items to a single worker and return raw results."""
    runtime = _runtime_config(docker_image=docker_image)
    ErrorCode = _import_dr_docker()[1]
    RuntimePrimitiveError = _import_dr_docker()[2]
    execute_batch_in_container = _import_dr_docker()[6]

    def build_request(batch_items: list[dict[str, Any]]) -> Any:
        req: dict[str, Any] = {
            "mode": "batch",
            "timeout_per_item": timeout_per_item,
            "items": batch_items,
        }
        docker_timeout = timeout_per_item * len(batch_items) * 1.5 + 10
        return _build_docker_runtime_request(
            stdin_payload=_serialize_request(req),
            timeout_seconds=docker_timeout,
            runtime=runtime,
            docker_env=docker_env,
        )

    def parse_results(runtime_result: Any) -> list[dict[str, Any]]:
        payload = _parse_worker_json(
            _completed_process_from_runtime_result(runtime_result)
        )

        if payload.get("error"):
            raise CodeExecutionInfrastructureError(
                stage="worker_payload_error",
                execution_mode=EXEC_MODE_DOCKER,
                detail=str(payload["error"]),
            )

        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise CodeExecutionInfrastructureError(
                stage="worker_payload_parse",
                execution_mode=EXEC_MODE_DOCKER,
                detail="missing results list in batch response",
            )
        for i, entry in enumerate(raw_results):
            if not isinstance(entry, dict):
                raise CodeExecutionInfrastructureError(
                    stage="worker_payload_invalid_per_input",
                    execution_mode=EXEC_MODE_DOCKER,
                    detail=f"batch result at index {i} is {type(entry).__name__}, expected dict",
                )
        return [cast(dict[str, Any], entry) for entry in raw_results]

    try:
        return execute_batch_in_container(
            items,
            adapter=_make_adapter(stream_limit=max(10_485_760, len(items) * 51_200)),
            build_request=build_request,
            parse_results=parse_results,
        )
    except RuntimePrimitiveError as exc:
        _raise_code_execution_infra_from_runtime_error(exc, error_code_type=ErrorCode)
    except ValueError as exc:
        raise CodeExecutionInfrastructureError(
            stage="worker_payload_mismatched_batch_count",
            execution_mode=EXEC_MODE_DOCKER,
            detail=str(exc),
        ) from exc


def _chunk_list(lst: list[Any], size: int) -> list[list[Any]]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def batch_run_test_cases(
    items: list[FunctionCallBatchItem],
    timeout_per_item: float = 30.0,
    *,
    chunk_size: int = 200,
    docker_image: str | None = None,
    docker_env: dict[str, str] | None = None,
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
            docker_image,
            docker_env,
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
    docker_image: str | None = None,
    docker_env: dict[str, str] | None = None,
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
            chunk,
            timeout_per_item,
            docker_image,
            docker_env,
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
    docker_image: str | None = None,
    docker_env: dict[str, str] | None = None,
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
            chunk,
            timeout_per_item,
            docker_image,
            docker_env,
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
