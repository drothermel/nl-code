"""Sandbox worker for isolated Python code execution.

Reads a JSON request from stdin, executes Python code in a restricted
environment, writes a JSON result to stdout. Designed to run inside a Docker
container with resource limits.

Supported modes:
  function_call — call a named function with inputs, return values
  assertion     — exec code + test_code, check for AssertionError
  unittest      — exec code + test_code, run unittest.TestCase classes
  batch         — process a list of independent items in sequence

Protocol:
  stdin:  JSON request (see mode-specific formats below)
  stdout: JSON result

This module has ZERO imports from the nl_code package — it must remain a
standalone script that runs in an isolated subprocess.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import logging
import os
import resource
import shutil
import signal
import tempfile
import unittest
from typing import Any, Callable, cast

from dr_docker.workers.json_stdio import (
    DEFAULT_WORKER_IN_CONTAINER_ENV_VAR,
    BoundedTextCapture as _BoundedStdoutCapture,
    DockerOnlyExecutionError,
    JsonWorkerExecutionConfig,
    OversizedPayloadError,
    is_running_in_container as _upstream_is_running_in_container,
    read_stdin_bounded as _read_stdin_bounded,
)

logger = logging.getLogger(__name__)

# Shallow copy for namespace isolation: exec()'d code gets its own builtins
# dict rather than a mutable reference to the live module dict.
_EXEC_BUILTINS: dict[str, Any] = dict(builtins.__dict__)

_DISALLOWED_NODES = (
    ast.AsyncWith,
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.Await,
)


class _AstValidator(ast.NodeVisitor):
    def visit(self, node: ast.AST) -> Any:
        if isinstance(node, _DISALLOWED_NODES):
            raise ValueError(f"ast node '{type(node).__name__}' is not allowed")
        return super().visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr.startswith("__") and node.attr.endswith("__"):
            raise ValueError(f"dunder attribute access '{node.attr}' is not allowed")
        return self.generic_visit(node)


def _validate_code_ast(code: str) -> None:
    tree = ast.parse(code, mode="exec")
    _AstValidator().visit(tree)


# ---------------------------------------------------------------------------
# Environment-variable configuration
# ---------------------------------------------------------------------------


def _stdin_limit_bytes() -> int:
    return _worker_execution_config().stdin_limit_bytes


def _stdout_limit_bytes() -> int:
    return _worker_execution_config().stdout_limit_bytes


def _worker_execution_config() -> JsonWorkerExecutionConfig:
    return JsonWorkerExecutionConfig.from_env()


def _is_running_in_container() -> bool:
    return _upstream_is_running_in_container()


def _require_docker_execution() -> None:
    if os.getenv(DEFAULT_WORKER_IN_CONTAINER_ENV_VAR) != "1":
        raise DockerOnlyExecutionError(
            "worker requires "
            f"{DEFAULT_WORKER_IN_CONTAINER_ENV_VAR}=1 from the Docker runner"
        )
    if not _is_running_in_container():
        raise DockerOnlyExecutionError("worker must run inside a Docker container")


def _set_resource_limits(*, skip_cpu: bool = False) -> None:
    try:
        _worker_execution_config().apply_resource_limits(skip_cpu=skip_cpu)
    except RuntimeError as exc:
        logger.debug("Unable to apply resource limits: %s", exc)


def _set_batch_cpu_limit(total_seconds: int) -> None:
    """Set RLIMIT_CPU to a generous total for batch execution."""
    _current_soft, current_hard = resource.getrlimit(resource.RLIMIT_CPU)
    if current_hard == resource.RLIM_INFINITY:
        target = total_seconds
    else:
        target = min(total_seconds, current_hard)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (target, target))
    except (OSError, ValueError) as exc:
        logger.debug("Unable to apply batch CPU limit %s: %s", target, exc)


# ---------------------------------------------------------------------------
# Bounded I/O helpers
# ---------------------------------------------------------------------------


def _as_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _error_payload(
    error: Any,
    stdout: str = "",
    *,
    stdout_truncated: bool = False,
    compile_success: bool | None = None,
    compile_error: str | None = None,
) -> dict[str, Any]:
    if compile_error is None and compile_success is False and error is not None:
        compile_error = str(error)
    return {
        "return_value": None,
        "return_type": None,
        "stdout": stdout,
        "stdout_truncated": stdout_truncated,
        "error": str(error) if error is not None else None,
        "compile_success": compile_success,
        "compile_error": compile_error,
    }


# ---------------------------------------------------------------------------
# Function-call mode
# ---------------------------------------------------------------------------


def _load_function_from_code(
    code: str, function_name: str
) -> tuple[Callable[..., Any] | None, _BoundedStdoutCapture, str | None]:
    exec_ns: dict[str, Any] = {"__builtins__": _EXEC_BUILTINS}
    stdout_capture = _BoundedStdoutCapture(_stdout_limit_bytes())
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, exec_ns, exec_ns)  # noqa: S102
    except Exception as exc:
        return None, stdout_capture, f"{type(exc).__name__}: {exc}"

    func_obj = exec_ns.get(function_name)
    if not callable(func_obj):
        return (
            None,
            stdout_capture,
            f"Function '{function_name}' not found in executed code",
        )
    return cast(Callable[..., Any], func_obj), stdout_capture, None


def _execute_loaded_function(
    func: Callable[..., Any],
    input_value: Any,
    prefix_stdout: str = "",
    *,
    prefix_stdout_truncated: bool = False,
) -> dict[str, Any]:
    stdout_capture = _BoundedStdoutCapture(_stdout_limit_bytes())
    if prefix_stdout:
        stdout_capture.write(prefix_stdout)
    if prefix_stdout_truncated:
        stdout_capture.truncated = True
    try:
        with contextlib.redirect_stdout(stdout_capture):
            if isinstance(input_value, dict):
                return_value = func(**input_value)
            else:
                return_value = func(input_value)
        return {
            "return_value": _as_jsonable(return_value),
            "return_type": type(return_value).__name__,
            "stdout": stdout_capture.getvalue(),
            "stdout_truncated": stdout_capture.truncated,
            "error": None,
            "compile_success": True,
            "compile_error": None,
        }
    except Exception as exc:
        return _error_payload(
            f"{type(exc).__name__}: {exc}",
            stdout=stdout_capture.getvalue(),
            stdout_truncated=stdout_capture.truncated,
            compile_success=True,
        )


def _execute_single_input(
    code: str, function_name: str, input_value: Any
) -> dict[str, Any]:
    func, setup_stdout, load_error = _load_function_from_code(code, function_name)
    if load_error is not None:
        return _error_payload(
            load_error,
            stdout=setup_stdout.getvalue(),
            stdout_truncated=setup_stdout.truncated,
        )
    assert func is not None
    return _execute_loaded_function(
        func,
        input_value,
        prefix_stdout=setup_stdout.getvalue(),
        prefix_stdout_truncated=setup_stdout.truncated,
    )


def _handle_batch_inputs(
    code: str, function_name: str, input_values: list[Any]
) -> dict[str, Any]:
    """Process multiple inputs with per-input function reload for isolation."""
    if not input_values:
        return {"results": [], "error": None}

    # Pre-flight: check that function loads at all
    _, setup_stdout, load_error = _load_function_from_code(code, function_name)
    if load_error is not None:
        error_payload = _error_payload(
            load_error,
            stdout=setup_stdout.getvalue(),
            stdout_truncated=setup_stdout.truncated,
        )
        return {
            "results": [error_payload.copy() for _ in input_values],
            "error": None,
        }

    # Execute each input with a fresh function load
    results: list[dict[str, Any]] = []
    for input_value in input_values:
        loaded_func, case_stdout, case_error = _load_function_from_code(
            code, function_name
        )
        if case_error is not None:
            results.append(
                _error_payload(
                    case_error,
                    stdout=case_stdout.getvalue(),
                    stdout_truncated=case_stdout.truncated,
                )
            )
            continue
        assert loaded_func is not None
        results.append(
            _execute_loaded_function(
                loaded_func,
                input_value,
                prefix_stdout=case_stdout.getvalue(),
                prefix_stdout_truncated=case_stdout.truncated,
            )
        )
    return {"results": results, "error": None}


def _handle_function_call(req: dict[str, Any]) -> dict[str, Any]:
    code = req["code"]
    function_name = req["function_name"]

    _validate_code_ast(code)

    if "input_values" in req:
        input_values = req["input_values"]
        if not isinstance(input_values, list):
            raise TypeError("input_values must be a list")
        return _handle_batch_inputs(code, function_name, input_values)
    else:
        return _execute_single_input(code, function_name, req["input_value"])


# ---------------------------------------------------------------------------
# Assertion mode (Pro datasets)
# ---------------------------------------------------------------------------


def _handle_assertion(req: dict[str, Any]) -> dict[str, Any]:
    code = req["code"]
    test_code = req["test_code"]
    combined = code + "\n\n" + test_code

    exec_ns: dict[str, Any] = {"__builtins__": _EXEC_BUILTINS}
    stdout_capture = _BoundedStdoutCapture(_stdout_limit_bytes())
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(combined, exec_ns, exec_ns)  # noqa: S102
        return {
            "passed": True,
            "error": None,
            "stdout": stdout_capture.getvalue(),
            "compile_success": True,
            "compile_error": None,
        }
    except AssertionError as exc:
        return {
            "passed": False,
            "error": f"AssertionError: {exc}" if str(exc) else "AssertionError",
            "stdout": stdout_capture.getvalue(),
            "compile_success": True,
            "compile_error": None,
        }
    except SyntaxError as exc:
        return {
            "passed": False,
            "error": f"SyntaxError: {exc}",
            "stdout": stdout_capture.getvalue(),
            "compile_success": False,
            "compile_error": f"SyntaxError: {exc}",
        }
    except Exception as exc:
        return {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "stdout": stdout_capture.getvalue(),
            "compile_success": True,
            "compile_error": None,
        }


# ---------------------------------------------------------------------------
# Unittest mode (ClassEval)
# ---------------------------------------------------------------------------


def _handle_unittest(req: dict[str, Any]) -> dict[str, Any]:
    code = req["code"]
    test_code = req["test_code"]
    test_class_names = req["test_class_names"]
    combined = code + "\n\n" + test_code

    with tempfile.TemporaryDirectory() as tmp_dir:
        namespace: dict[str, Any] = {
            "__builtins__": _EXEC_BUILTINS,
            "__file__": os.path.join(tmp_dir, "__eval__.py"),
        }
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_dir)
            exec(combined, namespace)  # noqa: S102
        except SyntaxError as exc:
            return {
                "all_passed": False,
                "total_tests_run": 0,
                "total_tests_passed": 0,
                "total_tests_failed": 0,
                "total_tests_errored": 0,
                "per_test_class": [],
                "error": f"SyntaxError: {exc}",
            }
        except Exception as exc:
            return {
                "all_passed": False,
                "total_tests_run": 0,
                "total_tests_passed": 0,
                "total_tests_failed": 0,
                "total_tests_errored": 0,
                "per_test_class": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            os.chdir(old_cwd)

        per_test_class: list[dict[str, Any]] = []
        os.chdir(tmp_dir)
        try:
            for raw_name in test_class_names:
                test_class_name = raw_name.strip()
                test_cls = namespace.get(test_class_name)
                if test_cls is None or not (
                    isinstance(test_cls, type)
                    and issubclass(test_cls, unittest.TestCase)
                ):
                    per_test_class.append(
                        {
                            "test_class_name": test_class_name,
                            "tests_run": 0,
                            "tests_passed": 0,
                            "tests_failed": 0,
                            "tests_errored": 0,
                            "failures": [f"Test class {test_class_name!r} not found"],
                            "errors": [],
                            "passed": False,
                        }
                    )
                    continue

                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(test_cls)
                stream = io.StringIO()
                runner = unittest.TextTestRunner(stream=stream, verbosity=0)
                result = runner.run(suite)

                failures = [f"{tc}: {msg}" for tc, msg in result.failures]
                errors = [f"{tc}: {msg}" for tc, msg in result.errors]
                tests_skipped = len(result.skipped)
                tests_passed = (
                    result.testsRun
                    - len(result.failures)
                    - len(result.errors)
                    - tests_skipped
                )

                per_test_class.append(
                    {
                        "test_class_name": test_class_name,
                        "tests_run": result.testsRun,
                        "tests_passed": tests_passed,
                        "tests_failed": len(result.failures),
                        "tests_errored": len(result.errors),
                        "tests_skipped": tests_skipped,
                        "failures": failures,
                        "errors": errors,
                        "passed": (
                            len(result.failures) == 0
                            and len(result.errors) == 0
                            and tests_passed > 0
                        ),
                    }
                )
        finally:
            os.chdir(old_cwd)

        total_run = sum(r["tests_run"] for r in per_test_class)
        total_passed = sum(r["tests_passed"] for r in per_test_class)
        total_failed = sum(r["tests_failed"] for r in per_test_class)
        total_errored = sum(r["tests_errored"] for r in per_test_class)

        return {
            "all_passed": (
                all(r["passed"] for r in per_test_class) and len(per_test_class) > 0
            ),
            "total_tests_run": total_run,
            "total_tests_passed": total_passed,
            "total_tests_failed": total_failed,
            "total_tests_errored": total_errored,
            "per_test_class": per_test_class,
            "error": None,
        }


# ---------------------------------------------------------------------------
# Batch mode — process multiple independent items in one container
# ---------------------------------------------------------------------------


class _ItemTimeoutError(Exception):
    pass


def _alarm_handler(signum: int, frame: Any) -> None:
    raise _ItemTimeoutError("item execution timed out")


def _cleanup_dir(path: str) -> None:
    """Clear a specific directory's contents without removing the directory."""
    try:
        entries = os.listdir(path)
    except OSError:
        return
    for entry in entries:
        entry_path = os.path.join(path, entry)
        try:
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path, ignore_errors=True)
            else:
                os.remove(entry_path)
        except OSError:
            pass


def _dispatch_item(item: dict[str, Any]) -> dict[str, Any]:
    """Route a single item to its mode handler."""
    mode = item.get("mode", "function_call")
    if mode == "function_call":
        return _handle_function_call(item)
    elif mode == "assertion":
        return _handle_assertion(item)
    elif mode == "unittest":
        return _handle_unittest(item)
    else:
        return {"error": f"unknown mode: {mode!r}"}


def _handle_batch(req: dict[str, Any]) -> dict[str, Any]:
    items = req["items"]
    if not isinstance(items, list):
        raise TypeError("batch items must be a list")

    timeout_per_item = int(req.get("timeout_per_item", 30))

    # Set generous CPU budget for the full batch
    _set_batch_cpu_limit(timeout_per_item * len(items))

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _alarm_handler)

    results: list[dict[str, Any]] = []
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory(prefix="nl_code_batch_") as batch_cwd:
            try:
                os.chdir(batch_cwd)
                for item in items:
                    signal.alarm(timeout_per_item)
                    try:
                        os.chdir(batch_cwd)
                        result = _dispatch_item(item)
                        results.append(result)
                    except _ItemTimeoutError:
                        results.append({"error": "item execution timed out"})
                    except Exception as exc:
                        results.append({"error": f"{type(exc).__name__}: {exc}"})
                    finally:
                        signal.alarm(0)
                        os.chdir(batch_cwd)
                        _cleanup_dir(batch_cwd)
            finally:
                os.chdir(original_cwd)
    finally:
        signal.signal(signal.SIGALRM, old_handler)

    return {"results": results, "error": None}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(*, set_limits: bool = True) -> int:
    try:
        _require_docker_execution()
        raw = _read_stdin_bounded(_stdin_limit_bytes())
        req = json.loads(raw)

        mode = req.get("mode", "function_call")

        skip = _worker_execution_config().skip_limits
        if set_limits and not skip and mode != "batch":
            _set_resource_limits()

        if mode == "batch":
            # Set non-CPU limits; batch handler sets CPU limit itself
            if set_limits and not skip:
                _set_resource_limits(skip_cpu=True)
            payload = _handle_batch(req)
        elif mode == "function_call":
            payload = _handle_function_call(req)
        elif mode == "assertion":
            payload = _handle_assertion(req)
        elif mode == "unittest":
            payload = _handle_unittest(req)
        else:
            payload = {"error": f"unknown mode: {mode!r}"}

        print(json.dumps(payload))
        return 0
    except DockerOnlyExecutionError as exc:
        payload = _error_payload(str(exc))
        print(json.dumps(payload))
        return 1
    except OversizedPayloadError as exc:
        payload = _error_payload(
            {
                "type": "oversized_payload",
                "message": str(exc),
                "max_stdin_bytes": exc.max_bytes,
                "actual_stdin_bytes": exc.actual_bytes,
            }
        )
        print(json.dumps(payload))
        return 1
    except Exception as exc:
        payload = _error_payload(f"{type(exc).__name__}: {exc}")
        print(json.dumps(payload))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
