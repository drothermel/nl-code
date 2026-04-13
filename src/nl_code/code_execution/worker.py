"""Sandbox worker for isolated Python code execution.

Reads a JSON request from stdin, executes Python code in a restricted
environment, writes a JSON result to stdout. Designed to run as a subprocess
with resource limits applied via resource.setrlimit().

Protocol:
  stdin:  JSON with {code, function_name, compile_only?, input_value?, input_values?}
  stdout: JSON with {results: [...], error: ...}

This module has ZERO imports from the nl_code package — it must remain a
standalone script that runs in an isolated subprocess.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import json
import logging
import os
import resource
import sys
from typing import Any, Callable, cast

logger = logging.getLogger(__name__)

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


def _set_resource_limits() -> None:
    cpu_seconds = int(os.getenv("NL_CODE_EVAL_CPU_SECONDS", "2"))
    memory_bytes = int(os.getenv("NL_CODE_EVAL_MEMORY_BYTES", "268435456"))
    file_bytes = int(os.getenv("NL_CODE_EVAL_FILE_BYTES", "1048576"))
    process_count = int(os.getenv("NL_CODE_EVAL_NPROC", "64"))

    limits: list[tuple[int, int, int]] = [
        (resource.RLIMIT_CPU, cpu_seconds, cpu_seconds),
        (resource.RLIMIT_AS, memory_bytes, memory_bytes),
        (resource.RLIMIT_FSIZE, file_bytes, file_bytes),
        (resource.RLIMIT_NPROC, process_count, process_count),
    ]
    for limit_name, soft, hard in limits:
        _current_soft, current_hard = resource.getrlimit(limit_name)
        if current_hard == resource.RLIM_INFINITY:
            target_soft, target_hard = soft, hard
        else:
            target_soft = min(soft, current_hard)
            target_hard = min(hard, current_hard)
        try:
            resource.setrlimit(limit_name, (target_soft, target_hard))
        except (OSError, ValueError) as exc:
            logger.debug(
                "Unable to apply resource limit %s=%s/%s: %s",
                limit_name,
                target_soft,
                target_hard,
                exc,
            )


def _as_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value


class OversizedPayloadError(ValueError):
    def __init__(self, max_bytes: int, actual_bytes: int) -> None:
        super().__init__(
            f"stdin payload exceeds limit ({actual_bytes} > {max_bytes} bytes)"
        )
        self.max_bytes = max_bytes
        self.actual_bytes = actual_bytes


def _stdin_limit_bytes() -> int:
    return int(os.getenv("NL_CODE_EVAL_MAX_STDIN_BYTES", "1048576"))


def _stdout_limit_bytes() -> int:
    return int(os.getenv("NL_CODE_EVAL_MAX_STDOUT_BYTES", "1048576"))


def _read_stdin_bounded(max_bytes: int) -> str:
    buffer = getattr(sys.stdin, "buffer", None)
    if buffer is not None:
        raw_bytes = buffer.read(max_bytes + 1)
    else:
        raw_bytes = sys.stdin.read(max_bytes + 1).encode("utf-8")
    if len(raw_bytes) > max_bytes:
        raise OversizedPayloadError(max_bytes=max_bytes, actual_bytes=len(raw_bytes))
    return raw_bytes.decode("utf-8")


class _BoundedStdoutCapture:
    def __init__(self, limit_bytes: int) -> None:
        self._limit_bytes = limit_bytes
        self._used_bytes = 0
        self._parts: list[str] = []
        self.truncated = False

    def write(self, value: str) -> int:
        encoded = value.encode("utf-8")
        remaining = self._limit_bytes - self._used_bytes
        if remaining <= 0:
            self.truncated = True
            return len(value)
        if len(encoded) <= remaining:
            self._parts.append(value)
            self._used_bytes += len(encoded)
            return len(value)
        self.truncated = True
        kept = encoded[:remaining].decode("utf-8", errors="ignore")
        if kept:
            self._parts.append(kept)
            self._used_bytes += len(kept.encode("utf-8"))
        return len(value)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return "".join(self._parts)


def _error_payload(
    error: Any, stdout: str = "", *, stdout_truncated: bool = False
) -> dict[str, Any]:
    return {
        "return_value": None,
        "return_type": None,
        "stdout": stdout,
        "stdout_truncated": stdout_truncated,
        "error": str(error) if error is not None else None,
    }


def _load_function_from_code(
    code: str, function_name: str
) -> tuple[Callable[..., Any] | None, _BoundedStdoutCapture, str | None]:
    exec_ns: dict[str, Any] = {"__builtins__": _EXEC_BUILTINS}
    stdout_capture = _BoundedStdoutCapture(_stdout_limit_bytes())
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, exec_ns, exec_ns)
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
        }
    except Exception as exc:
        return _error_payload(
            f"{type(exc).__name__}: {exc}",
            stdout=stdout_capture.getvalue(),
            stdout_truncated=stdout_capture.truncated,
        )


def _execute_request(code: str, function_name: str, input_value: Any) -> dict[str, Any]:
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


def main(*, set_limits: bool = True) -> int:
    try:
        raw = _read_stdin_bounded(_stdin_limit_bytes())
        req = json.loads(raw)
        code = req["code"]
        function_name = req["function_name"]

        skip = os.getenv("NL_CODE_EVAL_SKIP_LIMITS", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if set_limits and not skip:
            _set_resource_limits()

        _validate_code_ast(code)

        if req.get("compile_only"):
            try:
                compile(code, "<generated>", "exec")
                payload = {"results": [], "error": None}
            except Exception as exc:
                payload = {"results": [], "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(payload))
            return 0

        if "input_values" in req:
            input_values = req["input_values"]
            if not isinstance(input_values, list):
                raise TypeError("input_values must be a list")
            if not input_values:
                print(json.dumps({"results": [], "error": None}))
                return 0

            func, setup_stdout, load_error = _load_function_from_code(
                code, function_name
            )
            if load_error is not None:
                error_payload = _error_payload(
                    load_error,
                    stdout=setup_stdout.getvalue(),
                    stdout_truncated=setup_stdout.truncated,
                )
                payload = {
                    "results": [error_payload.copy() for _ in input_values],
                    "error": None,
                }
                print(json.dumps(payload))
                return 0

            assert func is not None
            setup_stdout_text = setup_stdout.getvalue()
            results: list[dict[str, Any]] = []
            for idx, input_value in enumerate(input_values):
                results.append(
                    _execute_loaded_function(
                        func,
                        input_value,
                        prefix_stdout=setup_stdout_text if idx == 0 else "",
                        prefix_stdout_truncated=setup_stdout.truncated
                        if idx == 0
                        else False,
                    )
                )
            print(json.dumps({"results": results, "error": None}))
            return 0

        input_value = req["input_value"]
        result = _execute_request(code, function_name, input_value)
        print(json.dumps(result))
        return 0
    except OversizedPayloadError as exc:
        print(json.dumps(_error_payload(str(exc))))
        return 1
    except Exception as exc:
        print(json.dumps(_error_payload(f"{type(exc).__name__}: {exc}")))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
