import io
import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from nl_code.code_execution.worker import (
    _BoundedStdoutCapture,
    _validate_code_ast,
    main,
)


class TestAstValidator:
    def test_allows_normal_code(self) -> None:
        _validate_code_ast("x = 1\n")

    def test_blocks_async_def(self) -> None:
        with pytest.raises(ValueError, match="AsyncFunctionDef"):
            _validate_code_ast("async def foo():\n    pass\n")

    def test_blocks_async_constructs(self) -> None:
        with pytest.raises(ValueError, match="not allowed"):
            _validate_code_ast("async def foo():\n    await bar()\n")

    def test_blocks_dunder_access(self) -> None:
        with pytest.raises(ValueError, match="dunder"):
            _validate_code_ast("x.__class__\n")

    def test_allows_normal_attributes(self) -> None:
        _validate_code_ast("x.foo\n")


class TestBoundedStdoutCapture:
    def test_basic_capture(self) -> None:
        cap = _BoundedStdoutCapture(1024)
        cap.write("hello")
        assert cap.getvalue() == "hello"
        assert cap.truncated is False

    def test_truncation(self) -> None:
        cap = _BoundedStdoutCapture(5)
        cap.write("hello world")
        assert cap.truncated is True
        assert len(cap.getvalue().encode("utf-8")) <= 5

    def test_multiple_writes(self) -> None:
        cap = _BoundedStdoutCapture(1024)
        cap.write("a")
        cap.write("b")
        assert cap.getvalue() == "ab"

    def test_exact_limit(self) -> None:
        cap = _BoundedStdoutCapture(5)
        cap.write("hello")
        assert cap.truncated is False
        assert cap.getvalue() == "hello"


def _run_worker(request: dict) -> dict:
    """Run worker main() with a mock stdin/stdout and return parsed JSON."""
    request_bytes = json.dumps(request).encode("utf-8")
    mock_stdin = MagicMock()
    mock_stdin.buffer = io.BytesIO(request_bytes)
    stdout_capture = io.StringIO()

    with (
        patch.object(sys, "stdin", mock_stdin),
        patch.object(sys, "stdout", stdout_capture),
    ):
        main(set_limits=False)

    return json.loads(stdout_capture.getvalue())


class TestWorkerMain:
    def test_single_input(self) -> None:
        result = _run_worker(
            {
                "code": "def double(x):\n    return x * 2\n",
                "function_name": "double",
                "input_value": 5,
            }
        )
        assert result["error"] is None
        assert result["return_value"] == 10
        assert result["return_type"] == "int"

    def test_batch_inputs(self) -> None:
        result = _run_worker(
            {
                "code": "def double(x):\n    return x * 2\n",
                "function_name": "double",
                "input_values": [1, 2, 3],
            }
        )
        assert result["error"] is None
        assert len(result["results"]) == 3
        assert result["results"][0]["return_value"] == 2
        assert result["results"][1]["return_value"] == 4
        assert result["results"][2]["return_value"] == 6

    def test_compile_only_valid(self) -> None:
        result = _run_worker(
            {
                "code": "def foo():\n    return 1\n",
                "function_name": "foo",
                "compile_only": True,
            }
        )
        assert result["error"] is None
        assert result["results"] == []

    def test_compile_only_invalid(self) -> None:
        result = _run_worker(
            {
                "code": "def foo(:\n",
                "function_name": "foo",
                "compile_only": True,
            }
        )
        assert result["error"] is not None
        assert "SyntaxError" in result["error"]

    def test_runtime_error(self) -> None:
        result = _run_worker(
            {
                "code": "def boom(x):\n    raise ValueError('nope')\n",
                "function_name": "boom",
                "input_value": 1,
            }
        )
        assert result["error"] is not None
        assert "ValueError" in result["error"]

    def test_function_not_found(self) -> None:
        result = _run_worker(
            {
                "code": "x = 1\n",
                "function_name": "missing",
                "input_value": 1,
            }
        )
        assert result["error"] is not None
        assert "not found" in result["error"]

    def test_async_code_blocked(self) -> None:
        result = _run_worker(
            {
                "code": "async def foo(x):\n    return x\n",
                "function_name": "foo",
                "input_value": 1,
            }
        )
        assert result["error"] is not None
        assert "not allowed" in result["error"]

    def test_stdout_captured(self) -> None:
        result = _run_worker(
            {
                "code": 'def greet(x):\n    print("hi")\n    return x\n',
                "function_name": "greet",
                "input_value": 1,
            }
        )
        assert result["error"] is None
        assert "hi" in result["stdout"]
