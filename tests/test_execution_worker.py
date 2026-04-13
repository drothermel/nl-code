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


class TestDockerOnlyGuard:
    def test_rejects_missing_runner_flag(self) -> None:
        stdout_capture = io.StringIO()
        mock_stdin = MagicMock()
        mock_stdin.buffer = io.BytesIO(b"{}")

        with (
            patch.object(sys, "stdin", mock_stdin),
            patch.object(sys, "stdout", stdout_capture),
            patch(
                "nl_code.code_execution.worker._is_running_in_container",
                return_value=True,
            ),
        ):
            exit_code = main(set_limits=False)

        payload = json.loads(stdout_capture.getvalue())
        assert exit_code == 1
        assert "NL_CODE_EVAL_IN_DOCKER" in payload["error"]


def _run_worker(request: dict) -> dict:
    """Run worker main() with a mock stdin/stdout and return parsed JSON."""
    request_bytes = json.dumps(request).encode("utf-8")
    mock_stdin = MagicMock()
    mock_stdin.buffer = io.BytesIO(request_bytes)
    stdout_capture = io.StringIO()

    with (
        patch.dict("os.environ", {"NL_CODE_EVAL_IN_DOCKER": "1"}),
        patch.object(sys, "stdin", mock_stdin),
        patch.object(sys, "stdout", stdout_capture),
        patch(
            "nl_code.code_execution.worker._is_running_in_container", return_value=True
        ),
    ):
        main(set_limits=False)

    return json.loads(stdout_capture.getvalue())


# ---------------------------------------------------------------------------
# Function-call mode
# ---------------------------------------------------------------------------


class TestFunctionCallMode:
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
        assert result["compile_success"] is True

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

    def test_batch_compile_success(self) -> None:
        result = _run_worker(
            {
                "code": "def f(x):\n    return x + 1\n",
                "function_name": "f",
                "input_values": [1],
            }
        )
        assert result["results"][0]["compile_success"] is True
        assert result["results"][0]["compile_error"] is None

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
        assert result["compile_success"] is True

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

    def test_batch_isolation(self) -> None:
        """Each input in a batch gets a fresh function load."""
        # Code with a mutable default that could leak between calls
        code = (
            "counter = [0]\ndef count(x):\n    counter[0] += 1\n    return counter[0]\n"
        )
        result = _run_worker(
            {
                "code": code,
                "function_name": "count",
                "input_values": [1, 2, 3],
            }
        )
        # Each call reloads, so counter always returns 1
        assert result["error"] is None
        for r in result["results"]:
            assert r["return_value"] == 1

    def test_import_works(self) -> None:
        """Full builtins allows import statements."""
        result = _run_worker(
            {
                "code": "import math\ndef root(x):\n    return math.sqrt(x)\n",
                "function_name": "root",
                "input_value": 4,
            }
        )
        assert result["error"] is None
        assert result["return_value"] == 2.0


# ---------------------------------------------------------------------------
# Assertion mode
# ---------------------------------------------------------------------------


class TestAssertionMode:
    def test_passing(self) -> None:
        result = _run_worker(
            {
                "mode": "assertion",
                "code": "def add(a, b):\n    return a + b\n",
                "test_code": "assert add(1, 2) == 3\nassert add(0, 0) == 0\n",
            }
        )
        assert result["passed"] is True
        assert result["error"] is None
        assert result["compile_success"] is True

    def test_failing(self) -> None:
        result = _run_worker(
            {
                "mode": "assertion",
                "code": "def add(a, b):\n    return a - b\n",
                "test_code": "assert add(1, 2) == 3\n",
            }
        )
        assert result["passed"] is False
        assert result["error"] is not None
        assert "AssertionError" in result["error"]
        assert result["compile_success"] is True

    def test_runtime_error(self) -> None:
        result = _run_worker(
            {
                "mode": "assertion",
                "code": "def f(x):\n    raise ValueError('bad')\n",
                "test_code": "f(1)\n",
            }
        )
        assert result["passed"] is False
        assert "ValueError" in result["error"]

    def test_syntax_error_in_code(self) -> None:
        result = _run_worker(
            {
                "mode": "assertion",
                "code": "def f(:\n",
                "test_code": "assert True\n",
            }
        )
        assert result["passed"] is False
        assert result["compile_success"] is False
        assert result["compile_error"] is not None

    def test_dunder_access_is_allowed(self) -> None:
        result = _run_worker(
            {
                "mode": "assertion",
                "code": "class Foo:\n    pass\n",
                "test_code": "assert Foo().__class__.__name__ == 'Foo'\n",
            }
        )
        assert result["passed"] is True
        assert result["error"] is None


# ---------------------------------------------------------------------------
# Unittest mode
# ---------------------------------------------------------------------------


class TestUnittestMode:
    def test_passing(self) -> None:
        code = "class Calculator:\n    def add(self, a, b):\n        return a + b\n"
        test_code = (
            "import unittest\n"
            "class TestCalc(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        c = Calculator()\n"
            "        self.assertEqual(c.add(1, 2), 3)\n"
        )
        result = _run_worker(
            {
                "mode": "unittest",
                "code": code,
                "test_code": test_code,
                "test_class_names": ["TestCalc"],
            }
        )
        assert result["all_passed"] is True
        assert result["total_tests_run"] == 1
        assert result["total_tests_passed"] == 1
        assert result["error"] is None
        assert len(result["per_test_class"]) == 1
        assert result["per_test_class"][0]["passed"] is True

    def test_failing(self) -> None:
        code = "class Calculator:\n    def add(self, a, b):\n        return a - b\n"
        test_code = (
            "import unittest\n"
            "class TestCalc(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        c = Calculator()\n"
            "        self.assertEqual(c.add(1, 2), 3)\n"
        )
        result = _run_worker(
            {
                "mode": "unittest",
                "code": code,
                "test_code": test_code,
                "test_class_names": ["TestCalc"],
            }
        )
        assert result["all_passed"] is False
        assert result["total_tests_failed"] == 1

    def test_missing_test_class(self) -> None:
        result = _run_worker(
            {
                "mode": "unittest",
                "code": "x = 1\n",
                "test_code": "",
                "test_class_names": ["TestMissing"],
            }
        )
        assert result["all_passed"] is False
        assert result["per_test_class"][0]["passed"] is False
        assert "not found" in result["per_test_class"][0]["failures"][0]

    def test_syntax_error(self) -> None:
        result = _run_worker(
            {
                "mode": "unittest",
                "code": "def f(:\n",
                "test_code": "",
                "test_class_names": ["TestFoo"],
            }
        )
        assert result["all_passed"] is False
        assert "SyntaxError" in (result.get("error") or "")

    def test_dunder_access_is_allowed(self) -> None:
        result = _run_worker(
            {
                "mode": "unittest",
                "code": "class Foo:\n    pass\n",
                "test_code": (
                    "import unittest\n"
                    "class TestFoo(unittest.TestCase):\n"
                    "    def test_name(self):\n"
                    "        self.assertEqual(Foo().__class__.__name__, 'Foo')\n"
                ),
                "test_class_names": ["TestFoo"],
            }
        )
        assert result["all_passed"] is True
        assert result["error"] is None

    def test_executes_inside_temp_dir(self) -> None:
        result = _run_worker(
            {
                "mode": "unittest",
                "code": (
                    "from pathlib import Path\n"
                    "Path('artifact.txt').write_text('ok', encoding='utf-8')\n"
                ),
                "test_code": (
                    "import unittest\n"
                    "from pathlib import Path\n"
                    "class TestFoo(unittest.TestCase):\n"
                    "    def test_artifact(self):\n"
                    "        self.assertTrue(Path('artifact.txt').is_file())\n"
                ),
                "test_class_names": ["TestFoo"],
            }
        )
        assert result["all_passed"] is True
        assert result["error"] is None


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


class TestBatchMode:
    def test_mixed_items(self) -> None:
        result = _run_worker(
            {
                "mode": "batch",
                "timeout_per_item": 5,
                "items": [
                    {
                        "mode": "function_call",
                        "code": "def f(x):\n    return x + 1\n",
                        "function_name": "f",
                        "input_values": [1, 2],
                    },
                    {
                        "mode": "assertion",
                        "code": "def g(x):\n    return x * 2\n",
                        "test_code": "assert g(3) == 6\n",
                    },
                ],
            }
        )
        assert result["error"] is None
        assert len(result["results"]) == 2
        # First item: function_call
        assert result["results"][0]["results"][0]["return_value"] == 2
        # Second item: assertion
        assert result["results"][1]["passed"] is True

    def test_item_error_doesnt_stop_batch(self) -> None:
        result = _run_worker(
            {
                "mode": "batch",
                "timeout_per_item": 5,
                "items": [
                    {
                        "mode": "function_call",
                        "code": "def f(x):\n    raise ValueError('boom')\n",
                        "function_name": "f",
                        "input_values": [1],
                    },
                    {
                        "mode": "function_call",
                        "code": "def g(x):\n    return x\n",
                        "function_name": "g",
                        "input_values": [42],
                    },
                ],
            }
        )
        assert result["error"] is None
        assert len(result["results"]) == 2
        # First item has an error
        assert result["results"][0]["results"][0]["error"] is not None
        # Second item succeeds
        assert result["results"][1]["results"][0]["return_value"] == 42

    def test_cleans_batch_cwd_between_items(self) -> None:
        result = _run_worker(
            {
                "mode": "batch",
                "timeout_per_item": 5,
                "items": [
                    {
                        "mode": "assertion",
                        "code": (
                            "from pathlib import Path\n"
                            "Path('leftover.txt').write_text('x', encoding='utf-8')\n"
                        ),
                        "test_code": "assert True\n",
                    },
                    {
                        "mode": "function_call",
                        "code": (
                            "from pathlib import Path\n"
                            "def exists(_):\n"
                            "    return Path('leftover.txt').exists()\n"
                        ),
                        "function_name": "exists",
                        "input_values": [0],
                    },
                ],
            }
        )
        assert result["error"] is None
        assert result["results"][0]["passed"] is True
        assert result["results"][1]["results"][0]["return_value"] is False
