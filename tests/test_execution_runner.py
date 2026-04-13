import pytest

from nl_code.code_execution.models import TestCase
from nl_code.code_execution.runner import (
    ExecutionError,
    _values_equal,
    check_compiles,
    run_function_batch,
    run_test_cases,
)


class TestValuesEqual:
    def test_int_equal(self) -> None:
        assert _values_equal(1, 1) is True

    def test_int_not_equal(self) -> None:
        assert _values_equal(1, 2) is False

    def test_float_close(self) -> None:
        assert _values_equal(1.0000000001, 1.0) is True

    def test_float_not_close(self) -> None:
        assert _values_equal(1.5, 2.0) is False

    def test_int_float_cross(self) -> None:
        assert _values_equal(1, 1.0) is True

    def test_string_equal(self) -> None:
        assert _values_equal("a", "a") is True

    def test_list_equal(self) -> None:
        assert _values_equal([1, 2], [1, 2]) is True


class TestCheckCompiles:
    def test_valid(self) -> None:
        ok, err = check_compiles("def foo():\n    return 1\n")
        assert ok is True
        assert err is None

    def test_invalid(self) -> None:
        ok, err = check_compiles("def foo(:\n")
        assert ok is False
        assert err is not None
        assert "SyntaxError" in err


class TestRunFunctionBatch:
    def test_basic(self) -> None:
        results = run_function_batch(
            "def double(x):\n    return x * 2\n",
            "double",
            [1, 2, 3],
        )
        assert len(results) == 3
        assert results[0].return_value == 2
        assert results[1].return_value == 4
        assert results[2].return_value == 6

    def test_empty_inputs(self) -> None:
        results = run_function_batch("def foo(x):\n    return x\n", "foo", [])
        assert results == []

    def test_runtime_error(self) -> None:
        results = run_function_batch(
            "def boom(x):\n    raise ValueError('nope')\n",
            "boom",
            [1],
        )
        assert len(results) == 1
        assert results[0].error is not None
        assert "ValueError" in results[0].error

    def test_timeout(self) -> None:
        # Use a subprocess-level timeout (sleep in a forked process to avoid
        # the worker's RLIMIT_CPU from killing it before the timeout fires).
        import subprocess
        from unittest.mock import patch

        with (
            patch(
                "nl_code.code_execution.runner.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="test", timeout=0.01),
            ),
            pytest.raises(ExecutionError, match="timeout"),
        ):
            run_function_batch(
                "def foo(x):\n    return x\n", "foo", [1], timeout_seconds=0.01
            )


class TestRunTestCases:
    def test_all_pass(self) -> None:
        test_cases = [
            TestCase(input_value=1, expected_output=2),
            TestCase(input_value=2, expected_output=4),
        ]
        results, rate = run_test_cases(
            "def double(x):\n    return x * 2\n",
            "double",
            test_cases,
        )
        assert rate == 1.0
        assert all(r.passed for r in results)

    def test_partial_pass(self) -> None:
        test_cases = [
            TestCase(input_value=1, expected_output=2),
            TestCase(input_value=2, expected_output=999),
        ]
        results, rate = run_test_cases(
            "def double(x):\n    return x * 2\n",
            "double",
            test_cases,
        )
        assert rate == 0.5
        assert results[0].passed is True
        assert results[1].passed is False

    def test_empty_cases(self) -> None:
        results, rate = run_test_cases("def foo(x):\n    return x\n", "foo", [])
        assert results == []
        assert rate == 0.0
