import pytest

from nl_code.code_execution.models import (
    AssertionBatchItem,
    CodeExecutionInfrastructureError,
    FunctionCallBatchItem,
    TestCase,
)
from nl_code.code_execution.runner import (
    EXEC_MODE_LOCAL,
    _values_equal,
    batch_run_assertion_tests,
    batch_run_test_cases,
    check_compiles,
    run_assertion_test,
    run_function_batch,
    run_test_cases,
    run_unittest_test,
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

    def test_float_int_cross(self) -> None:
        assert _values_equal(1.0, 1) is True

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


# ---------------------------------------------------------------------------
# Function-call mode (local worker)
# ---------------------------------------------------------------------------


class TestRunFunctionBatch:
    def test_basic(self) -> None:
        results = run_function_batch(
            "def double(x):\n    return x * 2\n",
            "double",
            [1, 2, 3],
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert len(results) == 3
        assert results[0].return_value == 2
        assert results[1].return_value == 4
        assert results[2].return_value == 6

    def test_compile_success_field(self) -> None:
        results = run_function_batch(
            "def f(x):\n    return x\n",
            "f",
            [1],
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert results[0].compile_success is True
        assert results[0].compile_error is None

    def test_empty_inputs(self) -> None:
        results = run_function_batch(
            "def foo(x):\n    return x\n",
            "foo",
            [],
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert results == []

    def test_runtime_error(self) -> None:
        results = run_function_batch(
            "def boom(x):\n    raise ValueError('nope')\n",
            "boom",
            [1],
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert len(results) == 1
        assert results[0].error is not None
        assert "ValueError" in results[0].error

    def test_timeout(self) -> None:
        import subprocess
        from unittest.mock import patch

        with (
            patch(
                "nl_code.code_execution.runner.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="test", timeout=0.01),
            ),
            pytest.raises(CodeExecutionInfrastructureError, match="timed out"),
        ):
            run_function_batch(
                "def foo(x):\n    return x\n",
                "foo",
                [1],
                timeout_seconds=0.01,
                execution_mode=EXEC_MODE_LOCAL,
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
            execution_mode=EXEC_MODE_LOCAL,
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
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert rate == 0.5
        assert results[0].passed is True
        assert results[1].passed is False

    def test_empty_cases(self) -> None:
        results, rate = run_test_cases(
            "def foo(x):\n    return x\n",
            "foo",
            [],
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert results == []
        assert rate == 0.0

    def test_compile_fields_propagated(self) -> None:
        test_cases = [TestCase(input_value=1, expected_output=1)]
        results, _ = run_test_cases(
            "def f(x):\n    return x\n",
            "f",
            test_cases,
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert results[0].compile_success is True


# ---------------------------------------------------------------------------
# Assertion mode (local worker)
# ---------------------------------------------------------------------------


class TestRunAssertionTest:
    def test_passing(self) -> None:
        result = run_assertion_test(
            "def add(a, b):\n    return a + b\n",
            "assert add(1, 2) == 3\n",
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert result.passed is True
        assert result.error is None

    def test_failing(self) -> None:
        result = run_assertion_test(
            "def add(a, b):\n    return a - b\n",
            "assert add(1, 2) == 3\n",
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert result.passed is False
        assert "AssertionError" in (result.error or "")


# ---------------------------------------------------------------------------
# Unittest mode (local worker)
# ---------------------------------------------------------------------------


class TestRunUnittestTest:
    def test_passing(self) -> None:
        code = "class Calc:\n    def add(self, a, b):\n        return a + b\n"
        test_code = (
            "import unittest\n"
            "class TestCalc(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(Calc().add(1, 2), 3)\n"
        )
        result = run_unittest_test(
            code,
            test_code,
            ["TestCalc"],
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert result.all_passed is True
        assert result.total_tests_run == 1

    def test_failing(self) -> None:
        code = "class Calc:\n    def add(self, a, b):\n        return 0\n"
        test_code = (
            "import unittest\n"
            "class TestCalc(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(Calc().add(1, 2), 3)\n"
        )
        result = run_unittest_test(
            code,
            test_code,
            ["TestCalc"],
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert result.all_passed is False
        assert result.total_tests_failed == 1


# ---------------------------------------------------------------------------
# Batch API (local worker)
# ---------------------------------------------------------------------------


class TestBatchRunTestCases:
    def test_basic(self) -> None:
        items = [
            FunctionCallBatchItem(
                code="def f(x):\n    return x + 1\n",
                function_name="f",
                test_cases=[
                    TestCase(input_value=1, expected_output=2),
                    TestCase(input_value=5, expected_output=6),
                ],
            ),
            FunctionCallBatchItem(
                code="def g(x):\n    return x * 2\n",
                function_name="g",
                test_cases=[TestCase(input_value=3, expected_output=6)],
            ),
        ]
        results = batch_run_test_cases(
            items,
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert len(results) == 2
        tc_results_0, rate_0 = results[0]
        assert rate_0 == 1.0
        assert len(tc_results_0) == 2
        tc_results_1, rate_1 = results[1]
        assert rate_1 == 1.0

    def test_empty(self) -> None:
        assert batch_run_test_cases([], execution_mode=EXEC_MODE_LOCAL) == []


class TestBatchRunAssertionTests:
    def test_basic(self) -> None:
        items = [
            AssertionBatchItem(
                code="def f(x):\n    return x + 1\n",
                test_code="assert f(1) == 2\n",
            ),
            AssertionBatchItem(
                code="def g(x):\n    return x - 1\n",
                test_code="assert g(1) == 99\n",
            ),
        ]
        results = batch_run_assertion_tests(
            items,
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert len(results) == 2
        assert results[0].passed is True
        assert results[1].passed is False


# ---------------------------------------------------------------------------
# Execution mode validation
# ---------------------------------------------------------------------------


class TestExecutionModeValidation:
    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid execution_mode"):
            run_function_batch(
                "def f(x): return x",
                "f",
                [1],
                execution_mode="invalid",
            )
