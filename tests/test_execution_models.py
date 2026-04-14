from nl_code.code_execution.models import (
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


class TestExecutionResult:
    def test_minimal(self) -> None:
        r = ExecutionResult(input_value=1)
        assert r.input_value == 1
        assert r.return_value is None
        assert r.error is None
        assert r.stdout == ""
        assert r.compile_success is None
        assert r.compile_error is None

    def test_full(self) -> None:
        r = ExecutionResult(
            input_value=[1, 2],
            return_value=3,
            return_type="int",
            stdout="hello\n",
            error=None,
            compile_success=True,
            compile_error=None,
        )
        assert r.return_value == 3
        assert r.return_type == "int"
        assert r.compile_success is True

    def test_compile_failure(self) -> None:
        r = ExecutionResult(
            input_value=1,
            compile_success=False,
            compile_error="SyntaxError: invalid syntax",
            error="SyntaxError: invalid syntax",
        )
        assert r.compile_success is False
        assert r.compile_error is not None


class TestTestCase:
    def test_construction(self) -> None:
        tc = TestCase(input_value=[1, 2], expected_output=3)
        assert tc.input_value == [1, 2]
        assert tc.expected_output == 3


class TestTestCaseResult:
    def test_defaults(self) -> None:
        r = TestCaseResult(input_value=1, expected_output=2)
        assert r.passed is False
        assert r.actual_output is None
        assert r.error is None
        assert r.compile_success is None
        assert r.compile_error is None

    def test_passed(self) -> None:
        r = TestCaseResult(
            input_value=1,
            expected_output=2,
            actual_output=2,
            passed=True,
        )
        assert r.passed is True


class TestCodeExecutionInfrastructureError:
    def test_construction(self) -> None:
        err = CodeExecutionInfrastructureError(
            stage="docker_unavailable",
            execution_mode="docker_worker",
            detail="Docker daemon not responding",
        )
        assert err.stage == "docker_unavailable"
        assert err.execution_mode == "docker_worker"
        assert err.detail == "Docker daemon not responding"
        assert "docker_unavailable" in str(err)
        assert "docker_worker" in str(err)

    def test_is_runtime_error(self) -> None:
        err = CodeExecutionInfrastructureError(
            stage="test", execution_mode="test", detail="test"
        )
        assert isinstance(err, RuntimeError)


class TestAssertionTestResult:
    def test_passed(self) -> None:
        r = AssertionTestResult(passed=True)
        assert r.error is None
        assert r.stdout == ""
        assert r.compile_success is None

    def test_failed(self) -> None:
        r = AssertionTestResult(
            passed=False,
            error="AssertionError",
            compile_success=True,
        )
        assert r.passed is False


class TestUnittestResult:
    def test_all_passed(self) -> None:
        detail = UnittestTestDetail(
            test_class_name="TestFoo",
            tests_run=2,
            tests_passed=2,
            tests_failed=0,
            tests_errored=0,
            passed=True,
        )
        r = UnittestResult(
            all_passed=True,
            total_tests_run=2,
            total_tests_passed=2,
            total_tests_failed=0,
            total_tests_errored=0,
            per_test_class=[detail],
        )
        assert r.all_passed is True
        assert r.error is None

    def test_error(self) -> None:
        r = UnittestResult(
            all_passed=False,
            total_tests_run=0,
            total_tests_passed=0,
            total_tests_failed=0,
            total_tests_errored=0,
            per_test_class=[],
            error="SyntaxError: invalid syntax",
        )
        assert r.all_passed is False
        assert r.error is not None


class TestBatchItemModels:
    def test_function_call_batch_item(self) -> None:
        item = FunctionCallBatchItem(
            code="def f(x): return x",
            function_name="f",
            test_cases=[TestCase(input_value=1, expected_output=1)],
        )
        assert item.function_name == "f"
        assert len(item.test_cases) == 1

    def test_assertion_batch_item(self) -> None:
        item = AssertionBatchItem(
            code="def f(x): return x", test_code="assert f(1) == 1"
        )
        assert item.code == "def f(x): return x"

    def test_unittest_batch_item(self) -> None:
        item = UnittestBatchItem(
            code="class Foo: pass",
            test_code="import unittest\nclass TestFoo(unittest.TestCase): pass",
            test_class_names=["TestFoo"],
        )
        assert item.test_class_names == ["TestFoo"]
