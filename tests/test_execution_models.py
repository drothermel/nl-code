from nl_code.code_execution.models import ExecutionResult, TestCase, TestCaseResult


class TestExecutionResult:
    def test_minimal(self) -> None:
        r = ExecutionResult(input_value=1)
        assert r.input_value == 1
        assert r.return_value is None
        assert r.error is None
        assert r.stdout == ""

    def test_full(self) -> None:
        r = ExecutionResult(
            input_value=[1, 2],
            return_value=3,
            return_type="int",
            stdout="hello\n",
            error=None,
        )
        assert r.return_value == 3
        assert r.return_type == "int"


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

    def test_passed(self) -> None:
        r = TestCaseResult(
            input_value=1,
            expected_output=2,
            actual_output=2,
            passed=True,
        )
        assert r.passed is True
