from nl_code.code_execution.models import TestCase
from nl_code.code_execution.runner import EXEC_MODE_LOCAL
from nl_code.code_validation import ValidationResult, validate_generated_code


class TestValidateGeneratedCode:
    def test_valid_no_tests(self) -> None:
        result = validate_generated_code(
            "def foo(x):\n    return x * 2\n",
            "foo",
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert isinstance(result, ValidationResult)
        assert result.is_valid_syntax is True
        assert result.has_expected_function is True
        assert result.test_pass_rate is None
        assert result.elapsed_seconds is not None

    def test_syntax_error(self) -> None:
        result = validate_generated_code(
            "def foo(:\n", "foo", execution_mode=EXEC_MODE_LOCAL
        )
        assert result.is_valid_syntax is False
        assert result.syntax_error is not None
        assert result.has_expected_function is None
        assert result.test_case_results == []

    def test_missing_function(self) -> None:
        result = validate_generated_code(
            "x = 1\n", "foo", execution_mode=EXEC_MODE_LOCAL
        )
        assert result.is_valid_syntax is True
        assert result.has_expected_function is False
        assert result.test_case_results == []

    def test_with_code_fences(self) -> None:
        raw = "Here is the code:\n```python\ndef foo(x):\n    return x + 1\n```\n"
        result = validate_generated_code(raw, "foo", execution_mode=EXEC_MODE_LOCAL)
        assert result.had_code_fences is True
        assert result.is_valid_syntax is True
        assert result.has_expected_function is True

    def test_with_test_cases(self) -> None:
        test_cases = [
            TestCase(input_value=1, expected_output=2),
            TestCase(input_value=5, expected_output=10),
        ]
        result = validate_generated_code(
            "def double(x):\n    return x * 2\n",
            "double",
            test_cases=test_cases,
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert result.test_pass_rate == 1.0
        assert len(result.test_case_results) == 2
        assert all(r.passed for r in result.test_case_results)

    def test_failing_test_cases(self) -> None:
        test_cases = [
            TestCase(input_value=1, expected_output=999),
        ]
        result = validate_generated_code(
            "def foo(x):\n    return x\n",
            "foo",
            test_cases=test_cases,
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert result.test_pass_rate == 0.0
        assert result.test_case_results[0].passed is False

    def test_skips_tests_on_syntax_error(self) -> None:
        test_cases = [TestCase(input_value=1, expected_output=1)]
        result = validate_generated_code(
            "def foo(:\n",
            "foo",
            test_cases=test_cases,
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert result.is_valid_syntax is False
        assert result.test_case_results == []
        assert result.test_pass_rate is None

    def test_skips_tests_on_missing_function(self) -> None:
        test_cases = [TestCase(input_value=1, expected_output=1)]
        result = validate_generated_code(
            "x = 1\n",
            "foo",
            test_cases=test_cases,
            execution_mode=EXEC_MODE_LOCAL,
        )
        assert result.has_expected_function is False
        assert result.test_case_results == []
        assert result.test_pass_rate is None

    def test_raw_output_preserved(self) -> None:
        raw = "```python\ndef foo(x):\n    return x\n```"
        result = validate_generated_code(raw, "foo", execution_mode=EXEC_MODE_LOCAL)
        assert result.raw_output == raw
