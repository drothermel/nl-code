import textwrap

import pytest

from nl_code.code_analysis import (
    FunctionAnalysis,
    analyze_code_style,
    analyze_function,
    check_function_exists,
    check_has_assert,
    check_has_print,
    check_has_raise,
    check_has_return,
    check_python_syntax,
    count_control_structures,
    extract_from_code_fences,
    extract_inline_comments,
    extract_string_literals,
    get_parameter_names,
    get_return_type_annotation,
)

SAMPLE_FN = textwrap.dedent("""\
    def foo(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        # a comment
        return a + b
""")

COMPLEX_FN = textwrap.dedent("""\
    def bar(items: list[int], threshold: int = 10) -> list[int]:
        result = []
        for item in items:
            if item > threshold:
                result.append(item)
            elif item == threshold:
                print("exact match")
        return result
""")


class TestExtractFromCodeFences:
    def test_with_fences(self) -> None:
        text = "Here is code:\n```python\nx = 1\n```\nDone."
        code, had = extract_from_code_fences(text)
        assert had is True
        assert code == "x = 1"

    def test_without_fences(self) -> None:
        text = "x = 1"
        code, had = extract_from_code_fences(text)
        assert had is False
        assert code == "x = 1"

    def test_no_language_tag(self) -> None:
        text = "```\ndef foo():\n    pass\n```"
        code, had = extract_from_code_fences(text)
        assert had is True
        assert "def foo" in code


class TestCheckPythonSyntax:
    def test_valid(self) -> None:
        valid, err = check_python_syntax("x = 1\n")
        assert valid is True
        assert err is None

    def test_invalid(self) -> None:
        valid, err = check_python_syntax("def (:\n")
        assert valid is False
        assert err is not None
        assert "SyntaxError" in err


class TestCheckFunctionExists:
    def test_exists(self) -> None:
        assert check_function_exists(SAMPLE_FN, "foo") is True

    def test_not_exists(self) -> None:
        assert check_function_exists(SAMPLE_FN, "bar") is False

    def test_syntax_error(self) -> None:
        assert check_function_exists("def (:", "foo") is False

    def test_async(self) -> None:
        assert check_function_exists("async def baz():\n    pass\n", "baz") is True


class TestCheckHasReturn:
    def test_has_return(self) -> None:
        assert check_has_return(SAMPLE_FN, "foo") is True

    def test_no_return(self) -> None:
        code = "def foo():\n    pass\n"
        assert check_has_return(code, "foo") is False

    def test_missing_function(self) -> None:
        assert check_has_return(SAMPLE_FN, "missing") is None


class TestCheckHasPrint:
    def test_has_print(self) -> None:
        assert check_has_print(COMPLEX_FN, "bar") is True

    def test_no_print(self) -> None:
        assert check_has_print(SAMPLE_FN, "foo") is False


class TestCheckHasRaise:
    def test_has_raise(self) -> None:
        code = "def foo():\n    raise ValueError('x')\n"
        assert check_has_raise(code, "foo") is True

    def test_no_raise(self) -> None:
        assert check_has_raise(SAMPLE_FN, "foo") is False


class TestCheckHasAssert:
    def test_has_assert(self) -> None:
        code = "def foo():\n    assert True\n"
        assert check_has_assert(code, "foo") is True

    def test_no_assert(self) -> None:
        assert check_has_assert(SAMPLE_FN, "foo") is False


class TestGetReturnTypeAnnotation:
    def test_has_annotation(self) -> None:
        assert get_return_type_annotation(SAMPLE_FN, "foo") == "int"

    def test_no_annotation(self) -> None:
        code = "def foo():\n    pass\n"
        assert get_return_type_annotation(code, "foo") is None

    def test_missing_function(self) -> None:
        assert get_return_type_annotation(SAMPLE_FN, "missing") is None


class TestGetParameterNames:
    def test_basic_params(self) -> None:
        assert get_parameter_names(SAMPLE_FN, "foo") == ["a", "b"]

    def test_varargs(self) -> None:
        code = "def foo(*args, **kwargs):\n    pass\n"
        assert get_parameter_names(code, "foo") == ["*args", "**kwargs"]

    def test_missing_function(self) -> None:
        assert get_parameter_names(SAMPLE_FN, "missing") is None


class TestExtractInlineComments:
    def test_extracts_comments(self) -> None:
        comments = extract_inline_comments(SAMPLE_FN, "foo")
        assert comments is not None
        assert "a comment" in comments

    def test_no_comments(self) -> None:
        code = "def foo():\n    return 1\n"
        comments = extract_inline_comments(code, "foo")
        assert comments == []

    def test_missing_function(self) -> None:
        assert extract_inline_comments(SAMPLE_FN, "missing") is None


class TestExtractStringLiterals:
    def test_extracts_strings(self) -> None:
        code = 'def foo():\n    x = "hello"\n    return x\n'
        literals = extract_string_literals(code, "foo")
        assert literals == ["hello"]

    def test_excludes_docstring(self) -> None:
        literals = extract_string_literals(SAMPLE_FN, "foo")
        assert literals is not None
        assert "Add two numbers." not in literals

    def test_missing_function(self) -> None:
        assert extract_string_literals(SAMPLE_FN, "missing") is None


class TestCountControlStructures:
    def test_counts(self) -> None:
        counts = count_control_structures(COMPLEX_FN, "bar")
        assert counts is not None
        assert counts.for_loops == 1
        assert counts.if_statements == 1
        assert counts.elif_branches == 1

    def test_empty_function(self) -> None:
        code = "def foo():\n    pass\n"
        counts = count_control_structures(code, "foo")
        assert counts is not None
        assert counts.for_loops == 0

    def test_missing_function(self) -> None:
        assert count_control_structures(SAMPLE_FN, "missing") is None


class TestAnalyzeCodeStyle:
    def test_metrics(self) -> None:
        metrics = analyze_code_style(COMPLEX_FN, "bar")
        assert metrics is not None
        assert metrics.total_lines > 0
        assert metrics.num_variables > 0
        assert "result" in metrics.variable_names
        assert "item" in metrics.variable_names

    def test_nesting_depth(self) -> None:
        metrics = analyze_code_style(COMPLEX_FN, "bar")
        assert metrics is not None
        assert metrics.max_nesting_depth >= 2

    def test_magic_numbers(self) -> None:
        code = "def foo():\n    x = 42\n    return x\n"
        metrics = analyze_code_style(code, "foo")
        assert metrics is not None
        assert metrics.num_magic_numbers >= 1

    def test_missing_function(self) -> None:
        assert analyze_code_style(SAMPLE_FN, "missing") is None


class TestAnalyzeFunction:
    def test_aggregated(self) -> None:
        result = analyze_function(SAMPLE_FN, "foo")
        assert isinstance(result, FunctionAnalysis)
        assert result.function_name == "foo"
        assert result.has_return is True
        assert result.has_print is False
        assert result.return_type_annotation == "int"
        assert result.parameter_names == ["a", "b"]
        assert result.has_docstring is True
        assert result.docstring_text == "Add two numbers."
        assert len(result.inline_comments) >= 1

    def test_raises_on_missing(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            analyze_function(SAMPLE_FN, "missing")
