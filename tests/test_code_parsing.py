import ast
import textwrap

import pytest

from nl_code.code_parsing import (
    find_first_function_name,
    find_named_assignment_in_body,
    find_named_function,
    find_named_function_in_module,
    get_comments,
    get_docstring,
    literal_eval_assignment_value,
    merge_code_components,
    remove_docstrings_and_comments,
)


class TestMergeCodeComponents:
    def test_joins_with_newline(self) -> None:
        result = merge_code_components("def foo():", "    pass")
        assert result == "def foo():\n    pass\n"

    def test_strips_trailing_whitespace(self) -> None:
        result = merge_code_components("a  ", "b  ")
        assert result == "a\nb\n"

    def test_single_component(self) -> None:
        result = merge_code_components("hello")
        assert result == "hello\n"

    def test_trailing_newline(self) -> None:
        result = merge_code_components("a", "b")
        assert result.endswith("\n")


class TestRemoveDocstringsAndComments:
    def test_removes_function_docstring(self) -> None:
        source = textwrap.dedent('''\
            def foo():
                """This is a docstring."""
                return 1
        ''')
        result = remove_docstrings_and_comments(source)
        assert '"""' not in result
        assert "return 1" in result

    def test_removes_comments(self) -> None:
        source = textwrap.dedent("""\
            def foo():
                # a comment
                return 1
        """)
        result = remove_docstrings_and_comments(source)
        assert "# a comment" not in result

    def test_removes_module_docstring(self) -> None:
        source = textwrap.dedent('''\
            """Module docstring."""
            x = 1
        ''')
        result = remove_docstrings_and_comments(source)
        assert "Module docstring" not in result
        assert "x = 1" in result

    def test_ends_with_newline(self) -> None:
        result = remove_docstrings_and_comments("x = 1\n")
        assert result.endswith("\n")


class TestFindFirstFunctionName:
    def test_finds_first(self) -> None:
        source = "def foo():\n    pass\ndef bar():\n    pass\n"
        assert find_first_function_name(source) == "foo"

    def test_skips_imports(self) -> None:
        source = "import os\ndef baz():\n    pass\n"
        assert find_first_function_name(source) == "baz"

    def test_raises_on_no_function(self) -> None:
        with pytest.raises(ValueError, match="no function definition"):
            find_first_function_name("x = 1\n")


class TestFindNamedFunction:
    def test_finds_function(self) -> None:
        source = "def foo():\n    pass\n"
        node = find_named_function(source, "foo")
        assert node.name == "foo"

    def test_finds_async_function(self) -> None:
        source = "async def bar():\n    pass\n"
        node = find_named_function(source, "bar")
        assert node.name == "bar"

    def test_raises_on_missing(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            find_named_function("x = 1\n", "foo")

    def test_in_module(self) -> None:
        tree = ast.parse("def baz():\n    pass\n")
        node = find_named_function_in_module(tree, "baz")
        assert node.name == "baz"


class TestFindNamedAssignment:
    def test_finds_assignment(self) -> None:
        source = "def check():\n    x = [1, 2]\n    pass\n"
        func = find_named_function(source, "check")
        assign = find_named_assignment_in_body(func.body, "x")
        assert assign is not None

    def test_returns_none_when_missing(self) -> None:
        source = "def check():\n    pass\n"
        func = find_named_function(source, "check")
        assert find_named_assignment_in_body(func.body, "x") is None

    def test_literal_eval(self) -> None:
        source = "def check():\n    inputs = [[1, 2], [3, 4]]\n"
        func = find_named_function(source, "check")
        assign = find_named_assignment_in_body(func.body, "inputs")
        assert assign is not None
        value = literal_eval_assignment_value(assign)
        assert value == [[1, 2], [3, 4]]


class TestGetDocstring:
    def test_function_docstring(self) -> None:
        source = 'def foo():\n    """Hello."""\n    pass\n'
        node = find_named_function(source, "foo")
        assert get_docstring(node) == "Hello."

    def test_no_docstring(self) -> None:
        node = find_named_function("def foo():\n    pass\n", "foo")
        assert get_docstring(node) == ""

    def test_module_docstring(self) -> None:
        tree = ast.parse('"""Module doc."""\nx = 1\n')
        assert get_docstring(tree) == "Module doc."


class TestGetComments:
    def test_extracts_comments(self) -> None:
        source = "# hello\nx = 1\n# world\n"
        result = get_comments(source)
        assert result == "# hello\n# world"

    def test_strip_hash(self) -> None:
        source = "# hello\n"
        result = get_comments(source, strip_hash=True)
        assert result == "hello"

    def test_no_comments_returns_none(self) -> None:
        assert get_comments("x = 1\n") is None
