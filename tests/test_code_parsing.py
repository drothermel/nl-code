import ast
import textwrap

import pytest

from nl_code.code_parsing import (
    extract_top_level_import_source,
    find_docstring_expr,
    find_first_function_name,
    find_first_function_node,
    find_named_assignment_in_body,
    find_named_function,
    find_named_function_in_module,
    get_comments,
    get_docstring,
    get_docstrings_and_comments,
    get_first_function_docstring,
    line_col_to_index,
    literal_eval_assignment_value,
    merge_code_components,
    node_references_name,
    node_span,
    parse_source_with_stub_body,
    remove_docstrings_preserving_comments,
    remove_docstrings_and_comments,
    remove_full_line_comments,
    replace_source_spans,
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


class TestRemoveDocstringsPreservingComments:
    def test_removes_docstrings_without_unparsing(self) -> None:
        source = textwrap.dedent('''\
            # keep this comment
            def foo():
                """This is a docstring."""
                # keep inner comment
                return 1  # keep inline comment
        ''')
        result = remove_docstrings_preserving_comments(source)
        assert result == textwrap.dedent("""\
            # keep this comment
            def foo():
                # keep inner comment
                return 1  # keep inline comment
        """)

    def test_removes_nested_docstrings(self) -> None:
        source = textwrap.dedent('''\
            """Module docstring."""
            class Foo:
                """Class docstring."""
                def bar(self):
                    """Function docstring."""
                    return 1
        ''')
        result = remove_docstrings_preserving_comments(source)
        assert "docstring" not in result
        assert "class Foo" in result
        assert "return 1" in result

    def test_returns_normalized_source_for_incomplete_stub(self) -> None:
        assert remove_docstrings_preserving_comments("def foo():") == "def foo():\n"


class TestParseSourceWithStubBody:
    def test_parses_complete_source(self) -> None:
        tree = parse_source_with_stub_body("def foo():\n    return 1\n")
        assert isinstance(tree.body[0], ast.FunctionDef)

    def test_parses_stub_ending_in_colon(self) -> None:
        tree = parse_source_with_stub_body("def foo():")
        assert isinstance(tree.body[0], ast.FunctionDef)


class TestFindFirstFunctionName:
    def test_finds_first(self) -> None:
        source = "def foo():\n    pass\ndef bar():\n    pass\n"
        assert find_first_function_name(source) == "foo"

    def test_skips_imports(self) -> None:
        source = "import os\ndef baz():\n    pass\n"
        assert find_first_function_name(source) == "baz"

    def test_finds_first_async(self) -> None:
        source = "async def fetch():\n    pass\ndef process():\n    pass\n"
        assert find_first_function_name(source) == "fetch"

    def test_raises_on_no_function(self) -> None:
        with pytest.raises(ValueError, match="no function definition"):
            find_first_function_name("x = 1\n")


class TestFindFirstFunctionNode:
    def test_finds_first_node(self) -> None:
        node = find_first_function_node("def foo():\n    pass\n")
        assert node.name == "foo"

    def test_can_parse_stub_body(self) -> None:
        node = find_first_function_node("def foo():", allow_stub_body=True)
        assert node.name == "foo"


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

    def test_find_docstring_expr(self) -> None:
        node = find_named_function('def foo():\n    """Hello."""\n    pass\n', "foo")
        expr = find_docstring_expr(node)
        assert expr is not None
        assert ast.get_source_segment('def foo():\n    """Hello."""\n    pass\n', expr)

    def test_first_function_docstring_allows_stub_body(self) -> None:
        assert (
            get_first_function_docstring('def foo():\n    """Hello."""\n') == "Hello."
        )

    def test_docstrings_and_comments(self) -> None:
        source = '# comment\ndef foo():\n    """Hello."""\n'
        assert get_docstrings_and_comments(source) == "comment\n\nHello."


class TestSourceSpans:
    def test_line_col_to_index_handles_utf8_byte_offsets(self) -> None:
        source = "é = 1\nvalue = 2\n"
        assign = ast.parse(source).body[1]
        assert line_col_to_index(source, assign.lineno, assign.col_offset) == len(
            "é = 1\n"
        )

    def test_node_span_and_replace_source_spans(self) -> None:
        source = "x = [1, 2, 3]\n"
        assign = ast.parse(source).body[0]
        assert isinstance(assign, ast.Assign)
        span = node_span(source, assign.value)
        assert source[slice(*span)] == "[1, 2, 3]"
        assert replace_source_spans(source, [(span, "[2]")]) == "x = [2]\n"

    def test_node_references_name(self) -> None:
        node = find_named_function("def check():\n    ref_func(1)\n", "check")
        assert node_references_name(node, "ref_func")
        assert not node_references_name(node, "candidate")


class TestImportAndCommentExtraction:
    def test_extract_top_level_import_source(self) -> None:
        source = textwrap.dedent("""\
            import os
            from collections import deque

            def foo():
                import sys
                return deque()
        """)
        assert extract_top_level_import_source(source) == (
            "import os\nfrom collections import deque\n"
        )

    def test_extract_imports_allows_stub_body(self) -> None:
        assert (
            extract_top_level_import_source(
                "import os\n\ndef foo():",
                allow_stub_body=True,
            )
            == "import os\n"
        )

    def test_remove_full_line_comments_keeps_inline_comments(self) -> None:
        source = "# remove me\nx = 1  # keep me\n    # remove indented\n"
        assert remove_full_line_comments(source) == "x = 1  # keep me\n"

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
