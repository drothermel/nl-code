import textwrap

import pytest

from nl_code.datasets.humaneval_task import (
    RawHumanEvalTask,
    build_assertion_test_code,
    build_function_source,
    build_function_without_comments,
    build_official_prompt,
    build_function_stub,
    extract_docstrings,
    extract_prompt_comment,
)

from conftest import make_humaneval_row


class TestHelperFunctions:
    def test_build_function_source(self) -> None:
        result = build_function_source("def foo():", "    pass")
        assert result == "def foo():\n    pass\n"

    def test_build_function_without_comments(self) -> None:
        result = build_function_without_comments(
            '# comment\ndef foo():\n    """Hello world."""\n',
            "    pass\n",
        )
        assert result == "def foo():\n    pass\n"

    def test_merge_rejects_non_strings(self) -> None:
        with pytest.raises(TypeError, match="prompt must be a string"):
            build_function_source(123, "x")

    def test_extract_docstrings(self) -> None:
        prompt = 'def foo():\n    """Hello world."""\n'
        assert extract_docstrings(prompt, "foo") == "Hello world."

    def test_extract_prompt_comment(self) -> None:
        prompt = "# a comment\ndef foo():\n    pass\n"
        assert extract_prompt_comment(prompt) == "a comment"

    def test_extract_prompt_comment_none(self) -> None:
        assert extract_prompt_comment("def foo():\n    pass\n") == ""

    def test_build_function_stub(self) -> None:
        prompt = 'def foo():\n    """Hello world."""\n'
        assert build_function_stub(prompt) == "def foo():\n"

    def test_build_assertion_test_code(self) -> None:
        test_source = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
        assert (
            build_assertion_test_code(test_source, "add")
            == "def check(candidate):\n    assert candidate(1, 2) == 3\n\n\ncheck(add)\n"
        )

    def test_build_official_prompt(self) -> None:
        prompt = 'def foo():\n    """Hello world."""\n'
        assert build_official_prompt(prompt) == (
            "Read the following function signature and docstring, and fully "
            "implement the function described. Your response should only contain "
            "the code for this function.\n\n"
            "```python\n"
            'def foo():\n    """Hello world."""\n'
            "```\n"
        )


@pytest.mark.docker
class TestRawHumanEvalTask:
    def test_non_code_fields(self) -> None:
        assert RawHumanEvalTask.non_code_fields == (
            "docstrings",
            "entry_point",
            "prompt_comment",
            "task_id",
            "validated",
            "version",
        )

    def test_construction(self, valid_raw_task: RawHumanEvalTask) -> None:
        assert valid_raw_task.task_id == "HumanEval/0"
        assert valid_raw_task.entry_point == "add"
        assert valid_raw_task.version == "v2"
        assert valid_raw_task.source__prompt
        assert valid_raw_task.source__canonical_solution == "    return a + b\n"
        assert valid_raw_task.source__test
        assert valid_raw_task.validated is False
        assert valid_raw_task.official_prompt == build_official_prompt(
            valid_raw_task.source__prompt
        )
        assert valid_raw_task.new_official_prompt == valid_raw_task.official_prompt

    def test_additional_derived_prompt_fields(self) -> None:
        row = make_humaneval_row(
            prompt=textwrap.dedent('''\
                # Add two integers.
                def add(a: int, b: int) -> int:
                    """Return the sum."""
            '''),
            canonical_solution="    return a + b\n",
            test=textwrap.dedent("""\
                def check(candidate):
                    inputs = [[1, 2], [0, 0]]
                    results = [3, 0]
                    for inp, expected in zip(inputs, results):
                        assert candidate(*inp) == expected
            """),
        )
        task = RawHumanEvalTask.model_validate(row)

        assert task.official_prompt == build_official_prompt(row["prompt"])
        assert task.new_official_prompt == task.official_prompt
        assert task.docstrings == "Return the sum."
        assert task.prompt_comment == "Add two integers."
        assert task.function_stub == textwrap.dedent("""\
            # Add two integers.
            def add(a: int, b: int) -> int:
        """)
        assert task.function_stub_with_comments == row["prompt"]
        assert task.new_code_stub == task.function_stub
        assert task.new_code_stub_with_comments == task.function_stub_with_comments
        assert task.function_with_comments == textwrap.dedent('''\
            # Add two integers.
            def add(a: int, b: int) -> int:
                """Return the sum."""
                return a + b
        ''')
        assert task.function == textwrap.dedent("""\
            def add(a: int, b: int) -> int:
                return a + b
        """)
        assert task.gt_solution_with_comments == task.function_with_comments
        assert task.gt_solution == task.function
        assert task.assertion_test_code == textwrap.dedent("""\
            def check(candidate):
                inputs = [[1, 2], [0, 0]]
                results = [3, 0]
                for inp, expected in zip(inputs, results):
                    assert candidate(*inp) == expected


            check(add)
        """)

    def test_computed_gt_solution(self, valid_raw_task: RawHumanEvalTask) -> None:
        assert '"""' not in valid_raw_task.gt_solution
        assert "#" not in valid_raw_task.gt_solution
        assert "def add" in valid_raw_task.gt_solution
        assert "return a + b" in valid_raw_task.gt_solution

    def test_computed_gt_solution_with_comments(
        self,
        valid_raw_task: RawHumanEvalTask,
    ) -> None:
        assert '"""' in valid_raw_task.gt_solution_with_comments
        assert "add" in valid_raw_task.gt_solution_with_comments

    def test_computed_function_without_comments(
        self,
        valid_raw_task: RawHumanEvalTask,
    ) -> None:
        assert '"""' not in valid_raw_task.function
        assert "#" not in valid_raw_task.function
        assert "def add" in valid_raw_task.function

    def test_computed_docstrings(self, valid_raw_task: RawHumanEvalTask) -> None:
        assert valid_raw_task.docstrings == "Add two integers and return the result."

    def test_computed_test_inputs(self, valid_raw_task: RawHumanEvalTask) -> None:
        assert valid_raw_task.test_inputs == [[1, 2], [0, 0], [-1, 1]]
        assert valid_raw_task.test_results == [3, 0, 0]

    def test_mismatched_test_inputs_and_results_raise(self) -> None:
        row = make_humaneval_row(
            test=textwrap.dedent("""\
                def check(candidate):
                    inputs = [[1, 2], [0, 0]]
                    results = [3]
                    for inp, expected in zip(inputs, results):
                        assert candidate(*inp) == expected
            """)
        )
        with pytest.raises(
            ValueError,
            match="test inputs and results must have the same length",
        ):
            RawHumanEvalTask.model_validate(row)

    def test_run_test_passes_gt(self, valid_raw_task: RawHumanEvalTask) -> None:
        assert valid_raw_task.run_test_on_gt_solution() is True

    def test_run_test_fails_bad_code(self, valid_raw_task: RawHumanEvalTask) -> None:
        bad_code = "def add(a, b):\n    return a - b\n"
        assert valid_raw_task.run_test(bad_code) is False

    def test_construction_allows_failing_solution(self) -> None:
        row = make_humaneval_row(canonical_solution="    return a - b\n")
        task = RawHumanEvalTask.model_validate(row)
        assert task.validated is False
        assert task.run_test_on_gt_solution() is False

    def test_validated_flag_skips_validation(self) -> None:
        row = make_humaneval_row(canonical_solution="    return a - b\n")
        row["validated"] = True
        task = RawHumanEvalTask.model_validate(row)
        assert task.validated is True
