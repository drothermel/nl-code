import textwrap

import pytest

from nl_code.datasets.mbpp_pro_task import RawMbppProTask

from conftest import make_mbpp_pro_row

pytestmark = pytest.mark.docker


def _expected_original_official_prompt(raw_problem: str) -> str:
    return (
        "You are an exceptionally intelligent coding assistant that "
        "consistently delivers accurate and reliable responses to user "
        "instructions. Write a solution of python file to the following problem\n"
        "@@ Instruction \n"
        f"{raw_problem.rstrip()}\n"
        "@@ Response\n"
    )


def _expected_new_official_prompt(raw_problem: str, new_problem: str) -> str:
    return (
        "You are an exceptionally intelligent coding assistant that "
        "consistently delivers accurate and reliable responses to user "
        "instructions. Write a solution of python file to the following "
        "problems, the solution of the second problem requires single or "
        "multiple calls to the first\n"
        "@@ Instruction \n"
        "```python\n"
        f"{raw_problem.rstrip()}\n"
        f"{new_problem.rstrip()}\n"
        "```\n"
        "@@ Response\n"
    )


def _task_input(row: dict[str, object]) -> dict[str, object]:
    return {
        "task_id": row["task_id"],
        "validated": row.get("validated", False),
        "source": {
            "raw_problem": row["raw_problem"],
            "raw_solution": row["raw_solution"],
            "new_problem": row["new_problem"],
            "new_solution": row["new_solution"],
            "test_code": row["test_code"],
        },
    }


class TestRawMbppProTask:
    def test_construction(self) -> None:
        row = make_mbpp_pro_row()
        row["task_id"] = "MbppPro/0"
        task = RawMbppProTask.model_validate(_task_input(row))
        assert task.task_id == "MbppPro/0"
        assert task.source.raw_problem == row["raw_problem"]
        assert task.source.raw_solution == row["raw_solution"]
        assert task.source.new_problem == row["new_problem"]
        assert task.source.new_solution == row["new_solution"]
        assert task.source.test_code == row["test_code"]
        assert task.version == "v3"
        assert task.validated is False
        assert "def add" in task.original_solution.code
        assert "return a + b" in task.original_solution.code
        assert "def add" in task.original_solution.code_with_comments
        assert "def add_pairs" in task.new_solution.code_with_comments
        assert task.prompts.original_official == _expected_original_official_prompt(
            row["raw_problem"]
        )
        assert task.prompts.new_official == _expected_new_official_prompt(
            row["raw_problem"], row["new_problem"]
        )

    def test_additional_derived_prompt_fields(self) -> None:
        row = make_mbpp_pro_row(
            raw_problem=textwrap.dedent("""\
                import math
                from collections import deque

                def add(a: int, b: int) -> int:
                    \"\"\"Add two integers.\"\"\"
            """),
            new_problem=textwrap.dedent("""\
                # Given a list of pairs, add each pair and return the list of sums.
                def add_pairs(pairs: list[tuple[int, int]]) -> list[int]:
            """),
            new_solution=(
                '    """Return sums for each input pair."""\n'
                "    result = []\n"
                "    for a, b in pairs:\n"
                "        result.append(add(a, b))\n"
                "    return result\n"
            ),
        )
        row["task_id"] = "MbppPro/0"
        task = RawMbppProTask.model_validate(_task_input(row))

        assert task.original_solution.code == textwrap.dedent("""\
            import math
            from collections import deque

            def add(a: int, b: int) -> int:
                return a + b
        """)
        assert task.original_solution.code_with_comments == textwrap.dedent("""\
            import math
            from collections import deque

            def add(a: int, b: int) -> int:
                \"\"\"Add two integers.\"\"\"
                return a + b
        """)
        assert task.new_solution.code == textwrap.dedent("""\
            def add_pairs(pairs: list[tuple[int, int]]) -> list[int]:
                result = []
                for a, b in pairs:
                    result.append(add(a, b))
                return result
        """)
        assert task.new_solution.code_with_comments == textwrap.dedent("""\
            # Given a list of pairs, add each pair and return the list of sums.
            def add_pairs(pairs: list[tuple[int, int]]) -> list[int]:
                \"\"\"Return sums for each input pair.\"\"\"
                result = []
                for a, b in pairs:
                    result.append(add(a, b))
                return result
        """)
        assert task.original_solution.imports == textwrap.dedent("""\
            import math
            from collections import deque
        """)
        assert (
            task.new_solution.problem_without_docstrings_and_comments
            == "def add_pairs(pairs: list[tuple[int, int]]) -> list[int]:\n"
        )
        assert task.original_solution.docstrings_and_comments == "Add two integers."
        assert task.new_solution.docstrings_and_comments == (
            "Given a list of pairs, add each pair and return the list of sums.\n\n"
            "Return sums for each input pair."
        )
        assert (
            task.new_solution.problem_comment
            == "Given a list of pairs, add each pair and return the list of sums."
        )
        assert task.original_solution.stub == textwrap.dedent("""\
            import math
            from collections import deque

            def add(a: int, b: int) -> int:
        """)
        assert task.original_solution.stub_with_comments == row["raw_problem"]
        assert task.new_solution.stub == textwrap.dedent("""\
            import math
            from collections import deque
            def add_pairs(pairs: list[tuple[int, int]]) -> list[int]:
        """)
        assert task.new_solution.stub_with_comments == row["new_problem"]
        assert task.new_solution.two_part_stub == textwrap.dedent("""\
            import math
            from collections import deque

            def add(a: int, b: int) -> int:


            def add_pairs(pairs: list[tuple[int, int]]) -> list[int]:
        """)
        assert task.new_solution.two_part_stub_with_comments == textwrap.dedent("""\
            import math
            from collections import deque

            def add(a: int, b: int) -> int:
                \"\"\"Add two integers.\"\"\"


            # Given a list of pairs, add each pair and return the list of sums.
            def add_pairs(pairs: list[tuple[int, int]]) -> list[int]:
        """)
        assert task.prompts.original_official == _expected_original_official_prompt(
            row["raw_problem"]
        )
        assert task.prompts.new_official == _expected_new_official_prompt(
            row["raw_problem"], row["new_problem"]
        )

    def test_gt_solution_contains_both_functions(self) -> None:
        row = make_mbpp_pro_row()
        row["task_id"] = "MbppPro/0"
        task = RawMbppProTask.model_validate(_task_input(row))
        assert "def add" in task.gt_solution.code
        assert "def add_pairs" in task.gt_solution.code

    def test_gt_solution_base_before_new(self) -> None:
        row = make_mbpp_pro_row()
        row["task_id"] = "MbppPro/0"
        task = RawMbppProTask.model_validate(_task_input(row))
        add_pos = task.gt_solution.code_with_comments.index("def add(")
        add_pairs_pos = task.gt_solution.code_with_comments.index("def add_pairs(")
        assert add_pos < add_pairs_pos
        assert (
            "\n\n\n# Given a list of pairs, add each pair and return the list of sums.\n"
            in task.gt_solution.code_with_comments
        )
        assert "\n\n\ndef add_pairs(" in task.gt_solution.code

    def test_gt_solution_with_comments(self) -> None:
        row = make_mbpp_pro_row()
        row["task_id"] = "MbppPro/0"
        task = RawMbppProTask.model_validate(_task_input(row))
        assert '"""' in task.gt_solution.code_with_comments
        assert "def add" in task.gt_solution.code_with_comments

    def test_gt_solution(self) -> None:
        row = make_mbpp_pro_row()
        row["task_id"] = "MbppPro/0"
        task = RawMbppProTask.model_validate(_task_input(row))
        assert '"""' not in task.gt_solution.code
        assert "#" not in task.gt_solution.code
        assert "def add" in task.gt_solution.code
        assert '"""' not in task.original_solution.code
        assert "#" not in task.original_solution.code
        assert '"""' not in task.new_solution.code
        assert "#" not in task.new_solution.code
        assert task.prompts.original_official.startswith(
            "You are an exceptionally intelligent coding assistant"
        )
        assert '"""' in task.prompts.original_official
        assert task.prompts.new_official.startswith(
            "You are an exceptionally intelligent coding assistant"
        )
        assert "```python\n" in task.prompts.new_official
        assert "# Given a list of pairs" in task.prompts.new_official

    def test_new_entry_point(self) -> None:
        row = make_mbpp_pro_row()
        row["task_id"] = "MbppPro/0"
        task = RawMbppProTask.model_validate(_task_input(row))
        assert task.target.name == "add_pairs"

    def test_new_description(self) -> None:
        row = make_mbpp_pro_row()
        row["task_id"] = "MbppPro/0"
        task = RawMbppProTask.model_validate(_task_input(row))
        assert "list of pairs" in task.description

    def test_new_docstring_must_be_present_in_new_solution(self) -> None:
        row = make_mbpp_pro_row(
            new_problem=textwrap.dedent("""\
                def add_pairs(pairs: list[tuple[int, int]]) -> list[int]:
                    \"\"\"Return sums for each input pair.\"\"\"
            """),
            new_solution=(
                "    result = []\n"
                "    for a, b in pairs:\n"
                "        result.append(add(a, b))\n"
                "    return result\n"
            ),
        )
        row["task_id"] = "MbppPro/0"

        with pytest.raises(
            ValueError, match="new function docstring must be present in new_solution"
        ):
            task = RawMbppProTask.model_validate(_task_input(row))
            _ = task.new_solution.docstrings_and_comments

    def test_run_test_passes_gt(self) -> None:
        row = make_mbpp_pro_row()
        row["task_id"] = "MbppPro/0"
        task = RawMbppProTask.model_validate(_task_input(row))
        assert task.run_test_on_gt_solution() is True

    def test_run_test_fails_bad_code(self) -> None:
        row = make_mbpp_pro_row()
        row["task_id"] = "MbppPro/0"
        task = RawMbppProTask.model_validate(_task_input(row))
        bad_code = "def add_pairs(pairs):\n    return []\n"
        assert task.run_test(bad_code) is False

    def test_construction_allows_failing_solution(self) -> None:
        row = make_mbpp_pro_row(
            new_solution="    return []\n",
        )
        row["task_id"] = "MbppPro/0"
        task = RawMbppProTask.model_validate(_task_input(row))
        assert task.validated is False
        assert task.run_test_on_gt_solution() is False

    def test_validated_flag_skips_validation(self) -> None:
        row = make_mbpp_pro_row(
            new_solution="    return []\n",
        )
        row["task_id"] = "MbppPro/0"
        row["validated"] = True
        task = RawMbppProTask.model_validate(_task_input(row))
        assert task.validated is True
