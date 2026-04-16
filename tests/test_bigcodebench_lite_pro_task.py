import textwrap

import pytest

from nl_code.datasets.bigcodebench_lite_pro_task import RawBigCodeBenchLiteProTask

from conftest import make_bigcodebench_lite_pro_row

pytestmark = pytest.mark.docker


class TestRawBigCodeBenchLiteProTask:
    def test_non_code_fields(self) -> None:
        assert RawBigCodeBenchLiteProTask.non_code_fields == (
            "new_description",
            "new_problem_comment",
            "new_docstrings_and_comments",
            "original_docstrings_and_comments",
            "task_id",
            "validated",
            "version",
        )

    def test_construction(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert task.task_id == "BigCodeBenchLitePro/23"
        assert task.source__raw_problem == row["raw_problem"]
        assert task.source__raw_solution == row["raw_solution"]
        assert task.source__new_problem == row["new_problem"]
        assert task.source__new_solution == row["new_solution"]
        assert task.source__test_code == row["test_code"]
        assert task.version == "v2"
        assert task.validated is False
        assert "def multiply" in task.original_function
        assert "return a * b" in task.original_function
        assert "def multiply" in task.original_function_with_docstrings_and_comments
        assert "def multiply_pairs" in task.new_function_with_docstrings_and_comments
        assert task.original_official_prompt == "def multiply(a: int, b: int) -> int:\n"
        assert (
            task.new_official_prompt
            == "def multiply_pairs(pairs: list[tuple[int, int]]) -> list[int]:\n"
        )

    def test_additional_derived_prompt_fields(self) -> None:
        row = make_bigcodebench_lite_pro_row(
            raw_problem=textwrap.dedent("""\
                import math
                from collections import deque

                def multiply(a: int, b: int) -> int:
                    \"\"\"Multiply two integers.\"\"\"
            """),
            new_problem=textwrap.dedent("""\
                # Given a list of pairs, multiply each pair and return the list of products.
                def multiply_pairs(pairs: list[tuple[int, int]]) -> list[int]:
            """),
            new_solution=(
                '    """Return products for each input pair."""\n'
                "    result = []\n"
                "    for a, b in pairs:\n"
                "        result.append(multiply(a, b))\n"
                "    return result\n"
            ),
        )
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)

        assert task.original_function == textwrap.dedent("""\
            import math
            from collections import deque

            def multiply(a: int, b: int) -> int:
                return a * b
        """)
        assert task.original_function_with_docstrings_and_comments == textwrap.dedent("""\
            import math
            from collections import deque

            def multiply(a: int, b: int) -> int:
                \"\"\"Multiply two integers.\"\"\"
                return a * b
        """)
        assert task.new_function == textwrap.dedent("""\
            def multiply_pairs(pairs: list[tuple[int, int]]) -> list[int]:
                result = []
                for a, b in pairs:
                    result.append(multiply(a, b))
                return result
        """)
        assert task.new_function_with_docstrings_and_comments == textwrap.dedent("""\
            # Given a list of pairs, multiply each pair and return the list of products.
            def multiply_pairs(pairs: list[tuple[int, int]]) -> list[int]:
                \"\"\"Return products for each input pair.\"\"\"
                result = []
                for a, b in pairs:
                    result.append(multiply(a, b))
                return result
        """)
        assert task.raw_problem_imports == textwrap.dedent("""\
            import math
            from collections import deque
        """)
        assert (
            task.new_problem_without_docstrings_and_comments
            == "def multiply_pairs(pairs: list[tuple[int, int]]) -> list[int]:\n"
        )
        assert task.original_docstrings_and_comments == "Multiply two integers."
        assert task.new_docstrings_and_comments == (
            "Given a list of pairs, multiply each pair and return the list of products.\n\n"
            "Return products for each input pair."
        )
        assert (
            task.new_problem_comment
            == "Given a list of pairs, multiply each pair and return the list of products."
        )
        assert task.original_function_stub == textwrap.dedent("""\
            import math
            from collections import deque

            def multiply(a: int, b: int) -> int:
        """)
        assert task.original_function_stub_with_comments == row["raw_problem"]
        assert task.new_function_stub == textwrap.dedent("""\
            import math
            from collections import deque
            def multiply_pairs(pairs: list[tuple[int, int]]) -> list[int]:
        """)
        assert task.new_function_stub_with_comments == row["new_problem"]
        assert task.new_two_part_function_stub == textwrap.dedent("""\
            import math
            from collections import deque

            def multiply(a: int, b: int) -> int:


            def multiply_pairs(pairs: list[tuple[int, int]]) -> list[int]:
        """)
        assert task.new_two_part_function_stub_with_comments == textwrap.dedent("""\
            import math
            from collections import deque

            def multiply(a: int, b: int) -> int:
                \"\"\"Multiply two integers.\"\"\"


            # Given a list of pairs, multiply each pair and return the list of products.
            def multiply_pairs(pairs: list[tuple[int, int]]) -> list[int]:
        """)
        assert task.original_official_prompt == task.original_function_stub
        assert (
            task.new_official_prompt == task.new_problem_without_docstrings_and_comments
        )

    def test_gt_solution_contains_both_functions(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert "def multiply" in task.gt_solution
        assert "def multiply_pairs" in task.gt_solution

    def test_gt_solution_base_before_new(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        mul_pos = task.gt_solution_with_comments.index("def multiply(")
        mul_pairs_pos = task.gt_solution_with_comments.index("def multiply_pairs(")
        assert mul_pos < mul_pairs_pos
        assert (
            "\n\n\n# Given a list of pairs, multiply each pair and return the list of products.\n"
            in task.gt_solution_with_comments
        )
        assert "\n\n\ndef multiply_pairs(" in task.gt_solution

    def test_gt_solution_with_comments(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert '"""' in task.gt_solution_with_comments
        assert "def multiply" in task.gt_solution_with_comments

    def test_gt_solution(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert '"""' not in task.gt_solution
        assert "#" not in task.gt_solution
        assert "def multiply" in task.gt_solution
        assert '"""' not in task.original_function
        assert "#" not in task.original_function
        assert '"""' not in task.new_function
        assert "#" not in task.new_function
        assert '"""' not in task.original_official_prompt
        assert '"""' not in task.new_official_prompt

    def test_new_entry_point(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert task.new_entry_point == "multiply_pairs"

    def test_new_description(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert "list of pairs" in task.new_description

    def test_new_docstring_must_be_present_in_new_solution(self) -> None:
        row = make_bigcodebench_lite_pro_row(
            new_problem=textwrap.dedent("""\
                def multiply_pairs(pairs: list[tuple[int, int]]) -> list[int]:
                    \"\"\"Return products for each input pair.\"\"\"
            """),
            new_solution=(
                "    result = []\n"
                "    for a, b in pairs:\n"
                "        result.append(multiply(a, b))\n"
                "    return result\n"
            ),
        )
        row["task_id"] = "BigCodeBenchLitePro/23"

        with pytest.raises(
            ValueError, match="new function docstring must be present in new_solution"
        ):
            RawBigCodeBenchLiteProTask.model_validate(row)

    def test_run_test_passes_gt(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert task.run_test_on_gt_solution() is True

    def test_run_test_fails_bad_code(self) -> None:
        row = make_bigcodebench_lite_pro_row()
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        bad_code = "def multiply_pairs(pairs):\n    return []\n"
        assert task.run_test(bad_code) is False

    def test_construction_allows_failing_solution(self) -> None:
        row = make_bigcodebench_lite_pro_row(
            new_solution="    return []\n",
        )
        row["task_id"] = "BigCodeBenchLitePro/23"
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert task.validated is False
        assert task.run_test_on_gt_solution() is False

    def test_validated_flag_skips_validation(self) -> None:
        row = make_bigcodebench_lite_pro_row(
            new_solution="    return []\n",
        )
        row["task_id"] = "BigCodeBenchLitePro/23"
        row["validated"] = True
        task = RawBigCodeBenchLiteProTask.model_validate(row)
        assert task.validated is True
