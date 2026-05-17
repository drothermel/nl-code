import textwrap

import pytest

from nl_code.datasets.humaneval_task import (
    HumanEvalTest,
    RawHumanEvalTask,
    build_assertion_test_code,
    build_function_source,
    parse_inputs_ref_func_test,
    parse_inputs_results_test,
)

from conftest import make_raw_humaneval_task_input


class TestHelperFunctions:
    def test_build_function_source(self) -> None:
        result = build_function_source("def foo():", "    pass")
        assert result == "def foo():\n    pass\n"

    def test_merge_rejects_non_strings(self) -> None:
        with pytest.raises(TypeError, match="prompt must be a string"):
            build_function_source(123, "x")

    def test_build_assertion_test_code(self) -> None:
        test_source = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
        assert (
            build_assertion_test_code(test_source, "add")
            == "def check(candidate):\n    assert candidate(1, 2) == 3\n\n\ncheck(add)\n"
        )

    def test_parse_inputs_results_test_preserves_test_suite(self) -> None:
        test_source = textwrap.dedent("""\
            import numpy as np

            def assertion(out, exp, atol):
                assert out == exp


            def check(candidate):
                inputs = [[1, 2], [3, 4]]
                results = [3, 7]
                for i, (inp, exp) in enumerate(zip(inputs, results)):
                    assertion(candidate(*inp), exp, 0)
        """)

        parsed = parse_inputs_results_test(test_source, "add")
        cases = list(parsed.iter_cases())

        assert parsed.shape == "inputs_results"
        assert parsed.inputs == [[1, 2], [3, 4]]
        assert parsed.results == [3, 7]
        assert len(cases) == 2
        assert cases[0].input_value == [1, 2]
        assert cases[0].expected_output == 3
        assert cases[0].has_expected_output is True
        assert parsed.source_for_index(0) == textwrap.dedent("""\
            import numpy as np

            def assertion(out, exp, atol):
                assert out == exp


            def check(candidate):
                inputs = [[1, 2]]
                results = [3]
                for i, (inp, exp) in enumerate(zip(inputs, results)):
                    assertion(candidate(*inp), exp, 0)
        """)
        assert parsed.assertion_test_code_for_index(0).endswith("\n\ncheck(add)\n")
        assert parsed.source_for_index(1) == textwrap.dedent("""\
            import numpy as np

            def assertion(out, exp, atol):
                assert out == exp


            def check(candidate):
                inputs = [[3, 4]]
                results = [7]
                for i, (inp, exp) in enumerate(zip(inputs, results)):
                    assertion(candidate(*inp), exp, 0)
        """)

    def test_parse_inputs_ref_func_test_preserves_test_suite(self) -> None:
        test_source = textwrap.dedent("""\
            import numpy as np

            def assertion(out, exp, atol):
                assert out == exp


            def ref_func(x):
                return x + 1


            def check(candidate):
                inputs = [[1], [2]]
                for i, inp in enumerate(inputs):
                    assertion(candidate(*inp), ref_func(*inp), 0)
        """)

        parsed = parse_inputs_ref_func_test(test_source, "increment")
        cases = list(parsed.iter_cases())

        assert parsed.shape == "inputs_ref_func"
        assert parsed.inputs == [[1], [2]]
        assert parsed.results is None
        assert len(cases) == 2
        assert cases[0].input_value == [1]
        assert cases[0].expected_output is None
        assert cases[0].has_expected_output is False
        assert parsed.source_for_index(0) == textwrap.dedent("""\
            import numpy as np

            def assertion(out, exp, atol):
                assert out == exp


            def ref_func(x):
                return x + 1


            def check(candidate):
                inputs = [[1]]
                for i, inp in enumerate(inputs):
                    assertion(candidate(*inp), ref_func(*inp), 0)
        """)
        assert "def ref_func" in parsed.source_for_index(1)
        assert parsed.source_for_index(1).endswith(
            "        assertion(candidate(*inp), ref_func(*inp), 0)\n"
        )

    def test_humaneval_test_validates_shape(self) -> None:
        with pytest.raises(
            ValueError,
            match="inputs_results test shape requires results",
        ):
            HumanEvalTest(
                source="def check(candidate):\n    pass\n",
                entry_point="foo",
                shape="inputs_results",
                inputs=[[1]],
            )


@pytest.mark.docker
class TestRawHumanEvalTask:
    def test_non_code_fields(self) -> None:
        assert RawHumanEvalTask.non_code_fields == (
            "entry_point",
            "task_id",
            "validated",
            "version",
        )

    def test_construction(self, valid_raw_task: RawHumanEvalTask) -> None:
        assert valid_raw_task.task_id == "HumanEval/0"
        assert valid_raw_task.entry_point == "add"
        assert valid_raw_task.version == "v2"
        assert valid_raw_task.source.prompt
        assert valid_raw_task.source.canonical_solution == "    return a + b\n"
        assert valid_raw_task.source.test
        assert valid_raw_task.test_suite.source == valid_raw_task.source.test
        assert valid_raw_task.validated is False

    def test_additional_derived_solution_fields(self) -> None:
        row = make_raw_humaneval_task_input(
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

        assert task.gt_solution_with_comments == textwrap.dedent('''\
            # Add two integers.
            def add(a: int, b: int) -> int:
                """Return the sum."""
                return a + b
        ''')
        assert task.gt_solution == textwrap.dedent("""\
            def add(a: int, b: int) -> int:
                return a + b
        """)
        assert task.test_suite.shape == "inputs_results"
        assert task.test_suite.inputs == [[1, 2], [0, 0]]
        assert task.test_suite.results == [3, 0]
        assert task.test_suite.assertion_test_code() == textwrap.dedent("""\
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

    def test_computed_test_suite(self, valid_raw_task: RawHumanEvalTask) -> None:
        assert valid_raw_task.test_suite.inputs == [[1, 2], [0, 0], [-1, 1]]
        assert valid_raw_task.test_suite.results == [3, 0, 0]

    def test_model_dump_serializes_source_only(
        self, valid_raw_task: RawHumanEvalTask
    ) -> None:
        dumped = valid_raw_task.model_dump(mode="json")

        assert dumped["source"]["prompt"] == valid_raw_task.source.prompt
        assert dumped["source"]["canonical_solution"] == (
            valid_raw_task.source.canonical_solution
        )
        assert dumped["source"]["test"] == valid_raw_task.source.test
        assert "test_suite" not in dumped
        assert "official_prompt" not in dumped
        assert "new_official_prompt" not in dumped
        assert "source__prompt" not in dumped
        assert "source__canonical_solution" not in dumped
        assert "source__test" not in dumped
        assert "test_inputs" not in dumped
        assert "test_results" not in dumped
        assert "assertion_test_code" not in dumped

    def test_test_suite_iter_cases_uses_matching_shape(
        self, valid_raw_task: RawHumanEvalTask
    ) -> None:
        cases = list(valid_raw_task.test_suite.iter_cases())

        assert [case.input_value for case in cases] == [[1, 2], [0, 0], [-1, 1]]
        assert [case.expected_output for case in cases] == [3, 0, 0]
        assert all(case.has_expected_output for case in cases)
        assert "inputs = [[1, 2]]" in valid_raw_task.test_suite.source_for_index(0)
        assert "results = [3]" in valid_raw_task.test_suite.source_for_index(0)

    def test_mismatched_test_inputs_and_results_raise(self) -> None:
        row = make_raw_humaneval_task_input(
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
        assert valid_raw_task.test_suite.run_test(valid_raw_task.gt_solution) is True

    def test_run_test_fails_bad_code(self, valid_raw_task: RawHumanEvalTask) -> None:
        bad_code = "def add(a, b):\n    return a - b\n"
        assert valid_raw_task.test_suite.run_test(bad_code) is False

    def test_construction_allows_failing_solution(self) -> None:
        row = make_raw_humaneval_task_input(canonical_solution="    return a - b\n")
        task = RawHumanEvalTask.model_validate(row)
        assert task.validated is False
        assert task.test_suite.run_test(task.gt_solution) is False

    def test_validated_flag_skips_validation(self) -> None:
        row = make_raw_humaneval_task_input(canonical_solution="    return a - b\n")
        row["validated"] = True
        task = RawHumanEvalTask.model_validate(row)
        assert task.validated is True
