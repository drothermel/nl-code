import textwrap

import pytest
from pydantic import ValidationError

from nl_code.datasets.classeval_task import (
    ClassEvalTestDetail,
    MethodDependencies,
    MethodInfo,
    RawClassEvalTask,
)

from conftest import make_classeval_row


class TestMethodDependencies:
    def test_construction_with_alias(self) -> None:
        deps = MethodDependencies.model_validate(
            {
                "Standalone": True,
                "lib_dependencies": ["math"],
                "field_dependencies": ["self.x"],
                "method_dependencies": ["helper"],
            }
        )
        assert deps.standalone is True
        assert deps.lib_dependencies == ["math"]
        assert deps.field_dependencies == ["self.x"]
        assert deps.method_dependencies == ["helper"]

    def test_construction_with_python_name(self) -> None:
        deps = MethodDependencies(  # ty: ignore[missing-argument]
            standalone=False,  # ty: ignore[unknown-argument]
            lib_dependencies=[],
            field_dependencies=[],
            method_dependencies=[],
        )
        assert deps.standalone is False


class TestMethodInfo:
    def test_construction(self) -> None:
        info = MethodInfo(
            method_name="add",
            method_description="Add two numbers.",
            solution_code="def add(self, a, b): return a + b",
            test_class="TestAdd",
            test_code="class TestAdd: ...",
            dependencies=MethodDependencies(  # ty: ignore[missing-argument]
                standalone=True,  # ty: ignore[unknown-argument]
                lib_dependencies=[],
                field_dependencies=[],
                method_dependencies=[],
            ),
        )
        assert info.method_name == "add"
        assert info.dependencies.standalone is True


class TestClassEvalTestDetail:
    def test_passed_true(self) -> None:
        detail = ClassEvalTestDetail(
            test_class_name="TestAdd",
            tests_run=2,
            tests_passed=2,
            tests_failed=0,
            tests_errored=0,
            failures=[],
            errors=[],
            passed=True,
        )
        assert detail.passed is True

    def test_passed_false_on_failure(self) -> None:
        detail = ClassEvalTestDetail(
            test_class_name="TestAdd",
            tests_run=2,
            tests_passed=1,
            tests_failed=1,
            tests_errored=0,
            failures=["test_foo: AssertionError"],
            errors=[],
            passed=False,
        )
        assert detail.passed is False


class TestRawClassEvalTask:
    def test_construction(self) -> None:
        row = make_classeval_row()
        task = RawClassEvalTask.model_validate(row)
        assert task.task_id == "ClassEval_0"
        assert task.class_name == "Calculator"
        assert task.validated is True

    def test_computed_gt_code(self) -> None:
        row = make_classeval_row()
        task = RawClassEvalTask.model_validate(row)
        assert "class Calculator" in task.gt_code
        assert task.gt_code == task.solution_code

    def test_gt_code_includes_imports(self) -> None:
        row = make_classeval_row()
        row["import_statement"] = ["import math"]
        row["solution_code"] = textwrap.dedent("""\
            class Calculator:
                def __init__(self):
                    self.result = 0

                def add(self, a, b):
                    return a + b

                def subtract(self, a, b):
                    return a - b

                def sqrt(self, x):
                    return math.sqrt(x)
        """)
        row["test"] = textwrap.dedent("""\
            import unittest

            class TestCalculatorAdd(unittest.TestCase):
                def test_add(self):
                    calc = Calculator()
                    self.assertEqual(calc.add(1, 2), 3)

            class TestCalculatorSubtract(unittest.TestCase):
                def test_subtract(self):
                    calc = Calculator()
                    self.assertEqual(calc.subtract(3, 1), 2)
        """)
        task = RawClassEvalTask.model_validate(row)
        assert task.gt_code.startswith("import math\n\n")
        assert "class Calculator" in task.gt_code

    def test_methods_info_parsed(self) -> None:
        row = make_classeval_row()
        task = RawClassEvalTask.model_validate(row)
        assert len(task.methods_info) == 2
        assert task.methods_info[0].method_name == "add"
        assert task.methods_info[1].method_name == "subtract"
        assert task.methods_info[0].dependencies.standalone is True

    def test_run_test_passes_gt(self) -> None:
        row = make_classeval_row()
        task = RawClassEvalTask.model_validate(row)
        result = task.run_test_on_gt_solution()
        assert result.all_passed is True
        assert result.total_tests_run == 3
        assert result.total_tests_passed == 3
        assert result.total_tests_failed == 0

    def test_run_test_fails_bad_code(self) -> None:
        row = make_classeval_row()
        task = RawClassEvalTask.model_validate(row)
        bad_code = textwrap.dedent("""\
            class Calculator:
                def __init__(self):
                    self.result = 0
                def add(self, a, b):
                    return 0
                def subtract(self, a, b):
                    return 0
        """)
        result = task.run_test(bad_code)
        assert result.all_passed is False
        assert result.total_tests_failed > 0

    def test_run_test_exec_error(self) -> None:
        row = make_classeval_row()
        task = RawClassEvalTask.model_validate(row)
        result = task.run_test("this is not valid python!!!")
        assert result.all_passed is False
        assert result.error is not None
        assert "SyntaxError" in result.error

    def test_run_test_per_test_class_detail(self) -> None:
        row = make_classeval_row()
        task = RawClassEvalTask.model_validate(row)
        result = task.run_test_on_gt_solution()
        by_name = {r.test_class_name: r for r in result.per_test_class}
        assert len(by_name) == 2
        assert by_name["TestCalculatorAdd"].tests_run == 2
        assert by_name["TestCalculatorAdd"].passed is True
        assert by_name["TestCalculatorSubtract"].tests_run == 1
        assert by_name["TestCalculatorSubtract"].passed is True

    def test_run_test_missing_test_class(self) -> None:
        row = make_classeval_row()
        row["test_classes"] = ["TestCalculatorAdd", "TestNonExistent"]
        row["validated"] = True
        task = RawClassEvalTask.model_validate(row)
        result = task.run_test(task.solution_code)
        assert result.all_passed is False
        by_name = {r.test_class_name: r for r in result.per_test_class}
        assert by_name["TestNonExistent"].passed is False
        assert "not found" in by_name["TestNonExistent"].failures[0]

    def test_validation_rejects_failing_solution(self) -> None:
        row = make_classeval_row(
            solution_code=textwrap.dedent("""\
                class Calculator:
                    def __init__(self):
                        self.result = 0
                    def add(self, a, b):
                        return 0
                    def subtract(self, a, b):
                        return 0
            """),
        )
        with pytest.raises(ValidationError):
            RawClassEvalTask.model_validate(row)

    def test_validated_flag_skips_validation(self) -> None:
        row = make_classeval_row(
            solution_code=textwrap.dedent("""\
                class Calculator:
                    def __init__(self):
                        self.result = 0
                    def add(self, a, b):
                        return 0
                    def subtract(self, a, b):
                        return 0
            """),
        )
        row["validated"] = True
        task = RawClassEvalTask.model_validate(row)
        assert task.validated is True
