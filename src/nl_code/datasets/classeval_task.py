import ast
import io
import os
import tempfile
import unittest

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MethodDependencies(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    standalone: bool = Field(alias="Standalone")
    lib_dependencies: list[str]
    field_dependencies: list[str]
    method_dependencies: list[str]


class MethodInfo(BaseModel):
    method_name: str
    method_description: str
    solution_code: str
    test_class: str
    test_code: str
    dependencies: MethodDependencies


class ClassEvalTestDetail(BaseModel):
    """Result of running a single unittest test class."""

    __test__ = False

    test_class_name: str
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_errored: int
    failures: list[str]
    errors: list[str]
    passed: bool


class ClassEvalTestResult(BaseModel):
    """Aggregated result across all test classes for a ClassEval task."""

    __test__ = False

    all_passed: bool
    total_tests_run: int
    total_tests_passed: int
    total_tests_failed: int
    total_tests_errored: int
    per_test_class: list[ClassEvalTestDetail]
    error: str | None = None


def _build_gt_code(import_statement: Any, solution_code: Any) -> str:
    if not isinstance(solution_code, str):
        raise ValueError("solution_code must be a string")
    if not isinstance(import_statement, list):
        raise ValueError("import_statement must be a list")
    imports = "\n".join(import_statement)
    if imports:
        return imports + "\n\n" + solution_code
    return solution_code


# ---------------------------------------------------------------------------
# Per-task fixes for dataset issues.
#
# Each entry maps a task_id to a dict with optional keys:
#   "solution": list of (old, new) string replacements on solution_code
#   "test":     list of (old, new) string replacements on test code
#   "test_classes_remove": list of test class names to drop
#   "reason":   human-readable explanation
# ---------------------------------------------------------------------------
_TASK_FIXES: dict[str, dict[str, Any]] = {
    "ClassEval_17": {
        "test": [
            # get_upcoming_events uses datetime.now(); 2024 dates are now in the past
            ("datetime(2024,", "datetime(2099,"),
        ],
        "reason": "Test dates in the past due to datetime.now() comparison",
    },
    "ClassEval_31": {
        "test": [
            # Float comparison at 1e-16 precision fails with assertEqual
            (
                "self.assertEqual(DataStatistics4.correlation_coefficient",
                "self.assertAlmostEqual(DataStatistics4.correlation_coefficient",
            ),
        ],
        "reason": "Float equality fails at 1e-16 precision",
    },
    "ClassEval_48": {
        "test": [
            # Hostname is machine-specific, not portable
            (
                "self.assertEqual(result, 'LAPTOP-2CS86KUM')",
                "self.assertIsInstance(result, str)",
            ),
        ],
        "reason": "Hardcoded hostname is machine-specific",
    },
    "ClassEval_51": {
        "solution": [
            # np.mat was removed in NumPy 2.0
            ("np.mat(", "np.asmatrix("),
            # float() on 1x1 matrix no longer works in NumPy 2.0
            ("float(ysum * xsum)", "float((ysum * xsum).item())"),
        ],
        "test": [
            # NumPy 2.0 returns np.float64 which differs at last decimal
            (
                "self.assertEqual(KappaCalculator.fleiss_kappa",
                "self.assertAlmostEqual(KappaCalculator.fleiss_kappa",
            ),
        ],
        "reason": "np.mat() removed and float() on 1x1 matrix broken in NumPy 2.0",
    },
    "ClassEval_58": {
        "auto_fail": True,
        "reason": "Solution bug: random mine placement can overlap, causing "
        "nondeterministic test failures (~20% of runs)",
    },
    "ClassEval_69": {
        "test_classes_remove": ["TestPDFHandler"],
        "reason": "TestPDFHandler is a fixture class with no test methods",
    },
}


def _apply_fixes(
    task_id: str,
    solution_code: str,
    test: str,
    test_classes: list[str],
) -> tuple[str, str, list[str], bool, bool, str | None]:
    """Apply hardcoded fixes.

    Returns (solution, test, test_classes, pp_solution, pp_test, auto_fail_reason).
    """
    fixes = _TASK_FIXES.get(task_id)
    if fixes is None:
        return solution_code, test, test_classes, False, False, None

    if fixes.get("auto_fail"):
        return solution_code, test, test_classes, False, False, fixes["reason"]

    pp_solution = False
    pp_test = False

    for old, new in fixes.get("solution", []):
        solution_code = solution_code.replace(old, new)
        pp_solution = True

    for old, new in fixes.get("test", []):
        test = test.replace(old, new)
        pp_test = True

    removals = fixes.get("test_classes_remove", [])
    if removals:
        test_classes = [tc for tc in test_classes if tc not in removals]
        pp_test = True

    return solution_code, test, test_classes, pp_solution, pp_test, None


class RawClassEvalTask(BaseModel):
    task_id: str
    class_name: str
    class_description: str
    class_constructor: str
    fields: list[str]
    import_statement: list[str]
    skeleton: str
    solution_code: str
    test: str
    test_classes: list[str]
    methods_info: list[MethodInfo]
    validated: bool = False
    postprocess_solution: bool = False
    postprocess_test: bool = False

    gt_code: str = Field(
        default_factory=lambda data: _build_gt_code(
            data.get("import_statement"), data.get("solution_code")
        )
    )

    @model_validator(mode="after")
    def validate_eval_task(self) -> Self:
        if self.validated:
            return self

        # Apply per-task fixes before validation
        (
            self.solution_code,
            self.test,
            self.test_classes,
            self.postprocess_solution,
            self.postprocess_test,
            auto_fail_reason,
        ) = _apply_fixes(self.task_id, self.solution_code, self.test, self.test_classes)

        if auto_fail_reason:
            raise ValueError(f"known dataset issue: {auto_fail_reason}")

        # Recompute gt_code if solution was modified
        if self.postprocess_solution:
            self.gt_code = _build_gt_code(self.import_statement, self.solution_code)

        ast.parse(self.gt_code)
        ast.parse(self.test)

        try:
            result = self.run_test_on_gt_solution()
        except Exception as exc:
            raise ValueError(
                "ground-truth solution raised an unexpected test error"
            ) from exc

        if not result.all_passed:
            raise ValueError("ground-truth solution does not pass its tests")
        self.validated = True
        return self

    def run_test(self, code: str) -> ClassEvalTestResult:
        parts = [p for p in (code, self.test) if p]
        full_source = "\n\n".join(parts)

        # Provide __file__ so tests using os.path.dirname(__file__) work in exec()
        with tempfile.TemporaryDirectory() as tmp_dir:
            return self._run_test_in_dir(full_source, tmp_dir)

    def _run_test_in_dir(self, full_source: str, tmp_dir: str) -> ClassEvalTestResult:
        namespace: dict[str, object] = {
            "__file__": os.path.join(tmp_dir, "__classeval__.py"),
        }
        try:
            exec(full_source, namespace)  # noqa: S102
        except Exception as exc:
            return ClassEvalTestResult(
                all_passed=False,
                total_tests_run=0,
                total_tests_passed=0,
                total_tests_failed=0,
                total_tests_errored=0,
                per_test_class=[],
                error=f"{type(exc).__name__}: {exc}",
            )

        per_test_class: list[ClassEvalTestDetail] = []
        for raw_name in self.test_classes:
            test_class_name = raw_name.strip()
            test_cls = namespace.get(test_class_name)
            if test_cls is None or not (
                isinstance(test_cls, type) and issubclass(test_cls, unittest.TestCase)
            ):
                per_test_class.append(
                    ClassEvalTestDetail(
                        test_class_name=test_class_name,
                        tests_run=0,
                        tests_passed=0,
                        tests_failed=0,
                        tests_errored=0,
                        failures=[f"Test class {test_class_name!r} not found"],
                        errors=[],
                        passed=False,
                    )
                )
                continue

            loader = unittest.TestLoader()
            suite = loader.loadTestsFromTestCase(test_cls)
            stream = io.StringIO()
            runner = unittest.TextTestRunner(stream=stream, verbosity=0)
            result = runner.run(suite)

            failures = [f"{tc}: {msg}" for tc, msg in result.failures]
            errors = [f"{tc}: {msg}" for tc, msg in result.errors]
            tests_passed = result.testsRun - len(result.failures) - len(result.errors)

            per_test_class.append(
                ClassEvalTestDetail(
                    test_class_name=test_class_name,
                    tests_run=result.testsRun,
                    tests_passed=tests_passed,
                    tests_failed=len(result.failures),
                    tests_errored=len(result.errors),
                    failures=failures,
                    errors=errors,
                    passed=(
                        len(result.failures) == 0
                        and len(result.errors) == 0
                        and result.testsRun > 0
                    ),
                )
            )

        total_run = sum(r.tests_run for r in per_test_class)
        total_passed = sum(r.tests_passed for r in per_test_class)
        total_failed = sum(r.tests_failed for r in per_test_class)
        total_errored = sum(r.tests_errored for r in per_test_class)

        return ClassEvalTestResult(
            all_passed=all(r.passed for r in per_test_class)
            and len(per_test_class) > 0,
            total_tests_run=total_run,
            total_tests_passed=total_passed,
            total_tests_failed=total_failed,
            total_tests_errored=total_errored,
            per_test_class=per_test_class,
        )

    def run_test_on_gt_solution(self) -> ClassEvalTestResult:
        return self.run_test(self.gt_code)
