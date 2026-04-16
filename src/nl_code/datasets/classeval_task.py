import ast
from typing import Any, ClassVar, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nl_code.code_execution.models import UnittestResult
from nl_code.code_execution.runner import run_unittest_test
from nl_code.code_parsing import remove_docstrings_and_comments


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


def _require_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _require_string_list(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a list[str]")
    return cast(list[str], list(value))


def _first_docstring_expr(
    node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Expr | None:
    if not node.body:
        return None

    first_stmt = node.body[0]
    if not isinstance(first_stmt, ast.Expr):
        return None
    if not isinstance(first_stmt.value, ast.Constant):
        return None
    if not isinstance(first_stmt.value.value, str):
        return None
    return first_stmt


def _remove_docstrings(source: str) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source.rstrip() + "\n" if source.strip() else ""
    line_numbers_to_remove: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            docstring_expr = _first_docstring_expr(node)
            if docstring_expr is None:
                continue
            assert docstring_expr.lineno is not None
            assert docstring_expr.end_lineno is not None
            line_numbers_to_remove.update(
                range(docstring_expr.lineno, docstring_expr.end_lineno + 1)
            )

    remaining_lines = [
        line
        for line_number, line in enumerate(source.splitlines(keepends=True), start=1)
        if line_number not in line_numbers_to_remove
    ]
    if not remaining_lines:
        return ""

    return "".join(remaining_lines).rstrip() + "\n"


def _build_import_block(import_statement: Any) -> str:
    imports = _require_string_list(import_statement, name="import_statement")
    if not imports:
        return ""
    return "\n".join(imports) + "\n"


def _build_gt_code(import_statement: Any, solution_code: Any) -> str:
    solution_code_str = _require_string(solution_code, name="solution_code")
    imports = _build_import_block(import_statement).rstrip()
    if imports:
        return imports + "\n\n" + solution_code_str
    return solution_code_str


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
        "auto_fail": True,
        "reason": "Hostname lookup for 0.0.0.0 is environment-dependent; "
        "the dataset's ground truth and tests do not behave consistently "
        "across Docker environments",
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
    model_config = ConfigDict(populate_by_name=True)
    raw_source_fields: ClassVar[tuple[str, ...]] = (
        "class_name",
        "class_description",
        "class_constructor",
        "fields",
        "import_statement",
        "skeleton",
        "solution_code",
        "test",
        "test_classes",
        "methods_info",
    )
    non_code_fields: ClassVar[tuple[str, ...]] = (
        "task_id",
        "validated",
        "version",
        "postprocess_solution",
        "postprocess_test",
        "auto_fail_reason",
        "source__class_name",
        "source__class_description",
        "source__fields",
        "source__import_statement",
        "source__test_classes",
        "source__methods_info",
        "class_name",
        "class_description",
        "fields",
        "import_statement",
        "test_classes",
        "methods_info",
    )

    task_id: str
    source__class_name: str
    source__class_description: str
    source__class_constructor: str
    source__fields: list[str]
    source__import_statement: list[str]
    source__skeleton: str
    source__solution_code: str
    source__test: str
    source__test_classes: list[str]
    source__methods_info: list[MethodInfo]
    version: Literal["v1", "v2"] = "v2"
    validated: bool = False
    postprocess_solution: bool = False
    postprocess_test: bool = False
    auto_fail_reason: str | None = None

    class_name: str = ""
    class_description: str = ""
    class_constructor: str = ""
    fields: list[str] = Field(default_factory=list)
    import_statement: list[str] = Field(default_factory=list)
    official_skeleton: str = ""
    class_stub_with_comments: str = ""
    class_stub: str = ""
    import_block: str = ""
    solution_code: str = ""
    test: str = ""
    test_classes: list[str] = Field(default_factory=list)
    methods_info: list[MethodInfo] = Field(default_factory=list)
    gt_code_with_comments: str = ""
    gt_code: str = ""

    @model_validator(mode="before")
    @classmethod
    def populate_source_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        for field_name in cls.raw_source_fields:
            source_field_name = f"source__{field_name}"
            if source_field_name not in normalized and field_name in normalized:
                normalized[source_field_name] = normalized[field_name]
        return normalized

    @model_validator(mode="after")
    def validate_eval_task(self) -> Self:
        self.class_name = self.source__class_name
        self.class_description = self.source__class_description
        self.class_constructor = self.source__class_constructor
        self.fields = list(self.source__fields)
        self.import_statement = list(self.source__import_statement)
        self.official_skeleton = self.source__skeleton
        self.class_stub_with_comments = self.official_skeleton
        self.class_stub = _remove_docstrings(self.official_skeleton)
        self.import_block = _build_import_block(self.import_statement)
        self.methods_info = list(self.source__methods_info)

        (
            self.solution_code,
            self.test,
            self.test_classes,
            self.postprocess_solution,
            self.postprocess_test,
            self.auto_fail_reason,
        ) = _apply_fixes(
            self.task_id,
            self.source__solution_code,
            self.source__test,
            list(self.source__test_classes),
        )

        self.gt_code_with_comments = _build_gt_code(
            self.import_statement, self.solution_code
        )
        self.gt_code = remove_docstrings_and_comments(self.gt_code_with_comments)
        return self

    def run_test(self, code: str) -> ClassEvalTestResult:
        result = run_unittest_test(code, self.test, self.test_classes)
        return _class_eval_result_from_unittest_result(result)

    def run_test_on_gt_solution(self) -> ClassEvalTestResult:
        return self.run_test(self.gt_code)


def _class_eval_result_from_unittest_result(
    result: UnittestResult,
) -> ClassEvalTestResult:
    return ClassEvalTestResult(
        all_passed=result.all_passed,
        total_tests_run=result.total_tests_run,
        total_tests_passed=result.total_tests_passed,
        total_tests_failed=result.total_tests_failed,
        total_tests_errored=result.total_tests_errored,
        per_test_class=[
            ClassEvalTestDetail(
                test_class_name=detail.test_class_name,
                tests_run=detail.tests_run,
                tests_passed=detail.tests_passed,
                tests_failed=detail.tests_failed,
                tests_errored=detail.tests_errored,
                failures=detail.failures,
                errors=detail.errors,
                passed=detail.passed,
            )
            for detail in result.per_test_class
        ],
        error=result.error,
    )
