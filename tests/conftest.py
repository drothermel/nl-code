import textwrap
from typing import Any
from unittest.mock import MagicMock

import pytest

from nl_code.datasets.humaneval_dataset import HumanEvalDataset
from nl_code.datasets.humaneval_task import RawHumanEvalTask


def make_humaneval_row(
    *,
    task_id: str = "HumanEval/0",
    entry_point: str = "add",
    prompt: str | None = None,
    canonical_solution: str | None = None,
    test: str | None = None,
) -> dict[str, Any]:
    if prompt is None:
        prompt = textwrap.dedent('''\
            def add(a: int, b: int) -> int:
                """Add two integers and return the result."""
        ''')
    if canonical_solution is None:
        canonical_solution = "    return a + b\n"
    if test is None:
        test = textwrap.dedent("""\
            def check(candidate):
                inputs = [[1, 2], [0, 0], [-1, 1]]
                results = [3, 0, 0]
                for inp, expected in zip(inputs, results):
                    assert candidate(*inp) == expected
        """)
    return {
        "task_id": task_id,
        "entry_point": entry_point,
        "prompt": prompt,
        "canonical_solution": canonical_solution,
        "test": test,
    }


def make_humaneval_pro_row(
    *,
    id: int = 0,
    raw_problem: str | None = None,
    raw_solution: str | None = None,
    new_problem: str | None = None,
    new_solution: str | None = None,
    test_code: str | None = None,
) -> dict[str, Any]:
    if raw_problem is None:
        raw_problem = textwrap.dedent('''\
            def add(a: int, b: int) -> int:
                """Add two integers."""
        ''')
    if raw_solution is None:
        raw_solution = "    return a + b\n"
    if new_problem is None:
        new_problem = textwrap.dedent("""\
            # Given a list of pairs, add each pair and return the list of sums.
            def add_pairs(pairs: list[tuple[int, int]]) -> list[int]:
        """)
    if new_solution is None:
        new_solution = (
            "    result = []\n"
            "    for a, b in pairs:\n"
            "        result.append(add(a, b))\n"
            "    return result\n"
        )
    if test_code is None:
        test_code = textwrap.dedent("""\
            assert add_pairs([(1, 2), (3, 4)]) == [3, 7]
            assert add_pairs([]) == []
            assert add_pairs([(0, 0)]) == [0]
        """)
    return {
        "id": id,
        "raw_problem": raw_problem,
        "raw_solution": raw_solution,
        "new_problem": new_problem,
        "new_solution": new_solution,
        "test_code": test_code,
    }


def make_mbpp_pro_row(
    *,
    id: int = 0,
    raw_problem: str | None = None,
    raw_solution: str | None = None,
    new_problem: str | None = None,
    new_solution: str | None = None,
    test_code: str | None = None,
) -> dict[str, Any]:
    if raw_problem is None:
        raw_problem = textwrap.dedent('''\
            def add(a: int, b: int) -> int:
                """Add two integers."""
        ''')
    if raw_solution is None:
        raw_solution = "    return a + b\n"
    if new_problem is None:
        new_problem = textwrap.dedent("""\
            # Given a list of pairs, add each pair and return the list of sums.
            def add_pairs(pairs: list[tuple[int, int]]) -> list[int]:
        """)
    if new_solution is None:
        new_solution = (
            "    result = []\n"
            "    for a, b in pairs:\n"
            "        result.append(add(a, b))\n"
            "    return result\n"
        )
    if test_code is None:
        test_code = textwrap.dedent("""\
            assert add_pairs([(1, 2), (3, 4)]) == [3, 7]
            assert add_pairs([]) == []
            assert add_pairs([(0, 0)]) == [0]
        """)
    return {
        "id": id,
        "raw_problem": raw_problem,
        "raw_solution": raw_solution,
        "new_problem": new_problem,
        "new_solution": new_solution,
        "test_code": test_code,
    }


def make_bigcodebench_lite_pro_row(
    *,
    id: str = "BigCodeBench/23",
    raw_problem: str | None = None,
    raw_solution: str | None = None,
    new_problem: str | None = None,
    new_solution: str | None = None,
    test_code: str | None = None,
) -> dict[str, Any]:
    if raw_problem is None:
        raw_problem = textwrap.dedent('''\
            def multiply(a: int, b: int) -> int:
                """Multiply two integers."""
        ''')
    if raw_solution is None:
        raw_solution = "    return a * b\n"
    if new_problem is None:
        new_problem = textwrap.dedent("""\
            # Given a list of pairs, multiply each pair and return the list of products.
            def multiply_pairs(pairs: list[tuple[int, int]]) -> list[int]:
        """)
    if new_solution is None:
        new_solution = (
            "    result = []\n"
            "    for a, b in pairs:\n"
            "        result.append(multiply(a, b))\n"
            "    return result\n"
        )
    if test_code is None:
        test_code = textwrap.dedent("""\
            assert multiply_pairs([(2, 3), (4, 5)]) == [6, 20]
            assert multiply_pairs([]) == []
            assert multiply_pairs([(0, 5)]) == [0]
        """)
    return {
        "id": id,
        "raw_problem": raw_problem,
        "raw_solution": raw_solution,
        "new_problem": new_problem,
        "new_solution": new_solution,
        "test_code": test_code,
    }


def make_classeval_row(
    *,
    task_id: str = "ClassEval_0",
    class_name: str = "Calculator",
    solution_code: str | None = None,
    test: str | None = None,
    test_classes: list[str] | None = None,
    methods_info: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if solution_code is None:
        solution_code = textwrap.dedent("""\
            class Calculator:
                \"\"\"A simple calculator.\"\"\"

                def __init__(self):
                    self.result = 0

                def add(self, a, b):
                    return a + b

                def subtract(self, a, b):
                    return a - b
        """)
    if test is None:
        test = textwrap.dedent("""\
            import unittest

            class TestCalculatorAdd(unittest.TestCase):
                def test_add_positive(self):
                    calc = Calculator()
                    self.assertEqual(calc.add(1, 2), 3)

                def test_add_zero(self):
                    calc = Calculator()
                    self.assertEqual(calc.add(0, 0), 0)

            class TestCalculatorSubtract(unittest.TestCase):
                def test_subtract_positive(self):
                    calc = Calculator()
                    self.assertEqual(calc.subtract(3, 1), 2)
        """)
    if test_classes is None:
        test_classes = ["TestCalculatorAdd", "TestCalculatorSubtract"]
    if methods_info is None:
        methods_info = [
            {
                "method_name": "add",
                "method_description": "def add(self, a, b):\n    return a + b",
                "solution_code": "def add(self, a, b):\n    return a + b",
                "test_class": "TestCalculatorAdd",
                "test_code": textwrap.dedent("""\
                    class TestCalculatorAdd(unittest.TestCase):
                        def test_add_positive(self):
                            calc = Calculator()
                            self.assertEqual(calc.add(1, 2), 3)

                        def test_add_zero(self):
                            calc = Calculator()
                            self.assertEqual(calc.add(0, 0), 0)
                """),
                "dependencies": {
                    "Standalone": True,
                    "lib_dependencies": [],
                    "field_dependencies": [],
                    "method_dependencies": [],
                },
            },
            {
                "method_name": "subtract",
                "method_description": "def subtract(self, a, b):\n    return a - b",
                "solution_code": "def subtract(self, a, b):\n    return a - b",
                "test_class": "TestCalculatorSubtract",
                "test_code": textwrap.dedent("""\
                    class TestCalculatorSubtract(unittest.TestCase):
                        def test_subtract_positive(self):
                            calc = Calculator()
                            self.assertEqual(calc.subtract(3, 1), 2)
                """),
                "dependencies": {
                    "Standalone": True,
                    "lib_dependencies": [],
                    "field_dependencies": [],
                    "method_dependencies": [],
                },
            },
        ]
    return {
        "task_id": task_id,
        "class_name": class_name,
        "class_description": "A simple calculator.",
        "class_constructor": "def __init__(self):\n    self.result = 0",
        "fields": ["self.result"],
        "import_statement": [],
        "skeleton": textwrap.dedent("""\
            class Calculator:
                \"\"\"A simple calculator.\"\"\"

                def __init__(self):
                    pass

                def add(self, a, b):
                    pass

                def subtract(self, a, b):
                    pass
        """),
        "solution_code": solution_code,
        "test": test,
        "test_classes": test_classes,
        "methods_info": methods_info,
    }


@pytest.fixture
def valid_row() -> dict[str, Any]:
    return make_humaneval_row()


@pytest.fixture
def valid_raw_task(valid_row: dict[str, Any]) -> RawHumanEvalTask:
    return RawHumanEvalTask.model_validate(valid_row)


def mock_hf_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    mock = MagicMock()
    mock.__len__ = lambda self: len(rows)
    mock.__iter__ = lambda self: iter(rows)
    return mock


@pytest.fixture
def loaded_dataset(monkeypatch: pytest.MonkeyPatch) -> HumanEvalDataset:
    rows = [
        make_humaneval_row(task_id="HumanEval/0"),
        make_humaneval_row(task_id="HumanEval/1"),
    ]
    monkeypatch.setattr(
        "nl_code.datasets.dataset.load_dataset",
        lambda *a, **kw: mock_hf_dataset(rows),
    )
    ds = HumanEvalDataset()
    ds.load()
    return ds
