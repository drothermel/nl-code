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
