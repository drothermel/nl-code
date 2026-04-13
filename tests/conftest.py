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
        "nl_code.datasets.humaneval_dataset.load_dataset",
        lambda *a, **kw: mock_hf_dataset(rows),
    )
    ds = HumanEvalDataset()
    ds.load_raw_samples()
    return ds
