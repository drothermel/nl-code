import textwrap

from nl_code.datasets.humaneval_task import RawHumanEvalTask
from nl_code.optim.humaneval_dspy_sample import code_stub, function_stub


def _raw_task(*, prompt: str) -> RawHumanEvalTask:
    return RawHumanEvalTask.model_validate(
        {
            "task_id": "HumanEval/0",
            "entry_point": "add",
            "source": {
                "prompt": prompt,
                "canonical_solution": "    return a + b\n",
                "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
            },
        }
    )


def test_code_stub_preserves_docstrings_and_comments() -> None:
    prompt = textwrap.dedent(
        """\
        # module note
        def add(a, b):
            \"\"\"Add two numbers.\"\"\"
        """
    )
    raw = _raw_task(prompt=prompt)

    assert '"""' in raw.code_stub
    assert "# module note" in raw.code_stub
    assert code_stub(raw) == raw.code_stub


def test_function_stub_strips_docstrings_and_preserves_comments() -> None:
    prompt = textwrap.dedent(
        """\
        # module note
        def add(a, b):
            \"\"\"Add two numbers.\"\"\"
        """
    )
    raw = _raw_task(prompt=prompt)

    assert '"""' not in raw.function_stub
    assert "# module note" in raw.function_stub
    assert "def add(a, b):" in raw.function_stub
    assert function_stub(raw) == raw.function_stub
