import pytest

from nl_code.code_execution.models import CodeExecutionInfrastructureError
from nl_code.datasets.gt_verification import run_batched_with_infra_isolation


def test_run_batched_with_infra_isolation_uses_code_execution_error_type() -> None:
    failing_items = {3}

    def run_batch(items: list[int]) -> list[str]:
        if any(item in failing_items for item in items):
            raise CodeExecutionInfrastructureError(
                stage="docker_unavailable",
                execution_mode="docker_worker",
                detail=f"failed for {items}",
            )
        return [f"ok-{item}" for item in items]

    results, infra_failures = run_batched_with_infra_isolation(
        [("a", 1), ("b", 2), ("c", 3), ("d", 4)],
        run_batch,
    )

    assert results == {"a": "ok-1", "b": "ok-2", "d": "ok-4"}
    assert set(infra_failures) == {"c"}
    assert infra_failures["c"].detail == "failed for [3]"


def test_run_batched_with_infra_isolation_does_not_catch_other_exception_types() -> None:
    def run_batch(items: list[int]) -> list[int]:
        raise RuntimeError(f"unexpected failure for {items}")

    with pytest.raises(RuntimeError, match=r"unexpected failure for \[1\]"):
        run_batched_with_infra_isolation([("a", 1)], run_batch)
