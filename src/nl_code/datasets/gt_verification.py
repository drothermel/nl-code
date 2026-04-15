from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from dr_docker import (
    run_batch_with_failure_isolation as _run_batch_with_failure_isolation,
)

from nl_code.code_execution.models import CodeExecutionInfrastructureError

TItem = TypeVar("TItem")
TResult = TypeVar("TResult")


def run_batched_with_infra_isolation(
    items_by_task_id: list[tuple[str, TItem]],
    run_batch: Callable[[list[TItem]], list[TResult]],
) -> tuple[dict[str, TResult], dict[str, CodeExecutionInfrastructureError]]:
    """Compatibility wrapper around dr-docker's batch failure isolation helper."""

    return _run_batch_with_failure_isolation(
        items_by_task_id,
        run_batch,
        infra_failure_type=CodeExecutionInfrastructureError,
    )
