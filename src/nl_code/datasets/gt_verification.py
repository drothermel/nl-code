from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from nl_code.code_execution.models import CodeExecutionInfrastructureError

TItem = TypeVar("TItem")
TResult = TypeVar("TResult")


def run_batched_with_infra_isolation(
    items_by_task_id: list[tuple[str, TItem]],
    run_batch: Callable[[list[TItem]], list[TResult]],
) -> tuple[dict[str, TResult], dict[str, CodeExecutionInfrastructureError]]:
    """Run batched Docker work while isolating infra failures to single samples."""

    results: dict[str, TResult] = {}
    infra_failures: dict[str, CodeExecutionInfrastructureError] = {}

    def process_chunk(chunk: list[tuple[str, TItem]]) -> None:
        if not chunk:
            return

        try:
            chunk_results = run_batch([item for _, item in chunk])
        except CodeExecutionInfrastructureError as exc:
            if len(chunk) == 1:
                infra_failures[chunk[0][0]] = exc
                return
            midpoint = len(chunk) // 2
            process_chunk(chunk[:midpoint])
            process_chunk(chunk[midpoint:])
            return

        for (task_id, _item), result in zip(chunk, chunk_results, strict=True):
            results[task_id] = result

    process_chunk(items_by_task_id)
    return results, infra_failures
