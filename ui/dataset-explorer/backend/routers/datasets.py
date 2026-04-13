from typing import Literal

from fastapi import APIRouter, Query

from backend.models.dataset_explorer import (
    DatasetOption,
    DatasetOverviewResponse,
    RawDetailResponse,
    TaskDetailResponse,
    TaskListResponse,
)
from backend.services.datasets import (
    get_overview,
    get_raw_detail,
    get_task_detail,
    list_dataset_options,
    list_tasks,
)

router = APIRouter()


@router.get("/", response_model=list[DatasetOption])
def get_datasets():
    return list_dataset_options()


@router.get("/{dataset_key}/overview", response_model=DatasetOverviewResponse)
def dataset_overview(dataset_key: str):
    return get_overview(dataset_key)


@router.get("/{dataset_key}/tasks", response_model=TaskListResponse)
def dataset_tasks(
    dataset_key: str,
    search: str | None = Query(None),
    status: Literal["all", "valid", "flawed"] = Query("all"),
    sort: Literal[
        "task_id",
        "description_length_chars",
        "derived_code_length_chars",
        "prompt_length_chars",
        "raw_source_length_chars",
        "test_length_chars",
    ] = Query("task_id"),
    descending: bool = Query(False),
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=200),
):
    return list_tasks(
        dataset_key,
        search=search,
        status=status,
        sort=sort,
        descending=descending,
        page=page,
        per_page=per_page,
    )


@router.get("/{dataset_key}/tasks/{task_id:path}", response_model=TaskDetailResponse)
def dataset_task_detail(dataset_key: str, task_id: str):
    return get_task_detail(dataset_key, task_id)


@router.get("/{dataset_key}/raw/{task_id:path}", response_model=RawDetailResponse)
def dataset_raw_detail(dataset_key: str, task_id: str):
    return get_raw_detail(dataset_key, task_id)
