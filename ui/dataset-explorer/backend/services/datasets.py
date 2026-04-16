import re
from collections import Counter
from statistics import median
from threading import Lock
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict

from backend.models.dataset_explorer import (
    ClassEvalRawDetail,
    CrossDatasetSeries,
    CrossDatasetSeriesValues,
    DatasetCompareResponse,
    DatasetCompareRow,
    DatasetOption,
    DatasetOverviewResponse,
    DatasetLandscapePoint,
    DatasetRefreshResponse,
    DerivedFieldSummary,
    FlawedErrorGroup,
    FlawedRawDetail,
    InspectorField,
    InspectorSection,
    MetricSummary,
    MetricDistribution,
    MetricScatter,
    NumericMetric,
    OverviewCounts,
    ProRawDetail,
    RawDetailResponse,
    RatioSummary,
    ScatterPoint,
    SummaryStats,
    TaskDetailResponse,
    TaskListResponse,
    TaskRow,
    HumanEvalRawDetail,
)
from nl_code.datasets.bigcodebench_lite_pro_dataset import BigCodeBenchLiteProDataset
from nl_code.datasets.bigcodebench_lite_pro_task import RawBigCodeBenchLiteProTask
from nl_code.datasets.classeval_dataset import ClassEvalDataset
from nl_code.datasets.classeval_task import RawClassEvalTask
from nl_code.datasets.dataset import Dataset
from nl_code.datasets.humaneval_dataset import HumanEvalDataset
from nl_code.datasets.humaneval_pro_dataset import HumanEvalProDataset
from nl_code.datasets.humaneval_pro_task import RawHumanEvalProTask
from nl_code.datasets.humaneval_task import RawHumanEvalTask
from nl_code.datasets.mbpp_pro_dataset import MbppProDataset
from nl_code.datasets.mbpp_pro_task import RawMbppProTask
from nl_code.datasets.task import CodeDataset, Task
from nl_code.evaluation.length import measure_length


class DatasetRegistryEntry(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: str
    dataset_id: CodeDataset
    label: str
    family: str
    dataset_type: type[Dataset]
    split: str


DATASET_REGISTRY: dict[str, DatasetRegistryEntry] = {
    "humaneval-plus": DatasetRegistryEntry(
        key="humaneval-plus",
        dataset_id=CodeDataset.HUMANEVAL_PLUS,
        label="HumanEval+",
        family="humaneval",
        dataset_type=HumanEvalDataset,
        split="test",
    ),
    "humaneval-pro": DatasetRegistryEntry(
        key="humaneval-pro",
        dataset_id=CodeDataset.HUMANEVAL_PRO,
        label="HumanEval Pro",
        family="pro",
        dataset_type=HumanEvalProDataset,
        split="train",
    ),
    "mbpp-pro": DatasetRegistryEntry(
        key="mbpp-pro",
        dataset_id=CodeDataset.MBPP_PRO,
        label="MBPP Pro",
        family="pro",
        dataset_type=MbppProDataset,
        split="train",
    ),
    "bigcodebench-lite-pro": DatasetRegistryEntry(
        key="bigcodebench-lite-pro",
        dataset_id=CodeDataset.BIGCODEBENCH_LITE_PRO,
        label="BigCodeBench Lite Pro",
        family="pro",
        dataset_type=BigCodeBenchLiteProDataset,
        split="train",
    ),
    "class-eval": DatasetRegistryEntry(
        key="class-eval",
        dataset_id=CodeDataset.CLASS_EVAL,
        label="ClassEval",
        family="classeval",
        dataset_type=ClassEvalDataset,
        split="test",
    ),
}

DATASET_CACHE: dict[str, Dataset] = {}
DATASET_LOAD_LOCKS: dict[str, Lock] = {
    key: Lock() for key in DATASET_REGISTRY
}

METRIC_LABELS: dict[str, str] = {
    "description_length_chars": "Description Length (chars)",
    "description_length_tokens": "Description Length (tokens)",
    "derived_code_length_chars": "Derived Code Length (chars)",
    "derived_code_length_tokens": "Derived Code Length (tokens)",
    "derived_code_length_lines": "Derived Code Length (lines)",
    "prompt_length_chars": "Prompt or Problem Length (chars)",
    "raw_source_length_chars": "Raw Source Length (chars)",
    "test_length_chars": "Test Length (chars)",
}

RATIO_DEFINITIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "description_to_prompt_ratio",
        "Description / Prompt",
        "description_length_chars",
        "prompt_length_chars",
    ),
    (
        "derived_code_to_raw_source_ratio",
        "Derived Code / Raw Source",
        "derived_code_length_chars",
        "raw_source_length_chars",
    ),
    (
        "test_to_derived_code_ratio",
        "Test / Derived Code",
        "test_length_chars",
        "derived_code_length_chars",
    ),
)


def list_dataset_options() -> list[DatasetOption]:
    return [_dataset_option(entry) for entry in DATASET_REGISTRY.values()]


def get_overview(dataset_key: str) -> DatasetOverviewResponse:
    entry = _get_entry(dataset_key)
    dataset = _get_dataset(entry)

    metric_values: dict[str, list[float]] = {key: [] for key in METRIC_LABELS}
    for task_id, task in dataset.tasks.items():
        raw = dataset.raw_samples[task_id]
        for key, value in _extract_metrics(entry.family, raw, task).items():
            if value is not None:
                metric_values[key].append(float(value))

    distributions = [
        MetricDistribution(key=key, label=METRIC_LABELS[key], values=values)
        for key, values in metric_values.items()
        if values
    ]

    return DatasetOverviewResponse(
        dataset=_dataset_option(entry),
        counts=OverviewCounts(
            raw_sample_count=len(dataset.raw_samples),
            task_count=len(dataset.tasks),
            flawed_count=len(dataset.flawed_raw_samples),
        ),
        distributions=distributions,
        scatter_plots=_build_scatter_plots(dataset, entry),
        flawed_error_groups=_build_error_groups(dataset),
    )


def refresh_dataset(dataset_key: str) -> DatasetRefreshResponse:
    entry = _get_entry(dataset_key)

    with DATASET_LOAD_LOCKS[entry.key]:
        DATASET_CACHE.pop(entry.key, None)
        dataset = entry.dataset_type().load()
        DATASET_CACHE[entry.key] = dataset

    return DatasetRefreshResponse(dataset=_dataset_option(entry), reloaded=True)


def get_comparison() -> DatasetCompareResponse:
    dataset_rows: list[DatasetCompareRow] = []
    metric_series_values: dict[str, list[CrossDatasetSeriesValues]] = {
        key: [] for key in METRIC_LABELS
    }
    ratio_series_values: dict[str, list[CrossDatasetSeriesValues]] = {
        key: [] for key, _, _, _ in RATIO_DEFINITIONS
    }
    landscape_points: list[DatasetLandscapePoint] = []

    for entry in DATASET_REGISTRY.values():
        dataset = _get_dataset(entry)
        metric_values: dict[str, list[float]] = {key: [] for key in METRIC_LABELS}
        ratio_values: dict[str, list[float]] = {
            key: [] for key, _, _, _ in RATIO_DEFINITIONS
        }

        for task_id, task in dataset.tasks.items():
            raw = dataset.raw_samples[task_id]
            metrics = _extract_metrics(entry.family, raw, task)

            for key, value in metrics.items():
                if value is not None:
                    metric_values[key].append(float(value))

            for ratio_key, _, numerator_key, denominator_key in RATIO_DEFINITIONS:
                numerator = metrics[numerator_key]
                denominator = metrics[denominator_key]
                if numerator is None or denominator in (None, 0):
                    continue
                ratio_values[ratio_key].append(float(numerator) / float(denominator))

        counts = OverviewCounts(
            raw_sample_count=len(dataset.raw_samples),
            task_count=len(dataset.tasks),
            flawed_count=len(dataset.flawed_raw_samples),
        )
        flawed_rate = (
            counts.flawed_count / counts.raw_sample_count
            if counts.raw_sample_count
            else 0.0
        )

        metrics = [
            MetricSummary(
                key=key,
                label=METRIC_LABELS[key],
                stats=_build_summary_stats(metric_values[key]),
            )
            for key in METRIC_LABELS
            if metric_values[key]
        ]
        ratios = [
            RatioSummary(
                key=ratio_key,
                label=label,
                numerator_key=numerator_key,
                denominator_key=denominator_key,
                stats=_build_summary_stats(ratio_values[ratio_key]),
            )
            for ratio_key, label, numerator_key, denominator_key in RATIO_DEFINITIONS
            if ratio_values[ratio_key]
        ]

        dataset_rows.append(
            DatasetCompareRow(
                dataset=_dataset_option(entry),
                counts=counts,
                flawed_rate=flawed_rate,
                metrics=metrics,
                ratios=ratios,
            )
        )

        for key, values in metric_values.items():
            if values:
                metric_series_values[key].append(
                    CrossDatasetSeriesValues(
                        dataset_key=entry.key,
                        dataset_label=entry.label,
                        values=values,
                    )
                )

        for ratio_key, values in ratio_values.items():
            if values:
                ratio_series_values[ratio_key].append(
                    CrossDatasetSeriesValues(
                        dataset_key=entry.key,
                        dataset_label=entry.label,
                        values=values,
                    )
                )

        prompt_stats = _stats_by_key(metrics).get("prompt_length_chars")
        derived_stats = _stats_by_key(metrics).get("derived_code_length_chars")
        if prompt_stats is not None and derived_stats is not None:
            landscape_points.append(
                DatasetLandscapePoint(
                    dataset_key=entry.key,
                    dataset_label=entry.label,
                    family=entry.family,
                    task_count=counts.task_count,
                    median_prompt_length_chars=prompt_stats.median,
                    median_derived_code_length_chars=derived_stats.median,
                )
            )

    return DatasetCompareResponse(
        datasets=dataset_rows,
        metric_series=[
            CrossDatasetSeries(
                key=key,
                label=METRIC_LABELS[key],
                datasets=metric_series_values[key],
            )
            for key in METRIC_LABELS
            if metric_series_values[key]
        ],
        ratio_series=[
            CrossDatasetSeries(
                key=ratio_key,
                label=label,
                datasets=ratio_series_values[ratio_key],
            )
            for ratio_key, label, _, _ in RATIO_DEFINITIONS
            if ratio_series_values[ratio_key]
        ],
        landscape_points=landscape_points,
    )


def list_tasks(
    dataset_key: str,
    *,
    search: str | None,
    status: str,
    sort: str,
    descending: bool,
    page: int,
    per_page: int,
) -> TaskListResponse:
    entry = _get_entry(dataset_key)
    dataset = _get_dataset(entry)
    rows = _build_task_rows(dataset, entry)

    if status != "all":
        rows = [row for row in rows if row.status == status]

    if search:
        needle = search.lower()
        rows = [
            row
            for row in rows
            if needle in row.task_id.lower()
            or needle in (row.entry_point_name or "").lower()
            or needle in (row.description_preview or "").lower()
            or needle in (row.error_summary or "").lower()
        ]

    rows.sort(key=lambda row: _row_sort_key(row, sort), reverse=descending)

    total = len(rows)
    start = (page - 1) * per_page
    end = start + per_page
    return TaskListResponse(
        dataset=_dataset_option(entry),
        total=total,
        page=page,
        per_page=per_page,
        rows=rows[start:end],
    )


def get_task_detail(dataset_key: str, task_id: str) -> TaskDetailResponse:
    entry = _get_entry(dataset_key)
    dataset = _get_dataset(entry)
    task = dataset.tasks.get(task_id)
    raw = dataset.raw_samples.get(task_id)

    if task is None or raw is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id!r} not found")

    metrics = _extract_metrics(entry.family, raw, task)
    ordered_task_ids = sorted(dataset.tasks.keys(), key=_natural_sort_key)
    prev_task_id, next_task_id = _neighbor_ids(ordered_task_ids, task_id)

    return TaskDetailResponse(
        dataset=_dataset_option(entry),
        task_id=task_id,
        entry_point_name=task.entry_point_name,
        description=task.description,
        gt_solution=task.gt_solution,
        metrics=[
            NumericMetric(key=key, label=METRIC_LABELS[key], value=float(value))
            for key, value in metrics.items()
            if value is not None
        ],
        derived_fields=_build_derived_fields(entry.family, raw, task),
        prev_task_id=prev_task_id,
        next_task_id=next_task_id,
    )


def get_raw_detail(dataset_key: str, task_id: str) -> RawDetailResponse:
    entry = _get_entry(dataset_key)
    dataset = _get_dataset(entry)
    ordered_ids = sorted(
        [*dataset.raw_samples.keys(), *dataset.flawed_raw_samples.keys()],
        key=_natural_sort_key,
    )
    prev_task_id, next_task_id = _neighbor_ids(ordered_ids, task_id)

    flawed = dataset.flawed_raw_samples.get(task_id)
    if flawed is not None:
        return FlawedRawDetail(
            detail_kind="flawed_raw_detail",
            dataset=_dataset_option(entry),
            task_id=task_id,
            title=f"{entry.label} flawed raw sample",
            error=flawed.error,
            sections=[
                InspectorSection(
                    title="Validation Error",
                    fields=[
                        InspectorField(key="error", label="Validation Error", kind="error", value=flawed.error),
                        InspectorField(key="raw_input", label="Raw Input JSON", kind="json", value=flawed.raw_input),
                    ],
                )
            ],
            raw_json=flawed.raw_input,
            prev_task_id=prev_task_id,
            next_task_id=next_task_id,
        )

    raw = dataset.raw_samples.get(task_id)
    task = dataset.tasks.get(task_id)
    if raw is None or task is None:
        raise HTTPException(status_code=404, detail=f"Raw task {task_id!r} not found")

    if entry.family == "humaneval":
        assert isinstance(raw, RawHumanEvalTask)
        return HumanEvalRawDetail(
            detail_kind="humaneval_raw_detail",
            dataset=_dataset_option(entry),
            task_id=task_id,
            title=f"{entry.label} raw sample",
            validated=raw.validated,
            sections=_humaneval_sections(raw),
            derived_fields=_build_derived_fields(entry.family, raw, task),
            raw_json=raw.model_dump(mode="json"),
            prev_task_id=prev_task_id,
            next_task_id=next_task_id,
        )

    if entry.family == "pro":
        assert isinstance(raw, RawHumanEvalProTask | RawMbppProTask | RawBigCodeBenchLiteProTask)
        return ProRawDetail(
            detail_kind="pro_raw_detail",
            dataset=_dataset_option(entry),
            task_id=task_id,
            title=f"{entry.label} raw sample",
            validated=raw.validated,
            sections=_pro_sections(raw),
            derived_fields=_build_derived_fields(entry.family, raw, task),
            raw_json=raw.model_dump(mode="json"),
            prev_task_id=prev_task_id,
            next_task_id=next_task_id,
        )

    if entry.family == "classeval":
        assert isinstance(raw, RawClassEvalTask)
        return ClassEvalRawDetail(
            detail_kind="classeval_raw_detail",
            dataset=_dataset_option(entry),
            task_id=task_id,
            title=f"{entry.label} raw sample",
            validated=raw.validated,
            sections=_classeval_sections(raw),
            derived_fields=_build_derived_fields(entry.family, raw, task),
            raw_json=raw.model_dump(mode="json"),
            prev_task_id=prev_task_id,
            next_task_id=next_task_id,
        )

    raise ValueError(f"Unsupported dataset family: {entry.family}")


def _build_task_rows(dataset: Dataset, entry: DatasetRegistryEntry) -> list[TaskRow]:
    rows: list[TaskRow] = []
    for task_id in sorted(dataset.tasks.keys(), key=_natural_sort_key):
        task = dataset.tasks[task_id]
        raw = dataset.raw_samples[task_id]
        metrics = _extract_metrics(entry.family, raw, task)
        rows.append(
            TaskRow(
                task_id=task_id,
                status="valid",
                has_derived_task=True,
                entry_point_name=task.entry_point_name,
                description_preview=_preview(task.description),
                description_length_chars=_safe_int(metrics["description_length_chars"]),
                description_length_tokens=_safe_int(metrics["description_length_tokens"]),
                derived_code_length_chars=_safe_int(metrics["derived_code_length_chars"]),
                derived_code_length_tokens=_safe_int(metrics["derived_code_length_tokens"]),
                derived_code_length_lines=_safe_int(metrics["derived_code_length_lines"]),
                prompt_length_chars=_safe_int(metrics["prompt_length_chars"]),
                raw_source_length_chars=_safe_int(metrics["raw_source_length_chars"]),
                test_length_chars=_safe_int(metrics["test_length_chars"]),
            )
        )

    for task_id in sorted(dataset.flawed_raw_samples.keys(), key=_natural_sort_key):
        flawed = dataset.flawed_raw_samples[task_id]
        rows.append(
            TaskRow(
                task_id=task_id,
                status="flawed",
                has_derived_task=False,
                error_summary=_preview(_error_group_key(flawed.error), 120),
            )
        )

    return rows


def _build_scatter_plots(dataset: Dataset, entry: DatasetRegistryEntry) -> list[MetricScatter]:
    description_vs_code: list[ScatterPoint] = []
    description_tokens_vs_code_tokens: list[ScatterPoint] = []
    prompt_vs_code: list[ScatterPoint] = []

    for task_id, task in dataset.tasks.items():
        raw = dataset.raw_samples[task_id]
        metrics = _extract_metrics(entry.family, raw, task)
        description_length = metrics["description_length_chars"]
        description_tokens = metrics["description_length_tokens"]
        prompt_length = metrics["prompt_length_chars"]
        code_length = metrics["derived_code_length_chars"]
        code_tokens = metrics["derived_code_length_tokens"]

        if description_length is not None and code_length is not None:
            description_vs_code.append(ScatterPoint(task_id=task_id, x=float(description_length), y=float(code_length)))
        if description_tokens is not None and code_tokens is not None:
            description_tokens_vs_code_tokens.append(
                ScatterPoint(task_id=task_id, x=float(description_tokens), y=float(code_tokens))
            )
        if prompt_length is not None and code_length is not None:
            prompt_vs_code.append(ScatterPoint(task_id=task_id, x=float(prompt_length), y=float(code_length)))

    plots = [
        MetricScatter(
            key="description-vs-code",
            label="Description vs Derived Code Length",
            x_key="description_length_chars",
            x_label=METRIC_LABELS["description_length_chars"],
            y_key="derived_code_length_chars",
            y_label=METRIC_LABELS["derived_code_length_chars"],
            points=description_vs_code,
        )
    ]
    if description_tokens_vs_code_tokens:
        plots.append(
            MetricScatter(
                key="description-tokens-vs-code-tokens",
                label="Description vs Derived Code Length (tokens)",
                x_key="description_length_tokens",
                x_label=METRIC_LABELS["description_length_tokens"],
                y_key="derived_code_length_tokens",
                y_label=METRIC_LABELS["derived_code_length_tokens"],
                points=description_tokens_vs_code_tokens,
            )
        )
    if prompt_vs_code:
        plots.append(
            MetricScatter(
                key="prompt-vs-code",
                label="Prompt or Problem vs Derived Code Length",
                x_key="prompt_length_chars",
                x_label=METRIC_LABELS["prompt_length_chars"],
                y_key="derived_code_length_chars",
                y_label=METRIC_LABELS["derived_code_length_chars"],
                points=prompt_vs_code,
            )
        )
    return plots


def _build_error_groups(dataset: Dataset) -> list[FlawedErrorGroup]:
    grouped_ids: dict[str, list[str]] = {}
    for task_id, flawed in dataset.flawed_raw_samples.items():
        grouped_ids.setdefault(_error_group_key(flawed.error), []).append(task_id)

    counts = Counter({key: len(ids) for key, ids in grouped_ids.items()})
    return [
        FlawedErrorGroup(
            error=error,
            count=count,
            task_ids=sorted(grouped_ids[error], key=_natural_sort_key)[:5],
        )
        for error, count in counts.most_common(10)
    ]


def _extract_metrics(family: str, raw: Any, task: Task) -> dict[str, int | None]:
    description_metrics = measure_length(task.description)
    derived_code_metrics = measure_length(task.gt_solution)
    metrics: dict[str, int | None] = {
        "description_length_chars": description_metrics.char_count,
        "description_length_tokens": description_metrics.token_count,
        "derived_code_length_chars": derived_code_metrics.char_count,
        "derived_code_length_tokens": derived_code_metrics.token_count,
        "derived_code_length_lines": _line_count(task.gt_solution),
        "prompt_length_chars": None,
        "raw_source_length_chars": None,
        "test_length_chars": None,
    }
    if family == "humaneval":
        assert isinstance(raw, RawHumanEvalTask)
        metrics["prompt_length_chars"] = len(raw.official_prompt)
        metrics["raw_source_length_chars"] = len(raw.gt_solution_with_comments)
        metrics["test_length_chars"] = len(raw.source__test)
        return metrics
    if family == "pro":
        assert isinstance(raw, RawHumanEvalProTask | RawMbppProTask | RawBigCodeBenchLiteProTask)
        metrics["prompt_length_chars"] = len(raw.new_official_prompt)
        metrics["raw_source_length_chars"] = len(raw.gt_solution_with_comments)
        metrics["test_length_chars"] = len(raw.source__test_code)
        return metrics

    assert isinstance(raw, RawClassEvalTask)
    metrics["prompt_length_chars"] = len(raw.class_description)
    metrics["raw_source_length_chars"] = len(raw.gt_code_with_comments)
    metrics["test_length_chars"] = len(raw.source__test)
    return metrics


def _build_derived_fields(family: str, raw: Any, task: Task) -> list[DerivedFieldSummary]:
    if family == "humaneval":
        assert isinstance(raw, RawHumanEvalTask)
        return [
            DerivedFieldSummary(name="Task.entry_point_name", value=task.entry_point_name, source="raw.entry_point"),
            DerivedFieldSummary(name="Task.description", value=task.description, source="raw.docstrings"),
            DerivedFieldSummary(
                name="Task.gt_solution",
                value=task.gt_solution,
                source="raw.gt_solution",
            ),
        ]
    if family == "pro":
        assert isinstance(raw, RawHumanEvalProTask | RawMbppProTask | RawBigCodeBenchLiteProTask)
        return [
            DerivedFieldSummary(name="Task.entry_point_name", value=task.entry_point_name, source="raw.new_entry_point"),
            DerivedFieldSummary(name="Task.description", value=task.description, source="raw.new_description"),
            DerivedFieldSummary(
                name="Task.gt_solution",
                value=task.gt_solution,
                source="raw.gt_solution",
            ),
        ]
    assert isinstance(raw, RawClassEvalTask)
    return [
        DerivedFieldSummary(name="Task.entry_point_name", value=task.entry_point_name, source="raw.class_name"),
        DerivedFieldSummary(name="Task.description", value=task.description, source="raw.class_description"),
        DerivedFieldSummary(name="Task.gt_solution", value=task.gt_solution, source="raw.gt_code"),
    ]


def _humaneval_sections(raw: RawHumanEvalTask) -> list[InspectorSection]:
    return [
        InspectorSection(
            title="Overview",
            fields=[
                InspectorField(key="entry_point", label="Entry Point", kind="text", value=raw.entry_point),
                InspectorField(key="validated", label="Validated", kind="text", value=str(raw.validated)),
                InspectorField(key="docstrings", label="Docstrings", kind="text", value=raw.docstrings),
                InspectorField(key="prompt_comment", label="Prompt Comment", kind="text", value=raw.prompt_comment),
            ],
        ),
        InspectorSection(
            title="Prompt and Solutions",
            fields=[
                InspectorField(key="source__prompt", label="Source Prompt", kind="code", value=raw.source__prompt),
                InspectorField(key="official_prompt", label="Official Prompt", kind="code", value=raw.official_prompt),
                InspectorField(key="function_stub", label="Function Stub", kind="code", value=raw.function_stub),
                InspectorField(
                    key="function_stub_with_comments",
                    label="Function Stub With Comments",
                    kind="code",
                    value=raw.function_stub_with_comments,
                ),
                InspectorField(
                    key="source__canonical_solution",
                    label="Source Canonical Solution",
                    kind="code",
                    value=raw.source__canonical_solution,
                ),
                InspectorField(
                    key="function_with_comments",
                    label="Function With Comments",
                    kind="code",
                    value=raw.function_with_comments,
                ),
                InspectorField(key="function", label="Function", kind="code", value=raw.function),
                InspectorField(
                    key="gt_solution_with_comments",
                    label="GT Solution With Comments",
                    kind="code",
                    value=raw.gt_solution_with_comments,
                ),
                InspectorField(
                    key="gt_solution",
                    label="GT Solution",
                    kind="code",
                    value=raw.gt_solution,
                ),
            ],
        ),
        InspectorSection(
            title="Tests",
            fields=[
                InspectorField(key="source__test", label="Source Test Harness", kind="code", value=raw.source__test),
                InspectorField(
                    key="assertion_test_code",
                    label="Assertion Test Code",
                    kind="code",
                    value=raw.assertion_test_code,
                ),
                InspectorField(key="test_inputs", label="Test Inputs", kind="json", value=raw.test_inputs),
                InspectorField(key="test_results", label="Test Results", kind="json", value=raw.test_results),
            ],
        ),
    ]


def _pro_sections(raw: RawHumanEvalProTask | RawMbppProTask | RawBigCodeBenchLiteProTask) -> list[InspectorSection]:
    return [
        InspectorSection(
            title="Overview",
            fields=[
                InspectorField(key="validated", label="Validated", kind="text", value=str(raw.validated)),
                InspectorField(key="new_entry_point", label="Derived Entry Point", kind="text", value=raw.new_entry_point),
                InspectorField(key="new_description", label="Derived Description", kind="text", value=raw.new_description),
            ],
        ),
        InspectorSection(
            title="Problems",
            fields=[
                InspectorField(key="source__raw_problem", label="Raw Problem", kind="code", value=raw.source__raw_problem),
                InspectorField(key="source__new_problem", label="New Problem", kind="code", value=raw.source__new_problem),
            ],
        ),
        InspectorSection(
            title="Solutions and Tests",
            fields=[
                InspectorField(key="source__raw_solution", label="Raw Solution", kind="code", value=raw.source__raw_solution),
                InspectorField(key="source__new_solution", label="New Solution", kind="code", value=raw.source__new_solution),
                InspectorField(
                    key="gt_solution_with_comments",
                    label="GT Solution With Comments",
                    kind="code",
                    value=raw.gt_solution_with_comments,
                ),
                InspectorField(key="gt_solution", label="GT Solution", kind="code", value=raw.gt_solution),
                InspectorField(key="source__test_code", label="Test Code", kind="code", value=raw.source__test_code),
            ],
        ),
    ]


def _classeval_sections(raw: RawClassEvalTask) -> list[InspectorSection]:
    try:
        test_result = raw.run_test_on_gt_solution().model_dump(mode="json")
    except Exception as exc:  # pragma: no cover - diagnostic surface
        test_result = {"error": f"{type(exc).__name__}: {exc}"}

    return [
        InspectorSection(
            title="Overview",
            fields=[
                InspectorField(key="class_name", label="Class Name", kind="text", value=raw.class_name),
                InspectorField(key="validated", label="Validated", kind="text", value=str(raw.validated)),
                InspectorField(
                    key="postprocess_solution",
                    label="Postprocessed Solution",
                    kind="text",
                    value=str(raw.postprocess_solution),
                ),
                InspectorField(
                    key="postprocess_test",
                    label="Postprocessed Test",
                    kind="text",
                    value=str(raw.postprocess_test),
                ),
            ],
        ),
        InspectorSection(
            title="Class Structure",
            fields=[
                InspectorField(key="class_description", label="Class Description", kind="text", value=raw.class_description),
                InspectorField(key="class_constructor", label="Class Constructor", kind="code", value=raw.class_constructor),
                InspectorField(key="fields", label="Fields", kind="json", value=raw.fields),
                InspectorField(key="import_statement", label="Import Statements", kind="json", value=raw.import_statement),
            ],
        ),
        InspectorSection(
            title="Code and Tests",
            fields=[
                InspectorField(key="skeleton", label="Skeleton", kind="code", value=raw.skeleton),
                InspectorField(key="solution_code", label="Solution Code", kind="code", value=raw.solution_code),
                InspectorField(key="gt_code", label="GT Code", kind="code", value=raw.gt_code),
                InspectorField(key="test", label="Test Code", kind="code", value=raw.test),
                InspectorField(key="test_classes", label="Test Classes", kind="json", value=raw.test_classes),
                InspectorField(key="methods_info", label="Methods Info", kind="json", value=raw.methods_info),
                InspectorField(key="test_result", label="GT Test Result", kind="json", value=test_result),
            ],
        ),
    ]


def _dataset_option(entry: DatasetRegistryEntry) -> DatasetOption:
    return DatasetOption(
        key=entry.key,
        dataset_id=entry.dataset_id,
        label=entry.label,
        family=entry.family,
        split=entry.split,
        loaded=entry.key in DATASET_CACHE,
    )


def _get_entry(dataset_key: str) -> DatasetRegistryEntry:
    entry = DATASET_REGISTRY.get(dataset_key)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown dataset {dataset_key!r}")
    return entry


def _get_dataset(entry: DatasetRegistryEntry) -> Dataset:
    cached = DATASET_CACHE.get(entry.key)
    if cached is not None:
        return cached

    with DATASET_LOAD_LOCKS[entry.key]:
        cached = DATASET_CACHE.get(entry.key)
        if cached is not None:
            return cached

        dataset = entry.dataset_type().load()
        DATASET_CACHE[entry.key] = dataset
        return dataset


def _neighbor_ids(ordered_ids: list[str], task_id: str) -> tuple[str | None, str | None]:
    try:
        index = ordered_ids.index(task_id)
    except ValueError:
        return None, None
    prev_task_id = ordered_ids[index - 1] if index > 0 else None
    next_task_id = ordered_ids[index + 1] if index < len(ordered_ids) - 1 else None
    return prev_task_id, next_task_id


def _row_sort_key(row: TaskRow, sort: str) -> tuple[int, str | int]:
    if sort == "task_id":
        return (0, row.task_id)
    numeric = getattr(row, sort, None)
    if isinstance(numeric, int):
        return (0, numeric)
    if numeric is None:
        return (1, row.task_id)
    return (0, str(numeric))


def _preview(value: str, limit: int = 80) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _safe_int(value: int | None) -> int | None:
    return int(value) if value is not None else None


def _error_group_key(error: str) -> str:
    return error.splitlines()[0][:160]


def _line_count(value: str) -> int:
    return len(value.rstrip().splitlines()) if value.strip() else 0


def _natural_sort_key(value: str) -> list[Any]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts if part]


def _build_summary_stats(values: list[float]) -> SummaryStats:
    ordered = sorted(values)
    return SummaryStats(
        count=len(ordered),
        min=ordered[0],
        median=float(median(ordered)),
        p90=_percentile(ordered, 0.9),
        max=ordered[-1],
    )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * weight


def _stats_by_key(
    metrics: list[MetricSummary],
) -> dict[str, SummaryStats]:
    return {metric.key: metric.stats for metric in metrics}
