from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from nl_code.datasets.task import CodeDataset


class DatasetOption(BaseModel):
    key: str
    dataset_id: CodeDataset
    label: str
    family: str
    split: str
    loaded: bool


class MetricDistribution(BaseModel):
    key: str
    label: str
    values: list[float]


class ScatterPoint(BaseModel):
    task_id: str
    x: float
    y: float


class MetricScatter(BaseModel):
    key: str
    label: str
    x_key: str
    x_label: str
    y_key: str
    y_label: str
    points: list[ScatterPoint]


class FlawedErrorGroup(BaseModel):
    error: str
    count: int
    task_ids: list[str]


class OverviewCounts(BaseModel):
    raw_sample_count: int
    task_count: int
    flawed_count: int


class DatasetOverviewResponse(BaseModel):
    dataset: DatasetOption
    counts: OverviewCounts
    distributions: list[MetricDistribution]
    scatter_plots: list[MetricScatter]
    flawed_error_groups: list[FlawedErrorGroup]


class DatasetRefreshResponse(BaseModel):
    dataset: DatasetOption
    reloaded: bool


class SummaryStats(BaseModel):
    count: int
    min: float
    median: float
    p90: float
    max: float


class MetricSummary(BaseModel):
    key: str
    label: str
    stats: SummaryStats


class RatioSummary(BaseModel):
    key: str
    label: str
    numerator_key: str
    denominator_key: str
    stats: SummaryStats


class CrossDatasetSeriesValues(BaseModel):
    dataset_key: str
    dataset_label: str
    values: list[float]


class CrossDatasetSeries(BaseModel):
    key: str
    label: str
    datasets: list[CrossDatasetSeriesValues]


class DatasetLandscapePoint(BaseModel):
    dataset_key: str
    dataset_label: str
    family: str
    task_count: int
    median_prompt_length_chars: float
    median_derived_code_length_chars: float


class DatasetCompareRow(BaseModel):
    dataset: DatasetOption
    counts: OverviewCounts
    flawed_rate: float
    metrics: list[MetricSummary]
    ratios: list[RatioSummary]


class DatasetCompareResponse(BaseModel):
    datasets: list[DatasetCompareRow]
    metric_series: list[CrossDatasetSeries]
    ratio_series: list[CrossDatasetSeries]
    landscape_points: list[DatasetLandscapePoint]


class TaskRow(BaseModel):
    task_id: str
    status: Literal["valid", "flawed"]
    has_derived_task: bool
    entry_point_name: str | None = None
    description_preview: str | None = None
    description_length_chars: int | None = None
    description_length_tokens: int | None = None
    derived_code_length_chars: int | None = None
    derived_code_length_tokens: int | None = None
    derived_code_length_lines: int | None = None
    prompt_length_chars: int | None = None
    raw_source_length_chars: int | None = None
    test_length_chars: int | None = None
    error_summary: str | None = None


class TaskListResponse(BaseModel):
    dataset: DatasetOption
    total: int
    page: int
    per_page: int
    rows: list[TaskRow]


class NumericMetric(BaseModel):
    key: str
    label: str
    value: float


class DerivedFieldSummary(BaseModel):
    name: str
    value: str
    source: str


class TaskDetailResponse(BaseModel):
    dataset: DatasetOption
    task_id: str
    entry_point_name: str
    description: str
    gt_solution: str
    metrics: list[NumericMetric]
    derived_fields: list[DerivedFieldSummary]
    prev_task_id: str | None
    next_task_id: str | None


class InspectorField(BaseModel):
    key: str
    label: str
    kind: Literal["text", "code", "json", "list", "error"]
    value: Any


class InspectorSection(BaseModel):
    title: str
    fields: list[InspectorField]


class RawDetailBase(BaseModel):
    dataset: DatasetOption
    task_id: str
    title: str
    validated: bool | None
    sections: list[InspectorSection]
    derived_fields: list[DerivedFieldSummary]
    raw_json: dict[str, Any]
    prev_task_id: str | None
    next_task_id: str | None


class HumanEvalRawDetail(RawDetailBase):
    detail_kind: Literal["humaneval_raw_detail"]


class ProRawDetail(RawDetailBase):
    detail_kind: Literal["pro_raw_detail"]


class ClassEvalRawDetail(RawDetailBase):
    detail_kind: Literal["classeval_raw_detail"]


class FlawedRawDetail(BaseModel):
    detail_kind: Literal["flawed_raw_detail"]
    dataset: DatasetOption
    task_id: str
    title: str
    error: str
    sections: list[InspectorSection]
    raw_json: dict[str, Any]
    prev_task_id: str | None
    next_task_id: str | None


RawDetailResponse = Annotated[
    HumanEvalRawDetail | ProRawDetail | ClassEvalRawDetail | FlawedRawDetail,
    Field(discriminator="detail_kind"),
]
