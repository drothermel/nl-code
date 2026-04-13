export interface DatasetOption {
  key: string;
  dataset_id: string;
  label: string;
  family: string;
  split: string;
  loaded: boolean;
}

export interface MetricDistribution {
  key: string;
  label: string;
  values: number[];
}

export interface ScatterPoint {
  task_id: string;
  x: number;
  y: number;
}

export interface MetricScatter {
  key: string;
  label: string;
  x_key: string;
  x_label: string;
  y_key: string;
  y_label: string;
  points: ScatterPoint[];
}

export interface FlawedErrorGroup {
  error: string;
  count: number;
  task_ids: string[];
}

export interface OverviewCounts {
  raw_sample_count: number;
  task_count: number;
  flawed_count: number;
}

export interface DatasetOverviewResponse {
  dataset: DatasetOption;
  counts: OverviewCounts;
  distributions: MetricDistribution[];
  scatter_plots: MetricScatter[];
  flawed_error_groups: FlawedErrorGroup[];
}

export interface SummaryStats {
  count: number;
  min: number;
  median: number;
  p90: number;
  max: number;
}

export interface MetricSummary {
  key: string;
  label: string;
  stats: SummaryStats;
}

export interface RatioSummary {
  key: string;
  label: string;
  numerator_key: string;
  denominator_key: string;
  stats: SummaryStats;
}

export interface CrossDatasetSeriesValues {
  dataset_key: string;
  dataset_label: string;
  values: number[];
}

export interface CrossDatasetSeries {
  key: string;
  label: string;
  datasets: CrossDatasetSeriesValues[];
}

export interface DatasetLandscapePoint {
  dataset_key: string;
  dataset_label: string;
  family: string;
  task_count: number;
  median_prompt_length_chars: number;
  median_derived_code_length_chars: number;
}

export interface DatasetCompareRow {
  dataset: DatasetOption;
  counts: OverviewCounts;
  flawed_rate: number;
  metrics: MetricSummary[];
  ratios: RatioSummary[];
}

export interface DatasetCompareResponse {
  datasets: DatasetCompareRow[];
  metric_series: CrossDatasetSeries[];
  ratio_series: CrossDatasetSeries[];
  landscape_points: DatasetLandscapePoint[];
}

export interface TaskRow {
  task_id: string;
  status: "valid" | "flawed";
  has_derived_task: boolean;
  entry_point_name: string | null;
  description_preview: string | null;
  description_length_chars: number | null;
  description_length_tokens: number | null;
  derived_code_length_chars: number | null;
  derived_code_length_tokens: number | null;
  derived_code_length_lines: number | null;
  prompt_length_chars: number | null;
  raw_source_length_chars: number | null;
  test_length_chars: number | null;
  error_summary: string | null;
}

export interface TaskListResponse {
  dataset: DatasetOption;
  total: number;
  page: number;
  per_page: number;
  rows: TaskRow[];
}

export interface NumericMetric {
  key: string;
  label: string;
  value: number;
}

export interface DerivedFieldSummary {
  name: string;
  value: string;
  source: string;
}

export interface TaskDetailResponse {
  dataset: DatasetOption;
  task_id: string;
  entry_point_name: string;
  description: string;
  gt_solution: string;
  metrics: NumericMetric[];
  derived_fields: DerivedFieldSummary[];
  prev_task_id: string | null;
  next_task_id: string | null;
}

export interface InspectorField {
  key: string;
  label: string;
  kind: "text" | "code" | "json" | "list" | "error";
  value: unknown;
}

export interface InspectorSection {
  title: string;
  fields: InspectorField[];
}

export interface RawDetailBase {
  dataset: DatasetOption;
  task_id: string;
  title: string;
  validated: boolean | null;
  sections: InspectorSection[];
  derived_fields: DerivedFieldSummary[];
  raw_json: Record<string, unknown>;
  prev_task_id: string | null;
  next_task_id: string | null;
}

export interface HumanEvalRawDetail extends RawDetailBase {
  detail_kind: "humaneval_raw_detail";
}

export interface ProRawDetail extends RawDetailBase {
  detail_kind: "pro_raw_detail";
}

export interface ClassEvalRawDetail extends RawDetailBase {
  detail_kind: "classeval_raw_detail";
}

export interface FlawedRawDetail {
  detail_kind: "flawed_raw_detail";
  dataset: DatasetOption;
  task_id: string;
  title: string;
  error: string;
  sections: InspectorSection[];
  raw_json: Record<string, unknown>;
  prev_task_id: string | null;
  next_task_id: string | null;
}

export type RawDetailResponse =
  | HumanEvalRawDetail
  | ProRawDetail
  | ClassEvalRawDetail
  | FlawedRawDetail;
