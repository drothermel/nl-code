import { useQuery } from "@tanstack/react-query";
import type {
  DatasetCompareResponse,
  DatasetOption,
  DatasetOverviewResponse,
  RawDetailResponse,
  TaskDetailResponse,
  TaskListResponse,
} from "@/types/datasetExplorer";
import { apiFetch } from "./client";

export interface TaskListParams {
  search?: string;
  status?: "all" | "valid" | "flawed";
  sort?:
    | "task_id"
    | "description_length_chars"
    | "derived_code_length_chars"
    | "prompt_length_chars"
    | "raw_source_length_chars"
    | "test_length_chars";
  descending?: boolean;
  page?: number;
  per_page?: number;
}

function safeDecodePathSegment(segment: string) {
  try {
    return decodeURIComponent(segment);
  } catch {
    return segment;
  }
}

export function encodeDatasetKey(datasetKey: string) {
  return encodeURIComponent(datasetKey);
}

export function decodeDatasetKey(datasetKey: string) {
  return safeDecodePathSegment(datasetKey);
}

export function encodeTaskPath(taskId: string) {
  return taskId.split("/").map(encodeURIComponent).join("/");
}

export function decodeTaskPath(taskPath: string) {
  return taskPath.split("/").map(safeDecodePathSegment).join("/");
}

export function useDatasets() {
  return useQuery({
    queryKey: ["datasets"],
    queryFn: () => apiFetch<DatasetOption[]>("/api/datasets/"),
  });
}

export function useOverview(datasetKey: string) {
  return useQuery({
    queryKey: ["dataset-overview", datasetKey],
    queryFn: () =>
      apiFetch<DatasetOverviewResponse>(`/api/datasets/${encodeDatasetKey(datasetKey)}/overview`),
    enabled: !!datasetKey,
  });
}

export function useDatasetComparison() {
  return useQuery({
    queryKey: ["dataset-comparison"],
    queryFn: () => apiFetch<DatasetCompareResponse>("/api/datasets/compare"),
  });
}

export function useTasks(datasetKey: string, params: TaskListParams) {
  return useQuery({
    queryKey: ["dataset-tasks", datasetKey, params],
    queryFn: () =>
      apiFetch<TaskListResponse>(
        `/api/datasets/${encodeDatasetKey(datasetKey)}/tasks`,
        params as Record<string, string | number | boolean | undefined>,
      ),
    enabled: !!datasetKey,
  });
}

export function useTaskDetail(datasetKey: string, taskId: string) {
  return useQuery({
    queryKey: ["dataset-task-detail", datasetKey, taskId],
    queryFn: () =>
      apiFetch<TaskDetailResponse>(
        `/api/datasets/${encodeDatasetKey(datasetKey)}/tasks/${encodeTaskPath(taskId)}`,
      ),
    enabled: !!datasetKey && !!taskId,
  });
}

export function useRawDetail(datasetKey: string, taskId: string) {
  return useQuery({
    queryKey: ["dataset-raw-detail", datasetKey, taskId],
    queryFn: () =>
      apiFetch<RawDetailResponse>(
        `/api/datasets/${encodeDatasetKey(datasetKey)}/raw/${encodeTaskPath(taskId)}`,
      ),
    enabled: !!datasetKey && !!taskId,
  });
}
