import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "./client";
import type {
  DatasetOption,
  DatasetOverviewResponse,
  RawDetailResponse,
  TaskDetailResponse,
  TaskListResponse,
} from "@/types/datasetExplorer";

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

export function encodeTaskPath(taskId: string) {
  return taskId.split("/").map(encodeURIComponent).join("/");
}

export function decodeTaskPath(taskPath: string) {
  return taskPath.split("/").map(decodeURIComponent).join("/");
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
      apiFetch<DatasetOverviewResponse>(`/api/datasets/${datasetKey}/overview`),
    enabled: !!datasetKey,
  });
}

export function useTasks(datasetKey: string, params: TaskListParams) {
  return useQuery({
    queryKey: ["dataset-tasks", datasetKey, params],
    queryFn: () =>
      apiFetch<TaskListResponse>(
        `/api/datasets/${datasetKey}/tasks`,
        params as Record<string, string | number | boolean | undefined>
      ),
    enabled: !!datasetKey,
  });
}

export function useTaskDetail(datasetKey: string, taskId: string) {
  return useQuery({
    queryKey: ["dataset-task-detail", datasetKey, taskId],
    queryFn: () =>
      apiFetch<TaskDetailResponse>(
        `/api/datasets/${datasetKey}/tasks/${encodeTaskPath(taskId)}`
      ),
    enabled: !!datasetKey && !!taskId,
  });
}

export function useRawDetail(datasetKey: string, taskId: string) {
  return useQuery({
    queryKey: ["dataset-raw-detail", datasetKey, taskId],
    queryFn: () =>
      apiFetch<RawDetailResponse>(
        `/api/datasets/${datasetKey}/raw/${encodeTaskPath(taskId)}`
      ),
    enabled: !!datasetKey && !!taskId,
  });
}
