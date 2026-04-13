import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";

export type TaskSortKey =
  | "task_id"
  | "description_length_chars"
  | "derived_code_length_chars"
  | "prompt_length_chars"
  | "raw_source_length_chars"
  | "test_length_chars";

export type TaskStatus = "all" | "valid" | "flawed";

export interface TaskFilters {
  search: string;
  status: TaskStatus;
  sort: TaskSortKey;
  descending: boolean;
  page: number;
}

const DEFAULTS: TaskFilters = {
  search: "",
  status: "all",
  sort: "task_id",
  descending: false,
  page: 1,
};

const SORT_KEYS = new Set<string>([
  "task_id",
  "description_length_chars",
  "derived_code_length_chars",
  "prompt_length_chars",
  "raw_source_length_chars",
  "test_length_chars",
]);

const STATUS_VALUES = new Set<string>(["all", "valid", "flawed"]);

function parseFilters(sp: URLSearchParams): TaskFilters {
  const rawSort = sp.get("sort");
  const rawStatus = sp.get("status");
  return {
    search: sp.get("search") ?? DEFAULTS.search,
    status: rawStatus && STATUS_VALUES.has(rawStatus) ? (rawStatus as TaskStatus) : DEFAULTS.status,
    sort: rawSort && SORT_KEYS.has(rawSort) ? (rawSort as TaskSortKey) : DEFAULTS.sort,
    descending: sp.get("descending") === "true",
    page: Math.max(1, Number(sp.get("page")) || DEFAULTS.page),
  };
}

function filtersToParams(filters: TaskFilters): URLSearchParams {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) {
    if (value !== DEFAULTS[key as keyof TaskFilters]) {
      params.set(key, String(value));
    }
  }
  return params;
}

export function useTaskFilters() {
  const [searchParams, setSearchParams] = useSearchParams();
  const filters = useMemo(() => parseFilters(searchParams), [searchParams]);

  const setFilters = useCallback(
    (patch: Partial<TaskFilters>) => {
      setSearchParams(
        (prev) => {
          const current = parseFilters(prev);
          const isFilterChange = Object.keys(patch).some((k) => k !== "page");
          return filtersToParams({
            ...current,
            ...patch,
            ...(isFilterChange ? { page: 1 } : {}),
          });
        },
        { replace: true },
      );
    },
    [setSearchParams],
  );

  return { filters, setFilters } as const;
}
