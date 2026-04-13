import { Link, useParams } from "react-router-dom";
import { encodeDatasetKey, encodeTaskPath, useTasks } from "@/api/datasets";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input, NativeSelect } from "@/components/ui/input";
import { TaskBrowserSkeleton } from "@/components/ui/page-skeletons";
import { PageError, PageLoading } from "@/components/ui/page-status";
import type { TaskSortKey, TaskStatus } from "@/hooks/useTaskFilters";
import { useTaskFilters } from "@/hooks/useTaskFilters";

export default function TaskBrowserPage() {
  const { datasetKey = "" } = useParams();
  const { filters, setFilters } = useTaskFilters();
  const { search, status, sort, descending, page } = filters;

  const { data, isLoading, error } = useTasks(datasetKey, {
    search: search || undefined,
    status,
    sort,
    descending,
    page,
    per_page: 25,
  });

  if (isLoading) {
    return <PageLoading label="tasks" skeleton={<TaskBrowserSkeleton />} />;
  }

  if (error || !data) {
    return <PageError label="tasks" error={error} />;
  }

  const maxPage = Math.max(1, Math.ceil(data.total / data.per_page));
  const encodedDatasetKey = encodeDatasetKey(datasetKey);

  return (
    <div className="animate-fade-in-up space-y-6 p-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight [text-wrap:balance]">Tasks</h1>
        <p className="text-sm text-muted-foreground">
          Browse valid and flawed tasks for {data.dataset.label}.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Filters</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-4">
          <label htmlFor="filter-search" className="space-y-2 text-sm">
            <span className="font-medium">Search</span>
            <Input
              id="filter-search"
              name="search"
              autoComplete="off"
              value={search}
              onChange={(event) => setFilters({ search: event.target.value })}
              placeholder="task id, entry point, error…"
            />
          </label>

          <label htmlFor="filter-status" className="space-y-2 text-sm">
            <span className="font-medium">Status</span>
            <NativeSelect
              id="filter-status"
              value={status}
              onChange={(event) => setFilters({ status: event.target.value as TaskStatus })}
            >
              <option value="all">All</option>
              <option value="valid">Valid</option>
              <option value="flawed">Flawed</option>
            </NativeSelect>
          </label>

          <label htmlFor="filter-sort" className="space-y-2 text-sm">
            <span className="font-medium">Sort</span>
            <NativeSelect
              id="filter-sort"
              value={sort}
              onChange={(event) => setFilters({ sort: event.target.value as TaskSortKey })}
            >
              <option value="task_id">Task ID</option>
              <option value="description_length_chars">Description length</option>
              <option value="derived_code_length_chars">Derived code length</option>
              <option value="prompt_length_chars">Prompt length</option>
              <option value="raw_source_length_chars">Raw source length</option>
              <option value="test_length_chars">Test length</option>
            </NativeSelect>
          </label>

          <label htmlFor="filter-order" className="space-y-2 text-sm">
            <span className="font-medium">Order</span>
            <NativeSelect
              id="filter-order"
              value={descending ? "desc" : "asc"}
              onChange={(event) => setFilters({ descending: event.target.value === "desc" })}
            >
              <option value="asc">Ascending</option>
              <option value="desc">Descending</option>
            </NativeSelect>
          </label>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="text-base">Results</CardTitle>
          <div className="text-sm text-muted-foreground">{data.total} rows</div>
        </CardHeader>
        <CardContent className="overflow-x-auto pt-2">
          <table className="w-full min-w-[960px] text-sm">
            <thead>
              <tr className="border-b text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="px-3 py-2">Task</th>
                <th className="px-3 py-2">Status</th>
                <th className="px-3 py-2">Entry Point</th>
                <th className="px-3 py-2">Description</th>
                <th className="px-3 py-2">Derived Code</th>
                <th className="px-3 py-2">Prompt</th>
                <th className="px-3 py-2">Raw Source</th>
                <th className="px-3 py-2">Test</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => {
                const target = row.has_derived_task
                  ? `/datasets/${encodedDatasetKey}/tasks/${encodeTaskPath(row.task_id)}`
                  : `/datasets/${encodedDatasetKey}/raw/${encodeTaskPath(row.task_id)}`;
                return (
                  <tr
                    key={row.task_id}
                    className="border-b align-top transition-colors hover:bg-muted/50"
                  >
                    <td className="px-3 py-3 font-mono text-xs">
                      <Link className="text-primary hover:underline" to={target}>
                        {row.task_id}
                      </Link>
                    </td>
                    <td className="px-3 py-3">
                      <Badge variant={row.status === "valid" ? "secondary" : "destructive"}>
                        {row.status}
                      </Badge>
                    </td>
                    <td className="px-3 py-3 font-mono text-xs text-muted-foreground">
                      {row.entry_point_name ?? "—"}
                    </td>
                    <td className="max-w-xs truncate px-3 py-3 text-xs text-muted-foreground">
                      {row.description_preview ?? row.error_summary ?? "—"}
                    </td>
                    <td className="px-3 py-3 tabular-nums">
                      {row.derived_code_length_chars ?? "—"}
                    </td>
                    <td className="px-3 py-3 tabular-nums">{row.prompt_length_chars ?? "—"}</td>
                    <td className="px-3 py-3 tabular-nums">{row.raw_source_length_chars ?? "—"}</td>
                    <td className="px-3 py-3 tabular-nums">{row.test_length_chars ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {!data.rows.length && (
            <p className="py-8 text-center text-sm text-muted-foreground">
              No rows matched the current filters.
            </p>
          )}

          <div className="mt-6 flex items-center justify-between">
            <Button
              variant="outline"
              onClick={() => setFilters({ page: Math.max(1, page - 1) })}
              disabled={page <= 1}
            >
              Previous
            </Button>
            <p className="text-sm text-muted-foreground">
              Page {page} of {maxPage}
            </p>
            <Button
              variant="outline"
              onClick={() => setFilters({ page: Math.min(maxPage, page + 1) })}
              disabled={page >= maxPage}
            >
              Next
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
