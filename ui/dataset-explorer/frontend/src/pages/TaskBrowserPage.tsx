import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { encodeTaskPath, useTasks } from "@/api/datasets";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function TaskBrowserPage() {
  const { datasetKey = "" } = useParams();
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<"all" | "valid" | "flawed">("all");
  const [sort, setSort] = useState<
    | "task_id"
    | "description_length_chars"
    | "derived_code_length_chars"
    | "prompt_length_chars"
    | "raw_source_length_chars"
    | "test_length_chars"
  >("task_id");
  const [descending, setDescending] = useState(false);
  const [page, setPage] = useState(1);

  useEffect(() => {
    setPage(1);
  }, [search, status, sort, descending]);

  const { data, isLoading, error } = useTasks(datasetKey, {
    search: search || undefined,
    status,
    sort,
    descending,
    page,
    per_page: 25,
  });

  if (isLoading) {
    return <div className="p-8 text-sm text-muted-foreground">Loading tasks…</div>;
  }

  if (error || !data) {
    return (
      <div className="p-8 text-sm text-destructive">
        Failed to load tasks: {error?.message ?? "Unknown error"}
      </div>
    );
  }

  const maxPage = Math.max(1, Math.ceil(data.total / data.per_page));

  return (
    <div className="space-y-6 p-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Tasks</h1>
        <p className="text-sm text-muted-foreground">
          Browse valid and flawed tasks for {data.dataset.label}.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Filters</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-4">
          <label className="space-y-2 text-sm">
            <span className="font-medium">Search</span>
            <input
              className="w-full rounded-md border bg-background px-3 py-2"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="task id, entry point, error…"
            />
          </label>

          <label className="space-y-2 text-sm">
            <span className="font-medium">Status</span>
            <select
              className="w-full rounded-md border bg-background px-3 py-2"
              value={status}
              onChange={(event) =>
                setStatus(event.target.value as "all" | "valid" | "flawed")
              }
            >
              <option value="all">All</option>
              <option value="valid">Valid</option>
              <option value="flawed">Flawed</option>
            </select>
          </label>

          <label className="space-y-2 text-sm">
            <span className="font-medium">Sort</span>
            <select
              className="w-full rounded-md border bg-background px-3 py-2"
              value={sort}
              onChange={(event) =>
                setSort(
                  event.target.value as
                    | "task_id"
                    | "description_length_chars"
                    | "derived_code_length_chars"
                    | "prompt_length_chars"
                    | "raw_source_length_chars"
                    | "test_length_chars"
                )
              }
            >
              <option value="task_id">Task ID</option>
              <option value="description_length_chars">Description length</option>
              <option value="derived_code_length_chars">Derived code length</option>
              <option value="prompt_length_chars">Prompt length</option>
              <option value="raw_source_length_chars">Raw source length</option>
              <option value="test_length_chars">Test length</option>
            </select>
          </label>

          <label className="space-y-2 text-sm">
            <span className="font-medium">Order</span>
            <select
              className="w-full rounded-md border bg-background px-3 py-2"
              value={descending ? "desc" : "asc"}
              onChange={(event) => setDescending(event.target.value === "desc")}
            >
              <option value="asc">Ascending</option>
              <option value="desc">Descending</option>
            </select>
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
                  ? `/datasets/${datasetKey}/tasks/${encodeTaskPath(row.task_id)}`
                  : `/datasets/${datasetKey}/raw/${encodeTaskPath(row.task_id)}`;
                return (
                  <tr key={row.task_id} className="border-b align-top">
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
                    <td className="px-3 py-3 text-xs text-muted-foreground">
                      {row.description_preview ?? row.error_summary ?? "—"}
                    </td>
                    <td className="px-3 py-3 tabular-nums">{row.derived_code_length_chars ?? "—"}</td>
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
            <Button variant="outline" onClick={() => setPage((current) => Math.max(1, current - 1))} disabled={page <= 1}>
              Previous
            </Button>
            <p className="text-sm text-muted-foreground">
              Page {page} of {maxPage}
            </p>
            <Button
              variant="outline"
              onClick={() => setPage((current) => Math.min(maxPage, current + 1))}
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
