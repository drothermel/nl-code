import { Link, useParams } from "react-router-dom";
import { decodeTaskPath, encodeDatasetKey, encodeTaskPath, useTaskDetail } from "@/api/datasets";
import PythonCodeBlock from "@/components/code/PythonCodeBlock";
import { isCodeDerivedField } from "@/components/detail/InspectorSections";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function TaskDetailPage() {
  const { datasetKey = "", "*": taskPath = "" } = useParams();
  const taskId = decodeTaskPath(taskPath);
  const { data, isLoading, error } = useTaskDetail(datasetKey, taskId);

  if (isLoading) {
    return <div className="p-8 text-sm text-muted-foreground">Loading task detail…</div>;
  }

  if (error || !data) {
    return (
      <div className="p-8 text-sm text-destructive">
        Failed to load task detail: {error?.message ?? "Unknown error"}
      </div>
    );
  }
  const encodedDatasetKey = encodeDatasetKey(datasetKey);

  return (
    <div className="space-y-6 p-8">
      <div className="space-y-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight [text-wrap:balance]">
            {data.task_id}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Derived task view for {data.dataset.label}
          </p>
        </div>

        <div className="flex flex-wrap gap-3 text-sm">
          {data.prev_task_id && (
            <Link
              className="text-primary hover:underline"
              to={`/datasets/${encodedDatasetKey}/tasks/${encodeTaskPath(data.prev_task_id)}`}
            >
              ← Previous
            </Link>
          )}
          {data.next_task_id && (
            <Link
              className="text-primary hover:underline"
              to={`/datasets/${encodedDatasetKey}/tasks/${encodeTaskPath(data.next_task_id)}`}
            >
              Next →
            </Link>
          )}
          <Link
            className="text-primary hover:underline"
            to={`/datasets/${encodedDatasetKey}/raw/${encodeTaskPath(data.task_id)}`}
          >
            Open raw inspector
          </Link>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {data.metrics.map((metric) => (
          <Card key={metric.key}>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs uppercase tracking-wide text-muted-foreground">
                {metric.label}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-semibold tabular-nums">{metric.value}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Description</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="whitespace-pre-wrap rounded-md border bg-card p-4 text-sm leading-7 text-card-foreground">
            {data.description}
          </pre>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Derived Field Mapping</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {data.derived_fields.map((field) => (
            <div key={field.name} className="rounded-md border p-3">
              <div className="text-sm font-medium">{field.name}</div>
              <div className="mt-1 text-xs text-muted-foreground">{field.source}</div>
              <div className="mt-3">
                {isCodeDerivedField(field.name) ? (
                  <PythonCodeBlock code={field.value} className="p-0" />
                ) : (
                  <pre className="overflow-x-auto rounded-md border bg-card p-3 text-xs leading-6 text-card-foreground">
                    {field.value}
                  </pre>
                )}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Derived Code</CardTitle>
        </CardHeader>
        <CardContent>
          <PythonCodeBlock code={data.gt_solution} />
        </CardContent>
      </Card>
    </div>
  );
}
