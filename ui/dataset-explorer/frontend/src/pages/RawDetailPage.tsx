import { AlertTriangle } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { decodeTaskPath, encodeTaskPath, useRawDetail } from "@/api/datasets";
import {
  DerivedFieldsCard,
  InspectorSections,
} from "@/components/detail/InspectorSections";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function RawDetailPage() {
  const { datasetKey = "", "*": taskPath = "" } = useParams();
  const taskId = decodeTaskPath(taskPath);
  const { data, isLoading, error } = useRawDetail(datasetKey, taskId);

  if (isLoading) {
    return <div className="p-8 text-sm text-muted-foreground">Loading raw detail…</div>;
  }

  if (error || !data) {
    return (
      <div className="p-8 text-sm text-destructive">
        Failed to load raw detail: {error?.message ?? "Unknown error"}
      </div>
    );
  }

  const hasDerivedFields = "derived_fields" in data;

  return (
    <div className="space-y-6 p-8">
      <div className="space-y-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">{data.title}</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {data.task_id} · {data.dataset.label}
          </p>
        </div>

        <div className="flex flex-wrap gap-3 text-sm">
          {data.prev_task_id && (
            <Link
              className="text-primary hover:underline"
              to={`/datasets/${datasetKey}/raw/${encodeTaskPath(data.prev_task_id)}`}
            >
              ← Previous
            </Link>
          )}
          {data.next_task_id && (
            <Link
              className="text-primary hover:underline"
              to={`/datasets/${datasetKey}/raw/${encodeTaskPath(data.next_task_id)}`}
            >
              Next →
            </Link>
          )}
          {hasDerivedFields && (
            <Link
              className="text-primary hover:underline"
              to={`/datasets/${datasetKey}/tasks/${encodeTaskPath(data.task_id)}`}
            >
              Open derived task
            </Link>
          )}
        </div>
      </div>

      {data.detail_kind === "flawed_raw_detail" && (
        <Card className="border-destructive/40">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base text-destructive">
              <AlertTriangle className="h-4 w-4" />
              Validation Failure
            </CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="overflow-x-auto rounded-md border border-destructive/30 bg-white p-4 text-xs leading-6 text-destructive">
              {data.error}
            </pre>
          </CardContent>
        </Card>
      )}

      {hasDerivedFields && <DerivedFieldsCard derivedFields={data.derived_fields} />}

      <InspectorSections sections={data.sections} />

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Raw JSON</CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="overflow-x-auto rounded-md border border-slate-200 bg-white p-4 text-xs leading-6 text-slate-900">
            {JSON.stringify(data.raw_json, null, 2)}
          </pre>
        </CardContent>
      </Card>
    </div>
  );
}
