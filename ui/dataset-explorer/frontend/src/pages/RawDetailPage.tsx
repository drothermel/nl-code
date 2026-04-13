import { AlertTriangle } from "lucide-react";
import { Link, useParams } from "react-router-dom";
import { decodeTaskPath, encodeDatasetKey, encodeTaskPath, useRawDetail } from "@/api/datasets";
import { DerivedFieldsCard, InspectorSections } from "@/components/detail/InspectorSections";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CodeBlock } from "@/components/ui/code-block";
import { DetailSkeleton } from "@/components/ui/page-skeletons";
import { PageError, PageLoading } from "@/components/ui/page-status";

export default function RawDetailPage() {
  const { datasetKey = "", "*": taskPath = "" } = useParams();
  const taskId = decodeTaskPath(taskPath);
  const { data, isLoading, error } = useRawDetail(datasetKey, taskId);

  if (isLoading) {
    return <PageLoading label="raw detail" skeleton={<DetailSkeleton />} />;
  }

  if (error || !data) {
    return <PageError label="raw detail" error={error} />;
  }

  const hasDerivedFields = "derived_fields" in data;
  const encodedDatasetKey = encodeDatasetKey(datasetKey);

  return (
    <div className="animate-fade-in-up space-y-6 p-8">
      <div className="space-y-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight [text-wrap:balance]">
            {data.title}
          </h1>
          <p className="mt-1 text-sm text-muted-foreground">
            {data.task_id} · {data.dataset.label}
          </p>
        </div>

        <div className="flex flex-wrap gap-3 text-sm">
          {data.prev_task_id && (
            <Link
              className="text-primary hover:underline"
              to={`/datasets/${encodedDatasetKey}/raw/${encodeTaskPath(data.prev_task_id)}`}
            >
              ← Previous
            </Link>
          )}
          {data.next_task_id && (
            <Link
              className="text-primary hover:underline"
              to={`/datasets/${encodedDatasetKey}/raw/${encodeTaskPath(data.next_task_id)}`}
            >
              Next →
            </Link>
          )}
          {hasDerivedFields && (
            <Link
              className="text-primary hover:underline"
              to={`/datasets/${encodedDatasetKey}/tasks/${encodeTaskPath(data.task_id)}`}
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
            <CodeBlock variant="error">{data.error}</CodeBlock>
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
          <CodeBlock>{JSON.stringify(data.raw_json, null, 2)}</CodeBlock>
        </CardContent>
      </Card>
    </div>
  );
}
