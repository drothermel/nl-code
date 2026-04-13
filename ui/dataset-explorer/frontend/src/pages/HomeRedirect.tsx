import { Navigate } from "react-router-dom";
import { encodeDatasetKey, useDatasets } from "@/api/datasets";
import { Button } from "@/components/ui/button";

export default function HomeRedirect() {
  const { data, isLoading, isError, error, refetch } = useDatasets();

  if (isLoading) {
    return <div className="p-8 text-sm text-muted-foreground">Loading datasets…</div>;
  }

  if (isError) {
    return (
      <div className="space-y-4 p-8">
        <p className="text-sm text-destructive">
          Failed to load datasets: {error?.message ?? "Unknown error"}
        </p>
        <Button variant="outline" onClick={() => void refetch()}>
          Retry
        </Button>
      </div>
    );
  }

  const firstDataset = data?.[0];
  if (!firstDataset) {
    return <div className="p-8 text-sm text-muted-foreground">No datasets available.</div>;
  }

  return <Navigate to={`/datasets/${encodeDatasetKey(firstDataset.key)}/overview`} replace />;
}
