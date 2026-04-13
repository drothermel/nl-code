import { Navigate } from "react-router-dom";
import { useDatasets } from "@/api/datasets";

export default function HomeRedirect() {
  const { data, isLoading } = useDatasets();

  if (isLoading) {
    return <div className="p-8 text-sm text-muted-foreground">Loading datasets…</div>;
  }

  const firstDataset = data?.[0];
  if (!firstDataset) {
    return <div className="p-8 text-sm text-muted-foreground">No datasets available.</div>;
  }

  return <Navigate to={`/datasets/${firstDataset.key}/overview`} replace />;
}
