import { BarChart3, Bug, Database, Layers3, TableProperties } from "lucide-react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import { decodeDatasetKey, encodeDatasetKey, useDatasets } from "@/api/datasets";
import { API_BASE } from "@/api/client";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

const GLOBAL_NAV_ITEMS = [
  { to: "/compare", label: "Compare", icon: <Layers3 className="h-4 w-4" /> },
];

const DATASET_NAV_ITEMS = [
  { suffix: "overview", label: "Overview", icon: <BarChart3 className="h-4 w-4" /> },
  { suffix: "tasks", label: "Tasks", icon: <TableProperties className="h-4 w-4" /> },
];

function currentDatasetKey(pathname: string) {
  const match = pathname.match(/^\/datasets\/([^/]+)/);
  return match ? decodeDatasetKey(match[1]) : "";
}

export default function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();
  const { data: datasets, isLoading } = useDatasets();
  const selectedDataset = currentDatasetKey(location.pathname);
  const apiBaseLabel = API_BASE || "/api";

  return (
    <aside className="flex min-h-screen w-72 flex-col border-r bg-card">
      <div className="border-b px-5 py-4">
        <div className="flex items-center gap-2">
          <Database className="h-5 w-5 text-primary" />
          <span className="font-semibold tracking-tight">Dataset Explorer</span>
        </div>
        <p className="mt-2 text-sm text-muted-foreground">
          Debug raw samples and understand dataset shape through aggregate views.
        </p>
      </div>

      <nav className="border-b p-3">
        {GLOBAL_NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                isActive
                  ? "bg-accent text-accent-foreground font-medium"
                  : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
              )
            }
          >
            {item.icon}
            {item.label}
          </NavLink>
        ))}
      </nav>

      <div className="border-b px-4 py-4">
        <label className="mb-2 block text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Dataset
        </label>
        <select
          className="w-full rounded-md border bg-background px-3 py-2 text-sm"
          value={selectedDataset}
          disabled={isLoading || !datasets?.length}
          onChange={(event) => {
            if (!event.target.value) {
              return;
            }
            navigate(`/datasets/${encodeDatasetKey(event.target.value)}/overview`);
          }}
        >
          <option value="">
            {datasets?.length ? "Jump to dataset..." : "Loading datasets..."}
          </option>
          {datasets?.map((dataset) => (
            <option key={dataset.key} value={dataset.key}>
              {dataset.label}
            </option>
          ))}
        </select>
      </div>

      <nav className="flex-1 space-y-1 p-3">
        <div className="px-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Dataset views
        </div>
        {selectedDataset ? (
          DATASET_NAV_ITEMS.map((item) => (
            <NavLink
              key={item.suffix}
              to={`/datasets/${encodeDatasetKey(selectedDataset)}/${item.suffix}`}
              className={({ isActive }) =>
                cn(
                  "flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
                  isActive
                    ? "bg-accent text-accent-foreground font-medium"
                    : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
                )
              }
            >
              {item.icon}
              {item.label}
            </NavLink>
          ))
        ) : (
          <p className="px-3 py-2 text-sm text-muted-foreground">
            Select a dataset to open overview and task views.
          </p>
        )}
      </nav>

      <div className="space-y-2 border-t p-4">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Bug className="h-3.5 w-3.5" />
          Raw inspectors show dataset-specific fields plus raw JSON.
        </div>
        <Badge variant="outline" className="font-mono text-[11px]">
          API: {apiBaseLabel}
        </Badge>
      </div>
    </aside>
  );
}
