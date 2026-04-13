import type { ReactNode } from "react";
import { AlertTriangle, BarChart3, GitBranch, Hash } from "lucide-react";
import { useParams } from "react-router-dom";
import { useOverview } from "@/api/datasets";
import Plot from "@/components/charts/Plot";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ScatterPoint } from "@/types/datasetExplorer";

function StatCard({
  icon,
  label,
  value,
}: {
  icon: ReactNode;
  label: string;
  value: number;
}) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {icon}
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-2xl font-semibold">{value}</p>
      </CardContent>
    </Card>
  );
}

function getScatterAxisRange(points: ScatterPoint[]) {
  if (!points.length) {
    return { min: 0, max: 1 };
  }

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;

  for (const point of points) {
    min = Math.min(min, point.x, point.y);
    max = Math.max(max, point.x, point.y);
  }

  const span = max - min;
  const padding = span === 0 ? Math.max(Math.abs(max) * 0.05, 1) : span * 0.05;

  return {
    min: min - padding,
    max: max + padding,
  };
}

export default function OverviewPage() {
  const { datasetKey = "" } = useParams();
  const { data, isLoading, error } = useOverview(datasetKey);

  if (isLoading) {
    return <div className="p-8 text-sm text-muted-foreground">Loading overview…</div>;
  }

  if (error || !data) {
    return (
      <div className="p-8 text-sm text-destructive">
        Failed to load overview: {error?.message ?? "Unknown error"}
      </div>
    );
  }

  return (
    <div className="space-y-6 p-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">{data.dataset.label}</h1>
        <p className="text-sm text-muted-foreground">
          {data.dataset.dataset_id} · split: {data.dataset.split} · family: {data.dataset.family}
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <StatCard icon={<Hash className="h-3.5 w-3.5" />} label="Valid Raw Samples" value={data.counts.raw_sample_count} />
        <StatCard icon={<GitBranch className="h-3.5 w-3.5" />} label="Derived Tasks" value={data.counts.task_count} />
        <StatCard icon={<AlertTriangle className="h-3.5 w-3.5" />} label="Flawed Raw Samples" value={data.counts.flawed_count} />
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        {data.distributions.map((distribution) => (
          <Card key={distribution.key}>
            <CardHeader>
              <CardTitle className="text-base">{distribution.label}</CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <Plot
                data={[
                  {
                    type: "histogram",
                    x: distribution.values,
                    marker: { color: "#0f766e" },
                    hovertemplate: "%{x}<extra></extra>",
                  },
                ]}
                layout={{
                  height: 320,
                  bargap: 0.08,
                  xaxis: { title: { text: distribution.label } },
                  yaxis: { title: { text: "Count" } },
                }}
                style={{ height: "320px" }}
              />
            </CardContent>
          </Card>
        ))}

        {data.scatter_plots.map((plot) => (
          (() => {
            const axisRange = getScatterAxisRange(plot.points);

            return (
              <Card key={plot.key}>
                <CardHeader>
                  <CardTitle className="text-base">{plot.label}</CardTitle>
                </CardHeader>
                <CardContent className="pt-2">
                  <Plot
                    data={[
                      {
                        type: "scatter",
                        mode: "markers",
                        x: plot.points.map((point) => point.x),
                        y: plot.points.map((point) => point.y),
                        text: plot.points.map((point) => point.task_id),
                        hovertemplate: "%{text}<br>x=%{x}<br>y=%{y}<extra></extra>",
                        marker: { color: "#2563eb", size: 8, opacity: 0.75 },
                      },
                    ]}
                    layout={{
                      height: 320,
                      shapes: [
                        {
                          type: "line",
                          x0: axisRange.min,
                          y0: axisRange.min,
                          x1: axisRange.max,
                          y1: axisRange.max,
                          line: { color: "#dc2626", width: 1.5, dash: "dash" },
                        },
                      ],
                      xaxis: {
                        title: { text: plot.x_label },
                        range: [axisRange.min, axisRange.max],
                      },
                      yaxis: {
                        title: { text: plot.y_label },
                        range: [axisRange.min, axisRange.max],
                        scaleanchor: "x",
                        scaleratio: 1,
                      },
                    }}
                    style={{ height: "320px" }}
                  />
                </CardContent>
              </Card>
            );
          })()
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <BarChart3 className="h-4 w-4" />
            Top Flawed Error Groups
          </CardTitle>
        </CardHeader>
        <CardContent>
          {data.flawed_error_groups.length ? (
            <div className="space-y-3">
              {data.flawed_error_groups.map((group) => (
                <div key={group.error} className="rounded-md border p-4">
                  <div className="flex items-center justify-between gap-4">
                    <p className="text-sm font-medium">{group.error}</p>
                    <span className="rounded-full bg-destructive/10 px-2 py-1 text-xs font-medium text-destructive">
                      {group.count}
                    </span>
                  </div>
                  <p className="mt-2 text-xs text-muted-foreground">
                    Example tasks: {group.task_ids.join(", ")}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No flawed raw samples for this dataset.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
