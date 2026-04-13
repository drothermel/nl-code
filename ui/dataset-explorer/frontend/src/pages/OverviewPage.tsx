import { AlertTriangle, BarChart3, GitBranch, Hash } from "lucide-react";
import type { ReactNode } from "react";
import { useParams } from "react-router-dom";
import { useOverview } from "@/api/datasets";
import Plot from "@/components/charts/Plot";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { OverviewSkeleton } from "@/components/ui/page-skeletons";
import { PageError, PageLoading } from "@/components/ui/page-status";
import { CHART_COLORS } from "@/lib/chartColors";
import { metricUnit, UNIT_TITLES } from "@/lib/metrics";
import type { MetricDistribution, MetricScatter, ScatterPoint } from "@/types/datasetExplorer";

const DISTRIBUTION_ORDER: Record<string, number> = {
  description_length_chars: 0,
  derived_code_length_chars: 1,
  prompt_length_chars: 2,
  raw_source_length_chars: 3,
  test_length_chars: 4,
  description_length_tokens: 5,
  derived_code_length_tokens: 6,
  derived_code_length_lines: 7,
};

const SCATTER_ORDER: Record<string, number> = {
  "description-vs-code": 0,
  "prompt-vs-code": 1,
  "description-tokens-vs-code-tokens": 2,
};

const HISTOGRAM_BIN_COUNT = 16;

function StatCard({ icon, label, value }: { icon: ReactNode; label: string; value: number }) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {icon}
          {label}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <p className="text-2xl font-semibold tabular-nums">{value}</p>
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

function getDistributionAxisRange(distributions: MetricDistribution[]) {
  const values = distributions.flatMap((distribution) => distribution.values);
  if (!values.length) {
    return undefined;
  }

  const maxValue = Math.max(...values);
  const paddedMax = maxValue === 0 ? 1 : maxValue * 1.05;
  return [0, paddedMax] as [number, number];
}

function getHistogramBinSize(axisRange: [number, number] | undefined) {
  if (!axisRange) {
    return 1;
  }

  const span = axisRange[1] - axisRange[0];
  if (span <= 0) {
    return 1;
  }

  return span / HISTOGRAM_BIN_COUNT;
}

function getHistogramMaxCount(
  distributions: MetricDistribution[],
  axisRange: [number, number] | undefined,
  binSize: number,
) {
  if (!distributions.length || !axisRange) {
    return undefined;
  }

  let maxCount = 0;
  for (const distribution of distributions) {
    const bins = Array.from({ length: HISTOGRAM_BIN_COUNT }, () => 0);
    for (const value of distribution.values) {
      if (value < axisRange[0] || value > axisRange[1]) {
        continue;
      }

      const rawIndex = Math.floor((value - axisRange[0]) / binSize);
      const binIndex = Math.min(HISTOGRAM_BIN_COUNT - 1, Math.max(0, rawIndex));
      bins[binIndex] += 1;
      maxCount = Math.max(maxCount, bins[binIndex]);
    }
  }

  return maxCount === 0 ? 1 : maxCount * 1.05;
}

function ScatterPlotCard({ plot }: { plot: MetricScatter }) {
  const axisRange = getScatterAxisRange(plot.points);

  return (
    <Card>
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
              marker: { color: CHART_COLORS.scatter, size: 8, opacity: 0.75 },
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
                line: { color: CHART_COLORS.referenceLine, width: 1.5, dash: "dash" },
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
}

function getScatterUnit(plot: MetricScatter) {
  const xUnit = metricUnit(plot.x_key);
  const yUnit = metricUnit(plot.y_key);
  return xUnit === yUnit ? xUnit : "other";
}

export default function OverviewPage() {
  const { datasetKey = "" } = useParams();
  const { data, isLoading, error } = useOverview(datasetKey);

  if (isLoading) {
    return <PageLoading label="overview" skeleton={<OverviewSkeleton />} />;
  }

  if (error || !data) {
    return <PageError label="overview" error={error} />;
  }

  return (
    <div className="animate-fade-in-up space-y-6 p-8">
      <div className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight [text-wrap:balance]">
          {data.dataset.label}
        </h1>
        <p className="text-sm text-muted-foreground">
          {data.dataset.dataset_id} · split: {data.dataset.split} · family: {data.dataset.family}
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <StatCard
          icon={<Hash className="h-3.5 w-3.5" />}
          label="Valid Raw Samples"
          value={data.counts.raw_sample_count}
        />
        <StatCard
          icon={<GitBranch className="h-3.5 w-3.5" />}
          label="Derived Tasks"
          value={data.counts.task_count}
        />
        <StatCard
          icon={<AlertTriangle className="h-3.5 w-3.5" />}
          label="Flawed Raw Samples"
          value={data.counts.flawed_count}
        />
      </div>

      {(["chars", "tokens", "lines", "other"] as const).map((unit) => {
        const distributions = [...data.distributions]
          .filter((distribution) => distribution.key !== "test_length_chars")
          .filter((distribution) => metricUnit(distribution.key) === unit)
          .sort(
            (left, right) =>
              (DISTRIBUTION_ORDER[left.key] ?? Number.MAX_SAFE_INTEGER) -
              (DISTRIBUTION_ORDER[right.key] ?? Number.MAX_SAFE_INTEGER),
          );
        const distributionAxisRange = getDistributionAxisRange(distributions);
        const histogramBinSize = getHistogramBinSize(distributionAxisRange);
        const histogramMaxCount = getHistogramMaxCount(
          distributions,
          distributionAxisRange,
          histogramBinSize,
        );
        const scatterPlots = [...data.scatter_plots]
          .filter((plot) => getScatterUnit(plot) === unit)
          .sort(
            (left, right) =>
              (SCATTER_ORDER[left.key] ?? Number.MAX_SAFE_INTEGER) -
              (SCATTER_ORDER[right.key] ?? Number.MAX_SAFE_INTEGER),
          );

        if (!distributions.length && !scatterPlots.length) {
          return null;
        }

        return (
          <section key={unit} className="space-y-3">
            <h2 className="text-lg font-semibold tracking-tight">{UNIT_TITLES[unit]}</h2>
            <div className="grid gap-6 xl:grid-cols-2">
              {distributions.map((distribution) => (
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
                          xbins: distributionAxisRange
                            ? {
                                start: distributionAxisRange[0],
                                end: distributionAxisRange[1],
                                size: histogramBinSize,
                              }
                            : undefined,
                          marker: { color: CHART_COLORS.histogram },
                          hovertemplate: "%{x}<extra></extra>",
                        },
                      ]}
                      layout={{
                        height: 320,
                        bargap: 0.08,
                        xaxis: {
                          title: { text: distribution.label },
                          range: distributionAxisRange,
                        },
                        yaxis: {
                          title: { text: "Count" },
                          range: histogramMaxCount ? [0, histogramMaxCount] : undefined,
                        },
                      }}
                      style={{ height: "320px" }}
                    />
                  </CardContent>
                </Card>
              ))}

              {scatterPlots.map((plot) => (
                <ScatterPlotCard key={plot.key} plot={plot} />
              ))}
            </div>
          </section>
        );
      })}

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
