import { ArrowUpDown, BarChart3 } from "lucide-react";
import { useMemo, useState } from "react";
import { useDatasetComparison } from "@/api/datasets";
import Plot from "@/components/charts/Plot";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { COMPARE_UNIT_TITLES, metricUnit } from "@/lib/metrics";
import type { CrossDatasetSeries, DatasetCompareRow, SummaryStats } from "@/types/datasetExplorer";

const METRIC_COLUMNS = [
  { key: "description_length_chars", label: "Description" },
  { key: "derived_code_length_chars", label: "Derived code" },
  { key: "prompt_length_chars", label: "Prompt/problem" },
  { key: "raw_source_length_chars", label: "Raw source" },
  { key: "test_length_chars", label: "Test" },
  { key: "description_length_tokens", label: "Description tokens" },
  { key: "derived_code_length_tokens", label: "Derived code tokens" },
] as const;

const RATIO_COLUMNS = [
  { key: "description_to_prompt_ratio", label: "Desc / Prompt" },
  { key: "derived_code_to_raw_source_ratio", label: "Code / Raw" },
  { key: "test_to_derived_code_ratio", label: "Test / Code" },
] as const;

const BOX_PLOT_METRIC_KEYS = new Set([
  "description_length_chars",
  "description_length_tokens",
  "derived_code_length_chars",
  "derived_code_length_tokens",
  "derived_code_length_lines",
]);

const SERIES_COLORS = ["#0f766e", "#2563eb", "#b45309", "#9333ea", "#dc2626", "#4f46e5"];

const FAMILY_COLORS: Record<string, string> = {
  humaneval: "#0f766e",
  pro: "#2563eb",
  classeval: "#b45309",
};

type SortKey =
  | "dataset_label"
  | "family"
  | "split"
  | "raw_sample_count"
  | "task_count"
  | "flawed_count"
  | "flawed_rate"
  | (typeof METRIC_COLUMNS)[number]["key"]
  | (typeof RATIO_COLUMNS)[number]["key"];

function getMetricStats(row: DatasetCompareRow, key: string): SummaryStats | null {
  return row.metrics.find((metric) => metric.key === key)?.stats ?? null;
}

function getRatioStats(row: DatasetCompareRow, key: string): SummaryStats | null {
  return row.ratios.find((ratio) => ratio.key === key)?.stats ?? null;
}

const numberFormatCache = new Map<number, Intl.NumberFormat>();

function getNumberFormat(digits: number) {
  let fmt = numberFormatCache.get(digits);
  if (!fmt) {
    fmt = new Intl.NumberFormat(undefined, {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    });
    numberFormatCache.set(digits, fmt);
  }
  return fmt;
}

function formatNumber(value: number, digits = 0) {
  return getNumberFormat(digits).format(value);
}

function formatMetricPair(stats: SummaryStats | null, digits = 0) {
  if (!stats) {
    return "—";
  }
  return `${formatNumber(stats.median, digits)} / ${formatNumber(stats.p90, digits)}`;
}

function formatPercent(value: number) {
  return `${formatNumber(value * 100, 1)}%`;
}

function formatSingleValue(stats: SummaryStats | null, digits = 0) {
  if (!stats) {
    return "—";
  }
  return formatNumber(stats.median, digits);
}

function sortValue(row: DatasetCompareRow, key: SortKey) {
  switch (key) {
    case "dataset_label":
      return row.dataset.label;
    case "family":
      return row.dataset.family;
    case "split":
      return row.dataset.split;
    case "raw_sample_count":
      return row.counts.raw_sample_count;
    case "task_count":
      return row.counts.task_count;
    case "flawed_count":
      return row.counts.flawed_count;
    case "flawed_rate":
      return row.flawed_rate;
    default:
      return (
        getMetricStats(row, key)?.median ??
        getRatioStats(row, key)?.median ??
        Number.NEGATIVE_INFINITY
      );
  }
}

function boxPlotTraces(series: CrossDatasetSeries) {
  return series.datasets.map((dataset, index) => ({
    type: "box" as const,
    name: dataset.dataset_label,
    y: dataset.values,
    boxpoints: false as const,
    marker: { color: SERIES_COLORS[index % SERIES_COLORS.length] },
    hovertemplate: `${dataset.dataset_label}<br>%{y}<extra></extra>`,
  }));
}

function getSharedYAxisRange(seriesList: CrossDatasetSeries[]) {
  const values = seriesList.flatMap((series) =>
    series.datasets.flatMap((dataset) => dataset.values),
  );
  if (!values.length) {
    return undefined;
  }

  const maxValue = Math.max(...values);
  const paddedMax = maxValue === 0 ? 1 : maxValue * 1.05;
  return [0, paddedMax] as [number, number];
}

function getLandscapeAxisRange(
  points: {
    median_prompt_length_chars: number;
    median_derived_code_length_chars: number;
  }[],
) {
  const values = points.flatMap((point) => [
    point.median_prompt_length_chars,
    point.median_derived_code_length_chars,
  ]);
  if (!values.length) {
    return [0, 1] as [number, number];
  }

  const maxValue = Math.max(...values);
  const paddedMax = maxValue === 0 ? 1 : maxValue * 1.15;
  return [0, paddedMax] as [number, number];
}

function SortHeader({
  label,
  sortKey,
  activeKey,
  descending,
  onToggle,
}: {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  descending: boolean;
  onToggle: (key: SortKey) => void;
}) {
  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-auto px-0 py-0 text-xs uppercase tracking-wide text-muted-foreground hover:bg-transparent"
      onClick={() => onToggle(sortKey)}
    >
      <span>{label}</span>
      <ArrowUpDown className="ml-1 h-3.5 w-3.5" />
      {activeKey === sortKey && <span className="ml-1 text-[10px]">{descending ? "↓" : "↑"}</span>}
    </Button>
  );
}

export default function ComparePage() {
  const { data, isLoading, error } = useDatasetComparison();
  const [sortKey, setSortKey] = useState<SortKey>("dataset_label");
  const [descending, setDescending] = useState(false);

  const rows = useMemo(() => {
    if (!data) {
      return [];
    }

    return [...data.datasets].sort((left, right) => {
      const leftValue = sortValue(left, sortKey);
      const rightValue = sortValue(right, sortKey);

      if (typeof leftValue === "string" && typeof rightValue === "string") {
        const comparison = leftValue.localeCompare(rightValue);
        return descending ? -comparison : comparison;
      }

      const comparison = Number(leftValue) - Number(rightValue);
      return descending ? -comparison : comparison;
    });
  }, [data, descending, sortKey]);

  const chartDatasetOrder = useMemo(() => {
    if (!data) {
      return [];
    }

    return [...data.datasets]
      .sort((left, right) => {
        const leftMedian =
          getMetricStats(left, "derived_code_length_chars")?.median ?? Number.NEGATIVE_INFINITY;
        const rightMedian =
          getMetricStats(right, "derived_code_length_chars")?.median ?? Number.NEGATIVE_INFINITY;
        return leftMedian - rightMedian;
      })
      .map((row) => row.dataset.label);
  }, [data]);

  const orderedMetricSeries = useMemo(() => {
    if (!data) {
      return [];
    }

    const order = new Map(chartDatasetOrder.map((label, index) => [label, index]));
    return data.metric_series
      .map((series) => ({
        ...series,
        datasets: [...series.datasets].sort(
          (left, right) =>
            (order.get(left.dataset_label) ?? Number.MAX_SAFE_INTEGER) -
            (order.get(right.dataset_label) ?? Number.MAX_SAFE_INTEGER),
        ),
      }))
      .filter((series) => BOX_PLOT_METRIC_KEYS.has(series.key));
  }, [chartDatasetOrder, data]);

  const groupedMetricSeries = useMemo(() => {
    const groups = new Map<string, CrossDatasetSeries[]>();

    for (const series of orderedMetricSeries) {
      const unit = metricUnit(series.key);
      const grouped = groups.get(unit) ?? [];
      grouped.push(series);
      groups.set(unit, grouped);
    }

    return groups;
  }, [orderedMetricSeries]);

  const sharedMetricYAxisRangesByUnit = useMemo(() => {
    const seriesByUnit = new Map<string, CrossDatasetSeries[]>();

    for (const series of orderedMetricSeries) {
      const unit = metricUnit(series.key);
      const grouped = seriesByUnit.get(unit) ?? [];
      grouped.push(series);
      seriesByUnit.set(unit, grouped);
    }

    return Object.fromEntries(
      [...seriesByUnit.entries()].map(([unit, groupedSeries]) => [
        unit,
        getSharedYAxisRange(groupedSeries),
      ]),
    ) as Record<string, [number, number] | undefined>;
  }, [orderedMetricSeries]);
  const landscapeAxisRange = useMemo(
    () => getLandscapeAxisRange(data?.landscape_points ?? []),
    [data],
  );

  function toggleSort(nextKey: SortKey) {
    if (nextKey === sortKey) {
      setDescending((current) => !current);
      return;
    }

    setSortKey(nextKey);
    setDescending(!["dataset_label", "family", "split"].includes(nextKey));
  }

  if (isLoading) {
    return <div className="p-8 text-sm text-muted-foreground">Loading dataset comparison…</div>;
  }

  if (error || !data) {
    return (
      <div className="p-8 text-sm text-destructive">
        Failed to load dataset comparison: {error?.message ?? "Unknown error"}
      </div>
    );
  }

  return (
    <div className="space-y-6 p-8">
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-primary" />
          <h1 className="text-2xl font-semibold tracking-tight">Compare Datasets</h1>
        </div>
        <p className="max-w-4xl text-sm text-muted-foreground">
          Cross-dataset view of sample counts, length distributions, and compression-style ratios.
          Ratio charts use character counts from valid derived tasks only and skip zero
          denominators. Categorical charts are ordered by median derived code length.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Summary table</CardTitle>
        </CardHeader>
        <CardContent className="overflow-x-auto pt-2">
          <table className="w-full min-w-[1600px] text-sm">
            <thead>
              <tr className="border-b text-left align-bottom">
                <th className="px-3 py-2">
                  <SortHeader
                    label="Dataset"
                    sortKey="dataset_label"
                    activeKey={sortKey}
                    descending={descending}
                    onToggle={toggleSort}
                  />
                </th>
                <th className="px-3 py-2">
                  <SortHeader
                    label="Family"
                    sortKey="family"
                    activeKey={sortKey}
                    descending={descending}
                    onToggle={toggleSort}
                  />
                </th>
                <th className="px-3 py-2">
                  <SortHeader
                    label="Split"
                    sortKey="split"
                    activeKey={sortKey}
                    descending={descending}
                    onToggle={toggleSort}
                  />
                </th>
                <th className="px-3 py-2">
                  <SortHeader
                    label="Valid Raw"
                    sortKey="raw_sample_count"
                    activeKey={sortKey}
                    descending={descending}
                    onToggle={toggleSort}
                  />
                </th>
                <th className="px-3 py-2">
                  <SortHeader
                    label="Tasks"
                    sortKey="task_count"
                    activeKey={sortKey}
                    descending={descending}
                    onToggle={toggleSort}
                  />
                </th>
                <th className="px-3 py-2">
                  <SortHeader
                    label="Flawed"
                    sortKey="flawed_count"
                    activeKey={sortKey}
                    descending={descending}
                    onToggle={toggleSort}
                  />
                </th>
                <th className="px-3 py-2">
                  <SortHeader
                    label="Flawed Rate"
                    sortKey="flawed_rate"
                    activeKey={sortKey}
                    descending={descending}
                    onToggle={toggleSort}
                  />
                </th>
                {METRIC_COLUMNS.map((column) => (
                  <th key={column.key} className="px-3 py-2">
                    <SortHeader
                      label={`${column.label} Med / P90`}
                      sortKey={column.key}
                      activeKey={sortKey}
                      descending={descending}
                      onToggle={toggleSort}
                    />
                  </th>
                ))}
                {RATIO_COLUMNS.map((column) => (
                  <th key={column.key} className="px-3 py-2">
                    <SortHeader
                      label={`${column.label} Median`}
                      sortKey={column.key}
                      activeKey={sortKey}
                      descending={descending}
                      onToggle={toggleSort}
                    />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.dataset.key} className="border-b align-top">
                  <td className="px-3 py-3">
                    <div className="font-medium">{row.dataset.label}</div>
                    <div className="font-mono text-xs text-muted-foreground">{row.dataset.key}</div>
                  </td>
                  <td className="px-3 py-3 text-xs text-muted-foreground">{row.dataset.family}</td>
                  <td className="px-3 py-3 text-xs text-muted-foreground">{row.dataset.split}</td>
                  <td className="px-3 py-3 tabular-nums">
                    {formatNumber(row.counts.raw_sample_count)}
                  </td>
                  <td className="px-3 py-3 tabular-nums">{formatNumber(row.counts.task_count)}</td>
                  <td className="px-3 py-3 tabular-nums">
                    {formatNumber(row.counts.flawed_count)}
                  </td>
                  <td className="px-3 py-3 tabular-nums">{formatPercent(row.flawed_rate)}</td>
                  {METRIC_COLUMNS.map((column) => (
                    <td key={column.key} className="px-3 py-3 tabular-nums">
                      {formatMetricPair(getMetricStats(row, column.key))}
                    </td>
                  ))}
                  {RATIO_COLUMNS.map((column) => (
                    <td key={column.key} className="px-3 py-3 tabular-nums">
                      {formatSingleValue(getRatioStats(row, column.key), 2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      {(["chars", "tokens", "lines", "other"] as const).map((unit) => {
        const seriesForUnit = groupedMetricSeries.get(unit) ?? [];
        if (!seriesForUnit.length) {
          return null;
        }

        return (
          <section key={unit} className="space-y-3">
            <h2 className="text-lg font-semibold tracking-tight">{COMPARE_UNIT_TITLES[unit]}</h2>
            <div className="grid gap-6 xl:grid-cols-2">
              {seriesForUnit.map((series) => (
                <Card key={series.key}>
                  <CardHeader>
                    <CardTitle className="text-base">{series.label}</CardTitle>
                  </CardHeader>
                  <CardContent className="pt-2">
                    <Plot
                      data={boxPlotTraces(series)}
                      layout={{
                        height: 520,
                        showlegend: false,
                        xaxis: {
                          title: { text: "Dataset" },
                          categoryorder: "array",
                          categoryarray: chartDatasetOrder,
                        },
                        yaxis: {
                          title: { text: series.label },
                          range: sharedMetricYAxisRangesByUnit[metricUnit(series.key)],
                        },
                      }}
                      className="aspect-square"
                      style={{ height: "100%" }}
                    />
                  </CardContent>
                </Card>
              ))}
            </div>
          </section>
        );
      })}

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Flawed samples by dataset</CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <Plot
              data={[
                {
                  type: "bar",
                  x: chartDatasetOrder,
                  y: chartDatasetOrder.map((label) => {
                    const row = data.datasets.find((dataset) => dataset.dataset.label === label);
                    return row?.counts.flawed_count ?? 0;
                  }),
                  text: chartDatasetOrder.map((label) => {
                    const row = data.datasets.find((dataset) => dataset.dataset.label === label);
                    return `${formatPercent(row?.flawed_rate ?? 0)}`;
                  }),
                  textposition: "outside",
                  marker: { color: "#dc2626" },
                  cliponaxis: false,
                  hovertemplate:
                    "%{x}<br>Flawed samples=%{y}<br>Flawed rate=%{text}<extra></extra>",
                },
              ]}
              layout={{
                height: 520,
                margin: { l: 70, r: 24, t: 40, b: 70 },
                xaxis: {
                  title: { text: "Dataset" },
                  categoryorder: "array",
                  categoryarray: chartDatasetOrder,
                },
                yaxis: { title: { text: "Flawed samples" } },
              }}
              className="aspect-square"
              style={{ height: "100%" }}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Dataset landscape</CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <Plot
              data={[
                {
                  type: "scatter",
                  mode: "text+markers",
                  x: data.landscape_points.map((point) => point.median_prompt_length_chars),
                  y: data.landscape_points.map((point) => point.median_derived_code_length_chars),
                  text: data.landscape_points.map((point) => point.dataset_label),
                  textposition: "top center",
                  cliponaxis: false,
                  marker: {
                    size: data.landscape_points.map((point) =>
                      Math.max(12, Math.sqrt(point.task_count) * 2.5),
                    ),
                    color: data.landscape_points.map(
                      (point) => FAMILY_COLORS[point.family] ?? "#4b5563",
                    ),
                    opacity: 0.8,
                  },
                  customdata: data.landscape_points.map((point) => [
                    point.family,
                    point.task_count,
                  ]),
                  hovertemplate:
                    "%{text}<br>Family=%{customdata[0]}<br>Tasks=%{customdata[1]}<br>Prompt median=%{x}<br>Code median=%{y}<extra></extra>",
                },
              ]}
              layout={{
                height: 520,
                margin: { l: 70, r: 24, t: 70, b: 70 },
                shapes: [
                  {
                    type: "line",
                    x0: landscapeAxisRange[0],
                    y0: landscapeAxisRange[0],
                    x1: landscapeAxisRange[1],
                    y1: landscapeAxisRange[1],
                    line: { color: "#dc2626", width: 1.5, dash: "dash" },
                  },
                ],
                xaxis: {
                  title: { text: "Median prompt/problem length (chars)" },
                  range: landscapeAxisRange,
                },
                yaxis: {
                  title: { text: "Median derived code length (chars)" },
                  range: landscapeAxisRange,
                  scaleanchor: "x",
                  scaleratio: 1,
                },
              }}
              className="aspect-square"
              style={{ height: "100%" }}
            />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
