import { useMemo, useState } from "react";
import { ArrowUpDown, BarChart3 } from "lucide-react";
import { useDatasetComparison } from "@/api/datasets";
import Plot from "@/components/charts/Plot";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type {
  CrossDatasetSeries,
  DatasetCompareRow,
  SummaryStats,
} from "@/types/datasetExplorer";

const METRIC_COLUMNS = [
  { key: "description_length_chars", label: "Description" },
  { key: "derived_code_length_chars", label: "Derived code" },
  { key: "prompt_length_chars", label: "Prompt/problem" },
  { key: "raw_source_length_chars", label: "Raw source" },
  { key: "test_length_chars", label: "Test" },
] as const;

const RATIO_COLUMNS = [
  { key: "description_to_prompt_ratio", label: "Desc / Prompt" },
  { key: "derived_code_to_raw_source_ratio", label: "Code / Raw" },
  { key: "test_to_derived_code_ratio", label: "Test / Code" },
] as const;

const SERIES_COLORS = [
  "#0f766e",
  "#2563eb",
  "#b45309",
  "#9333ea",
  "#dc2626",
  "#4f46e5",
];

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

function formatNumber(value: number, digits = 0) {
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
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
      return getMetricStats(row, key)?.median ?? getRatioStats(row, key)?.median ?? Number.NEGATIVE_INFINITY;
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

  function toggleSort(nextKey: SortKey) {
    if (nextKey === sortKey) {
      setDescending((current) => !current);
      return;
    }

    setSortKey(nextKey);
    setDescending(
      !["dataset_label", "family", "split"].includes(nextKey)
    );
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
          Cross-dataset view of sample counts, length distributions, and
          compression-style ratios. Ratio charts use character counts from valid
          derived tasks only and skip zero denominators.
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
                <th className="px-3 py-2"><SortHeader label="Dataset" sortKey="dataset_label" activeKey={sortKey} descending={descending} onToggle={toggleSort} /></th>
                <th className="px-3 py-2"><SortHeader label="Family" sortKey="family" activeKey={sortKey} descending={descending} onToggle={toggleSort} /></th>
                <th className="px-3 py-2"><SortHeader label="Split" sortKey="split" activeKey={sortKey} descending={descending} onToggle={toggleSort} /></th>
                <th className="px-3 py-2"><SortHeader label="Valid Raw" sortKey="raw_sample_count" activeKey={sortKey} descending={descending} onToggle={toggleSort} /></th>
                <th className="px-3 py-2"><SortHeader label="Tasks" sortKey="task_count" activeKey={sortKey} descending={descending} onToggle={toggleSort} /></th>
                <th className="px-3 py-2"><SortHeader label="Flawed" sortKey="flawed_count" activeKey={sortKey} descending={descending} onToggle={toggleSort} /></th>
                <th className="px-3 py-2"><SortHeader label="Flawed Rate" sortKey="flawed_rate" activeKey={sortKey} descending={descending} onToggle={toggleSort} /></th>
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
                  <td className="px-3 py-3 tabular-nums">{formatNumber(row.counts.raw_sample_count)}</td>
                  <td className="px-3 py-3 tabular-nums">{formatNumber(row.counts.task_count)}</td>
                  <td className="px-3 py-3 tabular-nums">{formatNumber(row.counts.flawed_count)}</td>
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

      <div className="grid gap-6 xl:grid-cols-2">
        {data.metric_series.map((series) => (
          <Card key={series.key}>
            <CardHeader>
              <CardTitle className="text-base">{series.label}</CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <Plot
                data={boxPlotTraces(series)}
                layout={{
                  height: 340,
                  showlegend: false,
                  xaxis: { title: { text: "Dataset" } },
                  yaxis: { title: { text: series.label } },
                }}
                style={{ height: "340px" }}
              />
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Flawed rate by dataset</CardTitle>
          </CardHeader>
          <CardContent className="pt-2">
            <Plot
              data={[
                {
                  type: "bar",
                  x: rows.map((row) => row.dataset.label),
                  y: rows.map((row) => row.flawed_rate * 100),
                  marker: { color: "#dc2626" },
                  hovertemplate: "%{x}<br>%{y:.1f}% flawed<extra></extra>",
                },
              ]}
              layout={{
                height: 320,
                xaxis: { title: { text: "Dataset" } },
                yaxis: { title: { text: "Flawed rate (%)" } },
              }}
              style={{ height: "320px" }}
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
                  marker: {
                    size: data.landscape_points.map((point) => Math.max(12, Math.sqrt(point.task_count) * 2.5)),
                    color: data.landscape_points.map((point) => FAMILY_COLORS[point.family] ?? "#4b5563"),
                    opacity: 0.8,
                  },
                  customdata: data.landscape_points.map((point) => [point.family, point.task_count]),
                  hovertemplate:
                    "%{text}<br>Family=%{customdata[0]}<br>Tasks=%{customdata[1]}<br>Prompt median=%{x}<br>Code median=%{y}<extra></extra>",
                },
              ]}
              layout={{
                height: 320,
                xaxis: { title: { text: "Median prompt/problem length (chars)" } },
                yaxis: { title: { text: "Median derived code length (chars)" } },
              }}
              style={{ height: "320px" }}
            />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        {data.ratio_series.map((series) => (
          <Card key={series.key}>
            <CardHeader>
              <CardTitle className="text-base">{series.label}</CardTitle>
            </CardHeader>
            <CardContent className="pt-2">
              <Plot
                data={boxPlotTraces(series)}
                layout={{
                  height: 320,
                  showlegend: false,
                  xaxis: { title: { text: "Dataset" } },
                  yaxis: { title: { text: series.label } },
                }}
                style={{ height: "320px" }}
              />
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
