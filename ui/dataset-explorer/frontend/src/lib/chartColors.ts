/** Centralized chart color constants for Plotly visualizations. */

export const CHART_COLORS = {
  scatter: "#2b8a8a",
  histogram: "#0f766e",
  referenceLine: "#e04040",
  grid: "#dde4e8",
  fallbackFamily: "#64748b",
} as const;

export const SERIES_PALETTE = [
  "#0f766e",
  "#2563eb",
  "#b45309",
  "#9333ea",
  "#dc2626",
  "#4f46e5",
] as const;

export const FAMILY_COLORS: Record<string, string> = {
  humaneval: "#0f766e",
  pro: "#2563eb",
  classeval: "#b45309",
};
