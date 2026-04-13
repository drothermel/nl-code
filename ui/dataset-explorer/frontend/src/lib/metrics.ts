export const UNIT_TITLES: Record<string, string> = {
  chars: "Character Length Charts",
  tokens: "Token Length Charts",
  lines: "Line Length Charts",
  other: "Other Charts",
};

export const COMPARE_UNIT_TITLES: Record<string, string> = {
  chars: "Character Length Comparisons",
  tokens: "Token Length Comparisons",
  lines: "Line Length Comparisons",
  other: "Other Metric Comparisons",
};

export function metricUnit(metricKey: string): string {
  if (metricKey.endsWith("_length_lines")) return "lines";
  if (metricKey.endsWith("_length_tokens")) return "tokens";
  if (metricKey.endsWith("_length_chars")) return "chars";
  return "other";
}
