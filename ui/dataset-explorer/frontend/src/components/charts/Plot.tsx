/**
 * Thin wrapper around react-plotly.js with sensible defaults for research dashboards.
 *
 * Usage:
 *   <Plot
 *     data={[{ type: 'scatter', mode: 'markers', x: xs, y: ys }]}
 *     layout={{ title: 'My Chart' }}
 *   />
 *
 * All Plotly trace types work: scatter, heatmap, bar, histogram, violin, box, etc.
 * Extend `defaultLayout` below to change global defaults (font, margins, colors).
 */

import ReactPlotly from "react-plotly.js";
import type { Data, Layout, Config } from "plotly.js";
import type { CSSProperties } from "react";
import { cn } from "@/lib/utils";

interface PlotProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
  style?: CSSProperties;
}

const defaultLayout: Partial<Layout> = {
  autosize: true,
  margin: { l: 60, r: 24, t: 40, b: 60 },
  font: { family: "Inter, system-ui, sans-serif", size: 12 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: { gridcolor: "#e5e7eb", zeroline: false },
  yaxis: { gridcolor: "#e5e7eb", zeroline: false },
};

const defaultConfig: Partial<Config> = {
  displayModeBar: "hover",
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
  responsive: true,
};

export default function Plot({
  data,
  layout,
  config,
  className,
  style,
}: PlotProps) {
  return (
    <div className={cn("w-full", className)} style={style}>
      <ReactPlotly
        data={data}
        layout={{ ...defaultLayout, ...layout }}
        config={{ ...defaultConfig, ...config }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
