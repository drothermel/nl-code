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

import type { Config, Data, Layout } from "plotly.js";
import Plotly from "plotly.js-cartesian-dist-min";
import type { CSSProperties } from "react";
import createPlotlyComponent from "react-plotly.js/factory";
import { CHART_COLORS } from "@/lib/chartColors";
import { cn } from "@/lib/utils";

const ReactPlot = createPlotlyComponent(Plotly);

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
  font: { family: '"DM Sans", system-ui, sans-serif', size: 12 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  xaxis: { gridcolor: CHART_COLORS.grid, zeroline: false },
  yaxis: { gridcolor: CHART_COLORS.grid, zeroline: false },
};

const defaultConfig: Partial<Config> = {
  displayModeBar: "hover",
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
  responsive: true,
};

export default function Plot({ data, layout, config, className, style }: PlotProps) {
  return (
    <div className={cn("w-full", className)} style={style}>
      <ReactPlot
        data={data}
        layout={{ ...defaultLayout, ...layout }}
        config={{ ...defaultConfig, ...config }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
