from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import matplotlib
import typer
from pydantic import BaseModel, ConfigDict


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402


DEFAULT_FULL5X_PATH = Path(
    "ui/dspy-eval-static-viewer/exports/task_variation_full5x.csv"
)
DEFAULT_OUTPUT_DIR = Path("figures/humaneval-encdec-prompt-optimization-eval-only")

BASELINE_COLUMN = "baseline_encdec_pass_rate"
TARGET_ORDER = ("encoder", "decoder")
METHOD_ORDER = ("MIPRO", "GEPA")
PASS_RATE_COLUMNS = {
    ("MIPRO", "encoder"): "mipro_encdec_encoder_pass_rate",
    ("MIPRO", "decoder"): "mipro_encdec_decoder_pass_rate",
    ("GEPA", "encoder"): "gepa_encdec_encoder_pass_rate",
    ("GEPA", "decoder"): "gepa_encdec_decoder_pass_rate",
}
METHOD_COLORS = {
    "MIPRO": "#1F6F8B",
    "GEPA": "#B85632",
}
TARGET_COLORS = {
    "encoder": "#1F6F8B",
    "decoder": "#6C7A89",
}
ZERO_COLOR = "#2D2D2D"
GRID_COLOR = "#D8DEE4"
TEXT_COLOR = "#202124"
MUTED_TEXT_COLOR = "#5F6368"


class EvalTaskScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    method: str
    target: str
    baseline: float
    optimized: float

    @property
    def delta(self) -> float:
        return self.optimized - self.baseline


class EvalSummaryCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    target: str
    baseline: float
    optimized: float
    delta: float
    n: int
    improved: int
    worsened: int
    unchanged: int


def main(
    full5x_path: Annotated[
        Path,
        typer.Option(
            "--full5x-path",
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Full-5x task variation CSV from the static viewer exports.",
        ),
    ] = DEFAULT_FULL5X_PATH,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            file_okay=False,
            dir_okay=True,
            help="Directory where PNG and SVG plots will be written.",
        ),
    ] = DEFAULT_OUTPUT_DIR,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = load_full5x_eval_scores(full5x_path)
    summary_cells = summarize_scores(scores)

    print_summary_table(summary_cells)

    written_paths: list[Path] = []
    written_paths.extend(plot_mipro_target_delta_bars(summary_cells, output_dir))
    written_paths.extend(plot_mipro_target_dumbbell(summary_cells, output_dir))
    written_paths.extend(plot_mipro_target_delta_distribution(scores, output_dir))
    written_paths.extend(plot_encoder_method_bars(summary_cells, output_dir))
    written_paths.extend(plot_method_target_heatmap(summary_cells, output_dir))

    typer.echo("\nWrote eval-only plots:")
    for path in written_paths:
        typer.echo(f"- {path}")


def load_full5x_eval_scores(path: Path) -> list[EvalTaskScore]:
    scores = []
    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            baseline = float(row[BASELINE_COLUMN])
            for (method, target), column in PASS_RATE_COLUMNS.items():
                scores.append(
                    EvalTaskScore(
                        task_id=row["task_id"],
                        method=method,
                        target=target,
                        baseline=baseline,
                        optimized=float(row[column]),
                    )
                )
    return scores


def summarize_scores(scores: list[EvalTaskScore]) -> list[EvalSummaryCell]:
    summaries = []
    for method in METHOD_ORDER:
        for target in TARGET_ORDER:
            matching_scores = [
                score
                for score in scores
                if score.method == method and score.target == target
            ]
            deltas = [score.delta for score in matching_scores]
            summaries.append(
                EvalSummaryCell(
                    method=method,
                    target=target,
                    baseline=mean(score.baseline for score in matching_scores),
                    optimized=mean(score.optimized for score in matching_scores),
                    delta=mean(deltas),
                    n=len(matching_scores),
                    improved=sum(delta > 0 for delta in deltas),
                    worsened=sum(delta < 0 for delta in deltas),
                    unchanged=sum(delta == 0 for delta in deltas),
                )
            )
    return summaries


def plot_mipro_target_delta_bars(
    summary_cells: list[EvalSummaryCell], output_dir: Path
) -> list[Path]:
    cells_by_key = cell_lookup(summary_cells)
    values = [cells_by_key[("MIPRO", target)].delta for target in TARGET_ORDER]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bars = ax.bar(
        [target.title() for target in TARGET_ORDER],
        values,
        color=[TARGET_COLORS[target] for target in TARGET_ORDER],
        width=0.55,
    )
    for bar, value in zip(bars, values, strict=True):
        annotate_bar(ax, bar, f"{value:+.3f}")

    ax.axhline(0, color=ZERO_COLOR, linewidth=1)
    ax.set_title("Eval-Only: Encoder Optimization Is The Lever", loc="left", pad=14)
    ax.set_ylabel("Saved prompt - baseline pass rate")
    ax.set_ylim(0, 0.07)
    style_axis(ax)
    ax.text(
        0,
        -0.18,
        "Full-5x saved-prompt eval export. Both-stage enc-dec is not available in this eval set.",
        transform=ax.transAxes,
        color=MUTED_TEXT_COLOR,
        fontsize=9,
    )
    return save_figure(fig, output_dir / "eval_only_mipro_target_delta_bars")


def plot_mipro_target_dumbbell(
    summary_cells: list[EvalSummaryCell], output_dir: Path
) -> list[Path]:
    cells = [cell_lookup(summary_cells)[("MIPRO", target)] for target in TARGET_ORDER]
    y_positions = list(reversed(range(len(cells))))

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for y_position, cell in zip(y_positions, cells, strict=True):
        ax.plot(
            [cell.baseline, cell.optimized],
            [y_position, y_position],
            color=GRID_COLOR,
            linewidth=5,
            solid_capstyle="round",
        )
        ax.scatter(cell.baseline, y_position, s=80, color="#8A8F98", label=None)
        ax.scatter(
            cell.optimized,
            y_position,
            s=90,
            color=TARGET_COLORS[cell.target],
            label=None,
            zorder=3,
        )
        midpoint = (cell.baseline + cell.optimized) / 2
        ax.text(
            midpoint,
            y_position + 0.16,
            f"{cell.delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_COLOR,
        )

    ax.scatter([], [], s=80, color="#8A8F98", label="Baseline")
    ax.scatter([], [], s=90, color="#1F6F8B", label="Saved prompt")
    ax.set_title("Eval-Only: MIPRO Encoder Moves Pass Rate Most", loc="left", pad=14)
    ax.set_xlabel("Mean pass rate")
    ax.set_yticks(y_positions, [cell.target.title() for cell in cells])
    ax.set_xlim(0.84, 0.93)
    ax.legend(frameon=False, ncols=2, loc="upper left", bbox_to_anchor=(0, 1.03))
    style_axis(ax)
    ax.text(
        0,
        -0.18,
        "Full-5x saved-prompt eval export.",
        transform=ax.transAxes,
        color=MUTED_TEXT_COLOR,
        fontsize=9,
    )
    return save_figure(fig, output_dir / "eval_only_mipro_target_dumbbell")


def plot_mipro_target_delta_distribution(
    scores: list[EvalTaskScore], output_dir: Path
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    for index, target in enumerate(TARGET_ORDER):
        deltas = sorted(
            score.delta
            for score in scores
            if score.method == "MIPRO" and score.target == target
        )
        jittered_x = [
            index + deterministic_jitter(score_index, len(deltas), width=0.18)
            for score_index, _delta in enumerate(deltas)
        ]
        ax.scatter(
            jittered_x,
            deltas,
            s=24,
            alpha=0.75,
            color=TARGET_COLORS[target],
            edgecolors="white",
            linewidths=0.35,
        )
        ax.scatter(
            [index],
            [mean(deltas)],
            s=130,
            color="white",
            edgecolors=ZERO_COLOR,
            linewidths=1.4,
            zorder=4,
        )
        ax.text(index, mean(deltas) + 0.055, f"mean {mean(deltas):+.3f}", ha="center")

    ax.axhline(0, color=ZERO_COLOR, linewidth=1)
    ax.set_title("Eval-Only: MIPRO Gains Are Task-Localized", loc="left", pad=14)
    ax.set_ylabel("Saved prompt - baseline pass rate")
    ax.set_xticks(range(len(TARGET_ORDER)), [target.title() for target in TARGET_ORDER])
    ax.set_ylim(-1.05, 1.05)
    style_axis(ax)
    ax.text(
        0,
        -0.18,
        "Each point is one HumanEval task in the full-5x saved-prompt export; outlined dot is the mean.",
        transform=ax.transAxes,
        color=MUTED_TEXT_COLOR,
        fontsize=9,
    )
    return save_figure(fig, output_dir / "eval_only_mipro_target_delta_distribution")


def plot_encoder_method_bars(
    summary_cells: list[EvalSummaryCell], output_dir: Path
) -> list[Path]:
    cells_by_key = cell_lookup(summary_cells)
    values = [cells_by_key[(method, "encoder")].delta for method in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bars = ax.bar(
        METHOD_ORDER,
        values,
        color=[METHOD_COLORS[method] for method in METHOD_ORDER],
        width=0.55,
    )
    for bar, value in zip(bars, values, strict=True):
        annotate_bar(ax, bar, f"{value:+.3f}")

    ax.axhline(0, color=ZERO_COLOR, linewidth=1)
    ax.set_title("Eval-Only: MIPRO Has The Larger Encoder Gain", loc="left", pad=14)
    ax.set_ylabel("Encoder saved prompt - baseline pass rate")
    ax.set_ylim(0, 0.07)
    style_axis(ax)
    ax.text(
        0,
        -0.18,
        "Full-5x saved-prompt eval export.",
        transform=ax.transAxes,
        color=MUTED_TEXT_COLOR,
        fontsize=9,
    )
    return save_figure(fig, output_dir / "eval_only_encoder_method_bars")


def plot_method_target_heatmap(
    summary_cells: list[EvalSummaryCell], output_dir: Path
) -> list[Path]:
    cells_by_key = cell_lookup(summary_cells)
    matrix = [
        [cells_by_key[(method, target)].delta for target in TARGET_ORDER]
        for method in METHOD_ORDER
    ]

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    cmap = LinearSegmentedColormap.from_list(
        "delta",
        ["#F6F0E6", "#86A9B4", "#1F6F8B"],
    )
    norm = TwoSlopeNorm(vmin=0, vcenter=0.03, vmax=0.06)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    for row_index, method in enumerate(METHOD_ORDER):
        for col_index, target in enumerate(TARGET_ORDER):
            cell = cells_by_key[(method, target)]
            ax.text(
                col_index,
                row_index,
                f"{cell.delta:+.3f}\nn={cell.n}",
                ha="center",
                va="center",
                color=TEXT_COLOR,
                fontsize=10,
            )

    ax.set_title(
        "Eval-Only: Encoder Beats Decoder For Both Optimizers", loc="left", pad=14
    )
    ax.set_xticks(range(len(TARGET_ORDER)), [target.title() for target in TARGET_ORDER])
    ax.set_yticks(range(len(METHOD_ORDER)), METHOD_ORDER)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Saved prompt - baseline pass rate")
    ax.text(
        0,
        -0.18,
        "Full-5x saved-prompt eval export.",
        transform=ax.transAxes,
        color=MUTED_TEXT_COLOR,
        fontsize=9,
    )
    return save_figure(fig, output_dir / "eval_only_method_target_heatmap")


def cell_lookup(
    summary_cells: list[EvalSummaryCell],
) -> dict[tuple[str, str], EvalSummaryCell]:
    return {(cell.method, cell.target): cell for cell in summary_cells}


def deterministic_jitter(index: int, count: int, *, width: float) -> float:
    if count <= 1:
        return 0
    position = index / (count - 1)
    return (position - 0.5) * width


def annotate_bar(axes: plt.Axes, bar: plt.Rectangle, label: str) -> None:
    value = bar.get_height()
    y_offset = 0.004 if value >= 0 else -0.006
    va = "bottom" if value >= 0 else "top"
    axes.text(
        bar.get_x() + bar.get_width() / 2,
        value + y_offset,
        label,
        ha="center",
        va=va,
        fontsize=10,
        color=TEXT_COLOR,
    )


def style_axis(axes: plt.Axes) -> None:
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["left"].set_color(GRID_COLOR)
    axes.spines["bottom"].set_color(GRID_COLOR)
    axes.tick_params(axis="both", colors=TEXT_COLOR)
    axes.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8)
    axes.set_axisbelow(True)


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    fig.tight_layout()
    paths = [stem.with_suffix(".png"), stem.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return paths


def mean(values: Iterable[float]) -> float:
    concrete_values = list(values)
    if not concrete_values:
        raise ValueError("cannot compute mean of empty values")
    return sum(concrete_values) / len(concrete_values)


def print_summary_table(summary_cells: list[EvalSummaryCell]) -> None:
    typer.echo("Full-5x saved-prompt eval deltas:")
    for target in TARGET_ORDER:
        for method in METHOD_ORDER:
            cell = cell_lookup(summary_cells)[(method, target)]
            typer.echo(
                f"- {method:5s} {target:7s}: "
                f"baseline={cell.baseline:.3f} saved_prompt={cell.optimized:.3f} "
                f"delta={cell.delta:+.3f} n={cell.n} "
                f"improved={cell.improved} worsened={cell.worsened}"
            )


if __name__ == "__main__":
    typer.run(main)
