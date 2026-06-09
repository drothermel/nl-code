from __future__ import annotations

import csv
from collections.abc import Iterable
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import matplotlib
import typer
from pydantic import BaseModel, ConfigDict


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402


GPT5_NANO_MODEL_NAME = "openrouter/openai/gpt-5-nano"
ENCDEC_FAMILY = "enc-dec"
OPTIMIZER_SUMMARY_SOURCE_KIND = "optimizer_summary_task_score"
BASELINE_PHASE = "baseline"
OPTIMIZED_PHASE = "optimized"
MIPRO_GENERATION_TYPE = "encdec"
GEPA_GENERATION_TYPE = "encdec_gepa"

DEFAULT_DATA_DIR = Path("data/humaneval-dspy-sample-performance")
DEFAULT_FULL5X_PATH = Path(
    "ui/dspy-eval-static-viewer/exports/task_variation_full5x.csv"
)
DEFAULT_OUTPUT_DIR = Path("figures/humaneval-encdec-prompt-optimization")

TARGET_ORDER = ("encoder", "decoder", "both")
METHOD_ORDER = ("MIPRO", "GEPA")
METHOD_COLORS = {
    "MIPRO": "#1F6F8B",
    "GEPA": "#B85632",
}
TARGET_COLORS = {
    "encoder": "#1F6F8B",
    "decoder": "#6C7A89",
    "both": "#B85632",
}
ZERO_COLOR = "#2D2D2D"
GRID_COLOR = "#D8DEE4"
TEXT_COLOR = "#202124"
MUTED_TEXT_COLOR = "#5F6368"


class PairedScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    target: str
    sample_id: str
    baseline: float
    optimized: float

    @property
    def delta(self) -> float:
        return self.optimized - self.baseline


class SummaryCell(BaseModel):
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


class Full5xEncoderDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: str
    delta: float
    n: int
    improved: int
    worsened: int
    unchanged: int


def main(
    data_dir: Annotated[
        Path,
        typer.Option(
            "--data-dir",
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Directory containing HumanEval DSPy sample-performance exports.",
        ),
    ] = DEFAULT_DATA_DIR,
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

    paired_scores = load_optimizer_summary_pairs(data_dir)
    summary_cells = summarize_pairs(paired_scores)
    full5x_encoder_deltas = load_full5x_encoder_deltas(full5x_path)

    print_summary_table(summary_cells)
    print_full5x_table(full5x_encoder_deltas)

    written_paths: list[Path] = []
    written_paths.extend(plot_target_delta_bars(summary_cells, output_dir))
    written_paths.extend(plot_mipro_target_dumbbell(summary_cells, output_dir))
    written_paths.extend(
        plot_mipro_target_delta_distribution(paired_scores, output_dir)
    )
    written_paths.extend(plot_encoder_method_bars(summary_cells, output_dir))
    written_paths.extend(plot_method_target_heatmap(summary_cells, output_dir))
    written_paths.extend(
        plot_encoder_evidence_source_comparison(
            summary_cells,
            full5x_encoder_deltas,
            output_dir,
        )
    )

    typer.echo("\nWrote plots:")
    for path in written_paths:
        typer.echo(f"- {path}")


def load_optimizer_summary_pairs(data_dir: Path) -> list[PairedScore]:
    evidence_path = data_dir / "humaneval_sample_performance_evidence.csv"
    row_values: dict[tuple[str, str, str, str, str, str], dict[str, float]] = (
        defaultdict(dict)
    )

    with evidence_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["source kind"] != OPTIMIZER_SUMMARY_SOURCE_KIND:
                continue
            if row["model name"] != GPT5_NANO_MODEL_NAME:
                continue
            if row["direct or enc-dec"] != ENCDEC_FAMILY:
                continue
            if row["phase"] not in {BASELINE_PHASE, OPTIMIZED_PHASE}:
                continue

            method = method_from_generation_type(row["generation type"])
            if method is None:
                continue

            key = (
                method,
                row["optimization target"],
                row["sample id"],
                row["session id"],
                row["run id"],
                row["split"],
            )
            row_values[key][row["phase"]] = float(row["pass rate"])

    paired_scores = []
    for key, values in row_values.items():
        if BASELINE_PHASE not in values or OPTIMIZED_PHASE not in values:
            continue
        method, target, sample_id, _session_id, _run_id, _split = key
        paired_scores.append(
            PairedScore(
                method=method,
                target=target,
                sample_id=sample_id,
                baseline=values[BASELINE_PHASE],
                optimized=values[OPTIMIZED_PHASE],
            )
        )
    return paired_scores


def method_from_generation_type(generation_type: str) -> str | None:
    if generation_type == MIPRO_GENERATION_TYPE:
        return "MIPRO"
    if generation_type == GEPA_GENERATION_TYPE:
        return "GEPA"
    return None


def summarize_pairs(paired_scores: list[PairedScore]) -> list[SummaryCell]:
    grouped: dict[tuple[str, str], list[PairedScore]] = defaultdict(list)
    for score in paired_scores:
        grouped[(score.method, score.target)].append(score)

    summaries = []
    for (method, target), scores in sorted(grouped.items()):
        deltas = [score.delta for score in scores]
        summaries.append(
            SummaryCell(
                method=method,
                target=target,
                baseline=mean(score.baseline for score in scores),
                optimized=mean(score.optimized for score in scores),
                delta=mean(deltas),
                n=len(scores),
                improved=sum(delta > 0 for delta in deltas),
                worsened=sum(delta < 0 for delta in deltas),
                unchanged=sum(delta == 0 for delta in deltas),
            )
        )
    return summaries


def load_full5x_encoder_deltas(path: Path) -> list[Full5xEncoderDelta]:
    column_by_method = {
        "MIPRO": "mipro_encdec_encoder_pass_rate",
        "GEPA": "gepa_encdec_encoder_pass_rate",
    }
    deltas_by_method: dict[str, list[float]] = defaultdict(list)

    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            baseline = float(row["baseline_encdec_pass_rate"])
            for method, column in column_by_method.items():
                deltas_by_method[method].append(float(row[column]) - baseline)

    return [
        Full5xEncoderDelta(
            method=method,
            delta=mean(deltas),
            n=len(deltas),
            improved=sum(delta > 0 for delta in deltas),
            worsened=sum(delta < 0 for delta in deltas),
            unchanged=sum(delta == 0 for delta in deltas),
        )
        for method, deltas in deltas_by_method.items()
    ]


def plot_target_delta_bars(
    summary_cells: list[SummaryCell], output_dir: Path
) -> list[Path]:
    cells_by_key = cell_lookup(summary_cells)
    values = [cells_by_key[("MIPRO", target)].delta for target in TARGET_ORDER]

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bars = ax.bar(
        [target.title() for target in TARGET_ORDER],
        values,
        color=[TARGET_COLORS[target] for target in TARGET_ORDER],
        width=0.58,
    )
    for bar, value in zip(bars, values, strict=True):
        annotate_bar(ax, bar, f"{value:+.3f}")

    ax.axhline(0, color=ZERO_COLOR, linewidth=1)
    ax.set_title("Encoder Optimization Is The Real Lever", loc="left", pad=14)
    ax.set_ylabel("Optimized - baseline pass rate")
    ax.set_ylim(-0.105, 0.09)
    style_axis(ax)
    ax.text(
        0,
        -0.18,
        "GPT-5 nano enc-dec MIPRO optimizer-summary task scores.",
        transform=ax.transAxes,
        color=MUTED_TEXT_COLOR,
        fontsize=9,
    )
    return save_figure(fig, output_dir / "target_delta_bars")


def plot_mipro_target_dumbbell(
    summary_cells: list[SummaryCell], output_dir: Path
) -> list[Path]:
    cells = [cell_lookup(summary_cells)[("MIPRO", target)] for target in TARGET_ORDER]
    y_positions = list(reversed(range(len(cells))))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
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
            y_position + 0.18,
            f"{cell.delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TEXT_COLOR,
        )

    ax.scatter([], [], s=80, color="#8A8F98", label="Baseline")
    ax.scatter([], [], s=90, color="#1F6F8B", label="Optimized")
    ax.set_title("MIPRO Helps When It Tunes The Encoder", loc="left", pad=14)
    ax.set_xlabel("Mean pass rate")
    ax.set_yticks(y_positions, [cell.target.title() for cell in cells])
    ax.set_xlim(0.84, 0.99)
    ax.legend(frameon=False, ncols=2, loc="upper left", bbox_to_anchor=(0, 1.03))
    style_axis(ax)
    ax.text(
        0,
        -0.18,
        "GPT-5 nano enc-dec MIPRO optimizer-summary task scores.",
        transform=ax.transAxes,
        color=MUTED_TEXT_COLOR,
        fontsize=9,
    )
    return save_figure(fig, output_dir / "mipro_target_dumbbell")


def plot_mipro_target_delta_distribution(
    paired_scores: list[PairedScore], output_dir: Path
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    target_to_deltas = {
        target: [
            score.delta
            for score in paired_scores
            if score.method == "MIPRO" and score.target == target
        ]
        for target in TARGET_ORDER
    }

    for index, target in enumerate(TARGET_ORDER):
        deltas = sorted(target_to_deltas[target])
        jittered_x = [
            index + deterministic_jitter(score_index, len(deltas), width=0.14)
            for score_index, _delta in enumerate(deltas)
        ]
        ax.scatter(
            jittered_x,
            deltas,
            s=28,
            alpha=0.75,
            color=TARGET_COLORS[target],
            edgecolors="white",
            linewidths=0.4,
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
        ax.text(index, mean(deltas) + 0.045, f"mean {mean(deltas):+.3f}", ha="center")

    ax.axhline(0, color=ZERO_COLOR, linewidth=1)
    ax.set_title(
        "Most MIPRO Target Rows Are Flat; Encoder Has The Mean Lift", loc="left", pad=14
    )
    ax.set_ylabel("Optimized - baseline pass rate")
    ax.set_xticks(range(len(TARGET_ORDER)), [target.title() for target in TARGET_ORDER])
    ax.set_ylim(-1.05, 1.05)
    style_axis(ax)
    ax.text(
        0,
        -0.18,
        "Each point is one paired optimizer-summary task score row; outlined dot is the mean.",
        transform=ax.transAxes,
        color=MUTED_TEXT_COLOR,
        fontsize=9,
    )
    return save_figure(fig, output_dir / "mipro_target_delta_distribution")


def plot_encoder_method_bars(
    summary_cells: list[SummaryCell], output_dir: Path
) -> list[Path]:
    cells_by_key = cell_lookup(summary_cells)
    values = [cells_by_key[(method, "encoder")].delta for method in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    bars = ax.bar(
        METHOD_ORDER,
        values,
        color=[METHOD_COLORS[method] for method in METHOD_ORDER],
        width=0.56,
    )
    for bar, value in zip(bars, values, strict=True):
        annotate_bar(ax, bar, f"{value:+.3f}")

    ax.axhline(0, color=ZERO_COLOR, linewidth=1)
    ax.set_title("MIPRO, Not GEPA, Improved The Encoder Setting", loc="left", pad=14)
    ax.set_ylabel("Optimized - baseline pass rate")
    ax.set_ylim(-0.055, 0.085)
    style_axis(ax)
    ax.text(
        0,
        -0.18,
        "GPT-5 nano enc-dec encoder optimizer-summary task scores.",
        transform=ax.transAxes,
        color=MUTED_TEXT_COLOR,
        fontsize=9,
    )
    return save_figure(fig, output_dir / "encoder_method_bars")


def plot_method_target_heatmap(
    summary_cells: list[SummaryCell], output_dir: Path
) -> list[Path]:
    cells_by_key = cell_lookup(summary_cells)
    matrix = [
        [cells_by_key[(method, target)].delta for target in TARGET_ORDER]
        for method in METHOD_ORDER
    ]

    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    cmap = LinearSegmentedColormap.from_list(
        "delta",
        ["#B85632", "#F6F0E6", "#1F6F8B"],
    )
    norm = TwoSlopeNorm(vmin=-0.09, vcenter=0, vmax=0.09)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    for row_index, method in enumerate(METHOD_ORDER):
        for col_index, target in enumerate(TARGET_ORDER):
            cell = cells_by_key[(method, target)]
            label = f"{cell.delta:+.3f}\nn={cell.n}"
            if method == "GEPA" and target == "both":
                label = "n=3\nlow coverage"
                ax.add_patch(
                    plt.Rectangle(
                        (col_index - 0.5, row_index - 0.5),
                        1,
                        1,
                        facecolor="#E7E9EC",
                        edgecolor="white",
                        linewidth=1.5,
                    )
                )
            ax.text(
                col_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=TEXT_COLOR,
                fontsize=10,
            )

    ax.set_title(
        "Only One Optimizer x Target Cell Is Clearly Strong", loc="left", pad=14
    )
    ax.set_xticks(range(len(TARGET_ORDER)), [target.title() for target in TARGET_ORDER])
    ax.set_yticks(range(len(METHOD_ORDER)), METHOD_ORDER)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Optimized - baseline pass rate")
    ax.text(
        0,
        -0.18,
        "GPT-5 nano enc-dec optimizer-summary task scores.",
        transform=ax.transAxes,
        color=MUTED_TEXT_COLOR,
        fontsize=9,
    )
    return save_figure(fig, output_dir / "method_target_heatmap")


def plot_encoder_evidence_source_comparison(
    summary_cells: list[SummaryCell],
    full5x_encoder_deltas: list[Full5xEncoderDelta],
    output_dir: Path,
) -> list[Path]:
    optimizer_cells = {
        cell.method: cell
        for cell in summary_cells
        if cell.target == "encoder" and cell.method in METHOD_ORDER
    }
    full5x_cells = {cell.method: cell for cell in full5x_encoder_deltas}
    sources = ("Optimizer summary", "Full-5x saved prompt")
    x_positions = list(range(len(sources)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for offset, method in [(-width / 2, "MIPRO"), (width / 2, "GEPA")]:
        values = [
            optimizer_cells[method].delta,
            full5x_cells[method].delta,
        ]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=width,
            label=method,
            color=METHOD_COLORS[method],
        )
        for bar, value in zip(bars, values, strict=True):
            annotate_bar(ax, bar, f"{value:+.3f}")

    ax.axhline(0, color=ZERO_COLOR, linewidth=1)
    ax.set_title("MIPRO Is The More Consistent Encoder Gain", loc="left", pad=14)
    ax.set_ylabel("Encoder optimized - baseline pass rate")
    ax.set_xticks(x_positions, sources)
    ax.set_ylim(-0.055, 0.085)
    ax.legend(frameon=False, ncols=2, loc="upper right", bbox_to_anchor=(1, 1.03))
    style_axis(ax)
    ax.text(
        0,
        -0.18,
        "Optimizer summary: GPT-5 nano paired task-score rows. Full-5x: saved-prompt export.",
        transform=ax.transAxes,
        color=MUTED_TEXT_COLOR,
        fontsize=9,
    )
    return save_figure(fig, output_dir / "encoder_evidence_source_comparison")


def cell_lookup(summary_cells: list[SummaryCell]) -> dict[tuple[str, str], SummaryCell]:
    return {(cell.method, cell.target): cell for cell in summary_cells}


def deterministic_jitter(index: int, count: int, *, width: float) -> float:
    if count <= 1:
        return 0
    position = index / (count - 1)
    return (position - 0.5) * width


def annotate_bar(axes: plt.Axes, bar: plt.Rectangle, label: str) -> None:
    value = bar.get_height()
    y_offset = 0.012 if value >= 0 else -0.018
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


def print_summary_table(summary_cells: list[SummaryCell]) -> None:
    typer.echo("Optimizer-summary deltas:")
    for target in TARGET_ORDER:
        for method in METHOD_ORDER:
            cell = cell_lookup(summary_cells)[(method, target)]
            typer.echo(
                f"- {method:5s} {target:7s}: "
                f"baseline={cell.baseline:.3f} optimized={cell.optimized:.3f} "
                f"delta={cell.delta:+.3f} n={cell.n}"
            )


def print_full5x_table(full5x_encoder_deltas: list[Full5xEncoderDelta]) -> None:
    typer.echo("\nFull-5x saved-prompt encoder deltas:")
    for cell in sorted(full5x_encoder_deltas, key=lambda value: value.method):
        typer.echo(
            f"- {cell.method:5s}: delta={cell.delta:+.3f} n={cell.n} "
            f"improved={cell.improved} worsened={cell.worsened}"
        )


if __name__ == "__main__":
    typer.run(main)
