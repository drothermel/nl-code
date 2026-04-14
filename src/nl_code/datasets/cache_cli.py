from collections.abc import Iterable
import os

import typer

from nl_code.datasets.cache import clear_snapshot, read_manifest
from nl_code.datasets.catalog import DATASET_TYPES_BY_KEY

app = typer.Typer(help="Manage parsed dataset caches.")


def _resolve_dataset_keys(dataset_key: str) -> Iterable[str]:
    if dataset_key == "all":
        return DATASET_TYPES_BY_KEY.keys()
    if dataset_key not in DATASET_TYPES_BY_KEY:
        raise typer.BadParameter(
            f"Unknown dataset key {dataset_key!r}. Expected one of: "
            + ", ".join(sorted([*DATASET_TYPES_BY_KEY.keys(), "all"]))
        )
    return [dataset_key]


@app.command()
def rebuild(
    dataset_key: str = typer.Argument(..., help="Dataset key or 'all'."),
    offline: bool = typer.Option(
        False, help="Use only locally available HF artifacts."
    ),
) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    for key in _resolve_dataset_keys(dataset_key):
        dataset = DATASET_TYPES_BY_KEY[key].model_construct()
        typer.echo(f"Rebuilding {key}...")
        dataset.rebuild_cache(offline=offline)
        manifest = read_manifest(dataset.dataset_id, dataset.split)
        if manifest is None:
            raise RuntimeError(
                "Rebuild completed but no cache manifest was written for "
                f"{dataset.dataset_id.value} (split={dataset.split})"
            )
        typer.echo(
            f"{key}: cached {manifest.task_count} tasks "
            f"({manifest.raw_sample_count} raw, {manifest.flawed_count} flawed)"
        )


@app.command()
def status(
    dataset_key: str = typer.Argument(..., help="Dataset key or 'all'."),
) -> None:
    for key in _resolve_dataset_keys(dataset_key):
        dataset = DATASET_TYPES_BY_KEY[key].model_construct()
        manifest = read_manifest(dataset.dataset_id, dataset.split)
        if manifest is None:
            typer.echo(f"{key}: missing")
            continue
        typer.echo(
            f"{key}: cached {manifest.task_count} tasks "
            f"({manifest.raw_sample_count} raw, {manifest.flawed_count} flawed) "
            f"built_at={manifest.built_at} revision={manifest.source_revision or 'unversioned'}"
        )


@app.command()
def clear(
    dataset_key: str = typer.Argument(..., help="Dataset key or 'all'."),
) -> None:
    for key in _resolve_dataset_keys(dataset_key):
        dataset = DATASET_TYPES_BY_KEY[key].model_construct()
        cleared = clear_snapshot(dataset.dataset_id, dataset.split)
        typer.echo(f"{key}: {'cleared' if cleared else 'missing'}")


if __name__ == "__main__":
    app()
