"""Validate HumanEval-Pro ground truth solutions.

Loads the HumanEval-Pro dataset from HuggingFace, runs each ground
truth solution against its test cases, and reports pass/fail status
for every task.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")

with app.setup:
    import time

    import marimo as mo
    import pandas as pd

    from nl_code.datasets.humaneval_pro_dataset import HumanEvalProDataset
    from nl_code.datasets.humaneval_pro_task import RawHumanEvalProTask


@app.cell(hide_code=True)
def _():
    ds = HumanEvalProDataset()
    ds.load()
    mo.vstack(
        [
            ds.dataset_id,
            mo.md(f"""
    **Dataset loaded:** {len(ds.raw_samples)} valid tasks,
    {len(ds.flawed_raw_samples)} flawed
    """),
        ]
    )
    return (ds,)


@app.cell
def _(ds):
    mo.inspect(ds, value=False, methods=True)
    return


@app.cell(column=1)
def _(ds):
    ds.model_fields
    return


@app.cell
def _(ds):
    sample = ds.get_raw_sample_at_index(0)
    sample
    return (sample,)


@app.cell
def _(sample):
    mo.inspect(
        sample,
        value=False,
    )
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _(sample):
    mo.vstack(
        [mo.md("## Source Fields")]
        + [
            mo.accordion(
                {
                    field: mo.vstack(
                        [
                            mo.md(f"### {field}"),
                            mo.ui.code_editor(value),
                        ]
                    )
                }
            )
            for field, value in sample.model_dump().items()
            if field.startswith("source__")
        ]
    )
    return


@app.cell
def _():
    return


@app.cell(column=3, hide_code=True)
def _(sample):
    mo.vstack(
        [
            mo.md("## New Entrypoint"),
            mo.md(f"`{sample.new_entry_point}`"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(sample):
    mo.vstack(
        [
            mo.md("## New Description"),
            mo.md(sample.new_description),
        ]
    )
    return


@app.cell(hide_code=True)
def _(sample):
    mo.vstack(
        [
            mo.md("## GT Solution"),
            mo.ui.code_editor(sample.gt_solution),
        ]
    )
    return


@app.cell(hide_code=True)
def _(sample):
    mo.vstack(
        [
            mo.md("## GT Solution Without Comments"),
            mo.ui.code_editor(sample.gt_solution_without_comments),
        ]
    )
    return


@app.cell(column=4, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
