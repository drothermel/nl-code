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

    from nl_code.datasets.humaneval_task import RawHumanEvalTask
    from nl_code.datasets import (
        RawHumanEvalProTask,
        RawHumanEvalTask,
        HumanEvalDataset,
        HumanEvalProDataset,
    )


@app.cell(hide_code=True)
def _():
    ds = HumanEvalDataset()
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


@app.cell(column=3, hide_code=True)
def _(sample):
    mo.vstack(
        [mo.md("## Derived Fields")]
        + [
            mo.accordion(
                {
                    field: mo.vstack(
                        [
                            mo.md(f"### {field}"),
                            mo.ui.code_editor(str(value)),
                        ]
                    )
                }
            )
            for field, value in sample.model_dump().items()
            if not field.startswith("source__")
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
