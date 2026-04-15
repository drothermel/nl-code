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


@app.cell(column=1)
def _(ds):
    ds.model_fields
    return


@app.cell
def _(ds):
    mo.inspect(ds, value=False, methods=True)
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
