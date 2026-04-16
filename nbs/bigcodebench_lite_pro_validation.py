"""Validate BigCodeBench-Lite-Pro ground truth solutions.

Loads the BigCodeBench-Lite-Pro dataset from HuggingFace, runs each
ground truth solution against its test cases, and reports pass/fail
status for every task.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo

    from nl_code.datasets import (
        BigCodeBenchLiteProDataset,
        RawBigCodeBenchLiteProTask,
    )


@app.cell(hide_code=True)
def _():
    ds = BigCodeBenchLiteProDataset()
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


@app.function(hide_code=True)
def render_sample_fields(sample, *, prefix=None, suppress_prefix=None):
    return mo.vstack(
        [
            mo.accordion(
                {
                    field: (
                        mo.plain_text(str(value))
                        if field in sample.non_code_fields
                        else mo.ui.code_editor(str(value))
                    )
                }
            )
            for field, value in sample.model_dump().items()
            if (prefix is None or field.startswith(prefix))
            and (
                suppress_prefix is None
                or not field.startswith(suppress_prefix)
            )
        ]
    )


@app.cell(column=1, hide_code=True)
def _(ds):
    ds.model_fields
    return


@app.cell(hide_code=True)
def _(ds):
    sample = ds.get_raw_sample_at_index(0)
    mo.inspect(
        sample,
        value=False,
    ) if False else mo.md("Toggle this to see the full sample.")
    return (sample,)


@app.cell(column=2, hide_code=True)
def _(sample):
    mo.vstack(
        [
            mo.md("## Source Fields"),
            render_sample_fields(sample, prefix="source__"),
        ]
    )
    return


@app.cell(column=3, hide_code=True)
def _(sample):
    mo.vstack(
        [
            mo.md("## Derived Fields"),
            render_sample_fields(sample, suppress_prefix="source__"),
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
