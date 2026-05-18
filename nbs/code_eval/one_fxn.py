import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")

with app.setup:
    import random

    import marimo as mo

    from nl_code.datasets import HumanEvalDataset


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


@app.cell(column=1, hide_code=True)
def _(ds):
    num_picker = mo.ui.number(
        start=0,
        stop=len(ds.raw_samples) - 1,
        step=1,
        value=110,  # random.randrange(len(ds.raw_samples)),
        label="Sample index",
    )
    num_picker
    return (num_picker,)


@app.cell(hide_code=True)
def _(ds, num_picker):
    ds.get_raw_sample_at_index(num_picker.value)
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    ### Extract Tests - Clean
    """)
    return


@app.cell
def _():
    # 13, 14, 82, 99, 129, 138 are exceptions
    return


@app.cell
def _(ds, num_picker):
    ds.get_raw_sample_at_index(num_picker.value).model_dump()["source"][
        "test"
    ].split("\n")
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
