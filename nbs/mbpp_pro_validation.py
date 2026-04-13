"""Validate MBPP-Pro ground truth solutions.

Loads the MBPP-Pro dataset from HuggingFace, runs each ground
truth solution against its test cases, and reports pass/fail status
for every task.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import time

    import marimo as mo
    import pandas as pd

    from nl_code.datasets.mbpp_pro_dataset import MbppProDataset
    from nl_code.datasets.mbpp_pro_task import RawMbppProTask


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # MBPP-Pro Ground Truth Validation

    This notebook loads the full MBPP-Pro dataset and runs every
    ground truth solution against its test cases to verify correctness.
    """)
    return


@app.cell
def _():
    ds = MbppProDataset()
    ds.load()
    return (ds,)


@app.cell(hide_code=True)
def _(ds, mo):
    mo.md(f"""
    **Dataset loaded:** {len(ds.raw_samples)} valid tasks,
    {len(ds.flawed_raw_samples)} flawed (skipped during loading)
    """)
    return


@app.cell
def _(ds, time):
    results = []
    for task_id in sorted(ds.raw_samples):
        raw: RawMbppProTask = ds.raw_samples[task_id]  # type: ignore[assignment]
        start = time.perf_counter()
        try:
            passed = raw.run_test_on_gt_solution()
            error = None
        except Exception as exc:
            passed = False
            error = str(exc)
        elapsed = time.perf_counter() - start

        results.append(
            {
                "task_id": task_id,
                "entry_point": raw.new_entry_point,
                "passed": passed,
                "elapsed_s": round(elapsed, 4),
                "error": error,
            }
        )

    results_df = pd.DataFrame(results)
    return (results_df,)


@app.cell(hide_code=True)
def _(mo, results_df):
    n_passed = int(results_df["passed"].sum())
    n_total = len(results_df)
    n_failed = n_total - n_passed
    status = "All passed" if n_failed == 0 else f"**{n_failed} failures**"

    mo.md(f"""
    ## Test Results

    | Metric | Value |
    |--------|-------|
    | Total tasks tested | {n_total} |
    | Passed | {n_passed} |
    | Failed | {n_failed} |
    | Status | {status} |
    """)
    return


@app.cell
def _(results_df):
    results_df
    return


@app.cell(hide_code=True)
def _(mo, results_df):
    failed = results_df[~results_df["passed"]]
    mo.stop(
        failed.empty,
        mo.md("No failures to display."),
    )
    mo.md("## Failed Tasks")
    return (failed,)


@app.cell
def _(failed):
    failed
    return


@app.cell(hide_code=True)
def _(ds, mo):
    mo.stop(
        len(ds.flawed_raw_samples) == 0,
        mo.md("No flawed samples to display."),
    )
    flawed_ids = sorted(ds.flawed_raw_samples.keys())
    flawed_info = [
        {"task_id": tid, "error": ds.flawed_raw_samples[tid].error[:200]}
        for tid in flawed_ids
    ]
    mo.md(
        f"## Flawed Samples ({len(flawed_ids)} tasks failed validation during loading)"
    )
    return (flawed_info,)


@app.cell
def _(flawed_info, pd):
    pd.DataFrame(flawed_info)
    return


if __name__ == "__main__":
    app.run()
