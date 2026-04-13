"""Verify ClassEval ground-truth solutions pass their test suites.

Loads the ClassEval dataset, runs each task's unittest suite against
the reference solution, and reports results.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    from nl_code.datasets.classeval_dataset import ClassEvalDataset


@app.cell(hide_code=True)
def _():
    mo.md("""
    # ClassEval Ground-Truth Verification

    Loads all ClassEval tasks and runs the unittest suites against
    the reference (ground-truth) solutions to verify they all pass.
    """)
    return


@app.cell(hide_code=True)
def _():
    ds = ClassEvalDataset()
    ds.load()
    mo.md(f"""
    **Loaded {len(ds.raw_samples)} valid tasks**
    ({len(ds.flawed_raw_samples)} flawed, skipped)
    """)
    return (ds,)


@app.cell(hide_code=True)
def _(ds):
    results = []
    for task_id, raw in sorted(ds.raw_samples.items()):
        result = raw.run_test_on_gt_solution()
        results.append(
            {
                "task_id": task_id,
                "class_name": raw.class_name,
                "all_passed": result.all_passed,
                "tests_run": result.total_tests_run,
                "tests_passed": result.total_tests_passed,
                "tests_failed": result.total_tests_failed,
                "tests_errored": result.total_tests_errored,
                "error": result.error,
            }
        )
    results
    return (results,)


@app.cell(hide_code=True)
def _(results):
    passed = sum(1 for r in results if r["all_passed"])
    total = len(results)
    status = "all pass" if passed == total else f"**{total - passed} failures**"
    mo.md(f"""
    ## Summary: {passed}/{total} tasks passed ({status})
    """)
    return


@app.cell(hide_code=True)
def _(results):
    failures = [r for r in results if not r["all_passed"]]
    if failures:
        mo.md("### Failed tasks")
        mo.ui.table(failures, selection=None, page_size=20)
    else:
        mo.md("_No failures._")
    return


@app.cell(hide_code=True)
def _(ds):
    task_ids = sorted(ds.raw_samples.keys())
    task_selector = mo.ui.dropdown(
        options=task_ids,
        value=task_ids[0],
        label="Task",
    )
    task_selector
    return task_ids, task_selector


@app.cell(hide_code=True)
def _(ds, task_selector):
    _raw = ds.raw_samples[task_selector.value]
    _result = _raw.run_test_on_gt_solution()
    _per_class = [
        {
            "test_class": r.test_class_name,
            "passed": r.passed,
            "run": r.tests_run,
            "passed_count": r.tests_passed,
            "failed": r.tests_failed,
            "errored": r.tests_errored,
        }
        for r in _result.per_test_class
    ]
    mo.vstack(
        [
            mo.md(
                f"""
                ## `{task_selector.value}` — `{_raw.class_name}`

                {_raw.class_description}

                ---
                """
            ),
            mo.accordion({"**Skeleton**": mo.md(f"```python\n{_raw.skeleton}```")}),
            mo.accordion(
                {"**Solution**": mo.md(f"```python\n{_raw.solution_code}```")}
            ),
            mo.accordion({"**Test suite**": mo.md(f"```python\n{_raw.test}```")}),
            mo.md("### Test results per class"),
            mo.ui.table(_per_class, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
