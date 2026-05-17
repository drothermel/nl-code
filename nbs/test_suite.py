"""Validate HumanEval-Plus ground truth solutions.

Loads the HumanEval-Plus dataset from HuggingFace, runs each ground
truth solution against its test cases, and reports pass/fail status
for every task.
"""

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo

    from nl_code.code_execution.models import TestCase
    from nl_code.code_execution.runner import run_test_cases
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


@app.function(hide_code=True)
def build_single_test_case_solution(code: str, entry_point: str) -> str:
    return f"""{code}

def run_single_test_case(input_value):
    return {entry_point}(*input_value)
"""


@app.cell
def _(ds):
    total_n = len(ds.raw_samples)
    val_n = 10
    seed = 42
    val_ids = [
        0
    ]  # TODO: seeded random sample of ids in range [0, total_n-1] (eg from ds.raw_samples)
    return (val_ids,)


@app.cell(hide_code=True)
def _(ds):
    sample = ds.get_raw_sample_at_index(0)
    mo.inspect(
        sample,
        value=False,
    ) if False else mo.md("Toggle this to see the full sample.")
    return (sample,)


@app.cell(hide_code=True)
def _(sample):
    mo.vstack(
        [
            mo.md("## Source Fields"),
            render_sample_fields(sample, prefix="source__"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(sample):
    mo.vstack(
        [
            mo.md("## Derived Fields"),
            render_sample_fields(sample, suppress_prefix="source__"),
        ]
    )
    return


@app.cell(column=1)
def _(ds, val_ids):
    ex = ds.get_raw_sample_at_index(val_ids[0])
    return (ex,)


@app.cell(hide_code=True)
def _(ex):
    mo.md(f"""
    ```
    {ex.source__prompt}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(ex):
    mo.md(f"""
    ```
    {ex.gt_solution}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(ex):
    first_test_case = TestCase(
        input_value=ex.test_inputs[0],
        expected_output=ex.test_results[0],
    )

    mo.md(f"""
    ### First parsed test case

    ```json
    {first_test_case.model_dump_json(indent=2)}
    ```
    """)
    return (first_test_case,)


@app.cell(hide_code=True)
def _(ex, first_test_case):
    gt_solution_for_single_case = build_single_test_case_solution(
        ex.gt_solution,
        ex.entry_point,
    )

    gt_first_test_results, gt_first_test_pass_rate = run_test_cases(
        code=gt_solution_for_single_case,
        function_name="run_single_test_case",
        test_cases=[first_test_case],
    )
    gt_first_test_result = gt_first_test_results[0]

    mo.vstack(
        [
            mo.md(f"""
    ### Run ground-truth solution on the first test case

    ```python
    gt_solution_for_single_case = build_single_test_case_solution(
        ex.gt_solution,
        ex.entry_point,
    )

    gt_first_test_results, gt_first_test_pass_rate = run_test_cases(
        code=gt_solution_for_single_case,
        function_name="run_single_test_case",
        test_cases=[first_test_case],
    )
    gt_first_test_result = gt_first_test_results[0]
    ```
    """),
            mo.md(f"**Pass rate:** {gt_first_test_pass_rate:.0%}"),
            mo.ui.table([gt_first_test_result.model_dump()]),
        ]
    )
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
