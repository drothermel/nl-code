import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    from datasets import load_dataset

    from nl_code.datasets.bigcodebench_lite_pro_dataset import (
        BigCodeBenchLiteProDataset,
    )
    from nl_code.datasets.classeval_dataset import ClassEvalDataset
    from nl_code.datasets.humaneval_dataset import HumanEvalDataset
    from nl_code.datasets.humaneval_pro_dataset import HumanEvalProDataset
    from nl_code.datasets.mbpp_pro_dataset import MbppProDataset


@app.function
def render_example(dataset_name: str, i: int, code: str):
    return mo.vstack(
        [
            mo.md(f"**{dataset_name}** (Sample `{i}`)"),
            mo.md(f"```python\n{code}\n```"),
        ]
    )


@app.cell(hide_code=True)
def _():
    humaneval_plus = HumanEvalDataset().load()
    f"HumanEval+: {len(humaneval_plus.tasks)} tasks"
    return (humaneval_plus,)


@app.cell(hide_code=True)
def _():
    humaneval_pro = HumanEvalProDataset().load()
    f"HumanEval Pro: {len(humaneval_pro.tasks)} tasks"
    return (humaneval_pro,)


@app.cell(hide_code=True)
def _():
    mbpp_pro = MbppProDataset().load()
    f"MBPP Pro: {len(mbpp_pro.tasks)} tasks"
    return (mbpp_pro,)


@app.cell(hide_code=True)
def _():
    bigcodebench_lite_pro = BigCodeBenchLiteProDataset().load()
    f"BigCodeBench Lite Pro: {len(bigcodebench_lite_pro.tasks)} tasks"
    return (bigcodebench_lite_pro,)


@app.cell
def _():
    _class_eval_base = ClassEvalDataset.model_construct()
    _class_eval_rows = load_dataset(
        _class_eval_base.dataset_id.value,
        split=_class_eval_base.split,
        revision=_class_eval_base.source_revision,
    )

    class_eval = _class_eval_base.model_copy(deep=True)
    class_eval.raw_samples = {}
    class_eval.tasks = {}
    class_eval.flawed_raw_samples = {}

    for row in _class_eval_rows:
        row_dict = dict(row)
        try:
            task_id = _class_eval_base._extract_task_id(row_dict)
            raw = _class_eval_base._parse_row(row_dict)
            class_eval.raw_samples[task_id] = raw
            class_eval.tasks[task_id] = _class_eval_base._to_task(task_id, raw)
        except Exception:
            pass

    f"ClassEval: {len(class_eval.tasks)} tasks"
    return (class_eval,)


@app.cell(column=1, hide_code=True)
def _(humaneval_plus):
    humaneval_plus_i = 0

    render_example(
        "HumanEval+",
        humaneval_plus_i,
        list(humaneval_plus.tasks.values())[humaneval_plus_i].gt_solution,
    )
    return


@app.cell(hide_code=True)
def _(humaneval_pro):
    humaneval_pro_i = 0

    render_example(
        "HumanEval Pro",
        humaneval_pro_i,
        list(humaneval_pro.tasks.values())[humaneval_pro_i].gt_solution,
    )
    return


@app.cell(column=2, hide_code=True)
def _(mbpp_pro):
    mbpp_pro_i = 0

    render_example(
        "MBPP Pro",
        mbpp_pro_i,
        list(mbpp_pro.tasks.values())[mbpp_pro_i].gt_solution,
    )
    return


@app.cell(hide_code=True)
def _(bigcodebench_lite_pro):
    bigcodebench_lite_pro_i = 0

    render_example(
        "BigCodeBench Lite Pro",
        bigcodebench_lite_pro_i,
        list(bigcodebench_lite_pro.tasks.values())[bigcodebench_lite_pro_i].gt_solution,
    )
    return


@app.cell(column=3, hide_code=True)
def _(class_eval):
    class_eval_i = 0

    render_example(
        "ClassEval",
        class_eval_i,
        list(class_eval.tasks.values())[class_eval_i].gt_solution,
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
