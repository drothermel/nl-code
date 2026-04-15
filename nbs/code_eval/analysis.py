import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    # All imports and constants here


@app.cell
def _():
    from nl_code.datasets.humaneval_dataset import HumanEvalDataset

    humaneval_plus = HumanEvalDataset().load()
    humaneval_plus
    return (humaneval_plus,)


@app.cell
def _():
    from nl_code.datasets.humaneval_pro_dataset import HumanEvalProDataset

    humaneval_pro = HumanEvalProDataset().load()
    humaneval_pro
    return (humaneval_pro,)


@app.cell
def _():
    from nl_code.datasets.mbpp_pro_dataset import MbppProDataset

    mbpp_pro = MbppProDataset().load()
    mbpp_pro
    return (mbpp_pro,)


@app.cell
def _():
    from nl_code.datasets.bigcodebench_lite_pro_dataset import BigCodeBenchLiteProDataset

    bigcodebench_lite_pro = BigCodeBenchLiteProDataset().load()
    bigcodebench_lite_pro
    return (bigcodebench_lite_pro,)


@app.cell
def _():
    from datasets import load_dataset
    from nl_code.datasets.classeval_dataset import ClassEvalDataset

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

    class_eval
    return (class_eval,)


@app.cell
def _(humaneval_pro):
    humaneval_pro_i = 0

    list(humaneval_pro.tasks.values())[humaneval_pro_i].gt_solution
    return


@app.cell
def _(mbpp_pro):
    mbpp_pro_i = 0

    list(mbpp_pro.tasks.values())[mbpp_pro_i].gt_solution
    return


@app.cell
def _(bigcodebench_lite_pro):
    bigcodebench_lite_pro_i = 0

    list(bigcodebench_lite_pro.tasks.values())[bigcodebench_lite_pro_i].gt_solution
    return


@app.cell
def _(class_eval):
    class_eval_i = 0

    list(class_eval.tasks.values())[class_eval_i].gt_solution
    return


@app.cell(column=1)
def _(humaneval_plus):
    humaneval_plus_i = 0

    list(humaneval_plus.tasks.values())[humaneval_plus_i].gt_solution
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
