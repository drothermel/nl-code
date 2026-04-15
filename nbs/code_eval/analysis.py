import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    from pprint import pformat

    from pydantic import BaseModel

    from nl_code.code_analysis import (
        analyze_code_style,
        analyze_function,
        check_function_exists,
        check_has_assert,
        check_has_print,
        check_has_raise,
        check_has_return,
        check_python_syntax,
        count_control_structures,
        extract_from_code_fences,
        extract_inline_comments,
        extract_string_literals,
        get_parameter_names,
        get_return_type_annotation,
    )
    from nl_code.datasets.bigcodebench_lite_pro_dataset import (
        BigCodeBenchLiteProDataset,
    )
    from nl_code.datasets.classeval_dataset import ClassEvalDataset
    from nl_code.datasets.humaneval_dataset import HumanEvalDataset
    from nl_code.datasets.humaneval_pro_dataset import HumanEvalProDataset
    from nl_code.datasets.mbpp_pro_dataset import MbppProDataset
    from nl_code.evaluation.length import compression_ratio, measure_length
    from nl_code.evaluation.overlap import lexical_overlap
    from nl_code.evaluation.tokenizer import tokenize

    def normalize_analysis_output(value):
        if isinstance(value, BaseModel):
            return normalize_analysis_output(value.model_dump())
        if isinstance(value, dict):
            return {key: normalize_analysis_output(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [normalize_analysis_output(item) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [normalize_analysis_output(item) for item in value]
            return sorted(normalized, key=repr)
        return value

    def analyze_code_sample(task, code_sample: str):
        extracted_code, had_fences = extract_from_code_fences(code_sample)
        syntax_valid, syntax_error = check_python_syntax(extracted_code)
        function_exists = check_function_exists(extracted_code, task.entry_point_name)

        try:
            aggregate_analysis = analyze_function(extracted_code, task.entry_point_name)
        except Exception as exc:
            aggregate_analysis = {
                "error": str(exc),
                "function_name": task.entry_point_name,
            }

        results = {
            "entry_point_name": task.entry_point_name,
            "code_fence_extraction": {
                "extracted_code": extracted_code,
                "had_fences": had_fences,
            },
            "syntax_validation": {
                "valid": syntax_valid,
                "error": syntax_error,
            },
            "expected_top_level_function_exists": function_exists,
            "function_structure_checks": {
                "has_return": check_has_return(extracted_code, task.entry_point_name),
                "has_print": check_has_print(extracted_code, task.entry_point_name),
                "has_raise": check_has_raise(extracted_code, task.entry_point_name),
                "has_assert": check_has_assert(extracted_code, task.entry_point_name),
                "return_type_annotation": get_return_type_annotation(
                    extracted_code, task.entry_point_name
                ),
                "parameter_names": get_parameter_names(
                    extracted_code, task.entry_point_name
                ),
            },
            "comments_and_strings": {
                "inline_comments": extract_inline_comments(
                    extracted_code, task.entry_point_name
                ),
                "string_literals": extract_string_literals(
                    extracted_code, task.entry_point_name
                ),
            },
            "control_flow_counts": count_control_structures(
                extracted_code, task.entry_point_name
            ),
            "style_metrics": analyze_code_style(extracted_code, task.entry_point_name),
            "aggregate_analysis": aggregate_analysis,
            "text_code_metrics": {
                "code_tokens": tokenize(extracted_code),
                "description_tokens": tokenize(task.description),
                "code_length": measure_length(extracted_code),
                "description_length": measure_length(task.description),
                "compression_ratio": compression_ratio(
                    task.description, extracted_code
                ),
                "lexical_overlap": lexical_overlap(task.description, extracted_code),
            },
        }
        return normalize_analysis_output(results)


@app.function(hide_code=True)
def render_example(dataset_name: str, requested_i: int, dataset):
    tasks = list(dataset.tasks.values())
    actual_i = min(requested_i, len(tasks) - 1)
    task = tasks[actual_i]
    analysis_results = analyze_code_sample(task, task.gt_solution)
    header = f"**{dataset_name}** (Sample `{actual_i}`)"
    if actual_i != requested_i:
        header = f"**{dataset_name}** (Requested `{requested_i}`, showing `{actual_i}`)"

    return mo.vstack(
        [
            mo.md(header),
            mo.md(f"```python\n{task.gt_solution}\n```"),
            mo.accordion(
                {
                    "Analysis": mo.md(
                        f"```python\n{pformat(analysis_results, sort_dicts=False, width=100)}\n```"
                    )
                },
                lazy=True,
            ),
        ]
    )


@app.cell(hide_code=True)
def _():
    datasets = {
        "HumanEval+": HumanEvalDataset().load(),
        "HumanEval Pro": HumanEvalProDataset().load(),
        "MBPP Pro": MbppProDataset().load(),
        "BigCodeBench Lite Pro": BigCodeBenchLiteProDataset().load(),
        "ClassEval": ClassEvalDataset().load(),
    }

    mo.vstack(
        [
            mo.md(f"- `{name}`: {len(dataset.tasks)} tasks")
            for name, dataset in datasets.items()
        ]
    )
    return (datasets,)


@app.cell(column=1, hide_code=True)
def _(datasets):
    max_sample_ind = max(len(dataset.tasks) - 1 for dataset in datasets.values())
    sample_ind = mo.ui.number(
        start=0,
        stop=max_sample_ind,
        step=1,
        value=0,
        label="Sample index",
    )

    mo.vstack(
        [
            mo.md(f"**Maximum shared sample index: `{max_sample_ind}`**"),
            sample_ind,
        ]
    )
    return (sample_ind,)


@app.cell(hide_code=True)
def _(datasets, sample_ind):
    render_example("HumanEval+", int(sample_ind.value or 0), datasets["HumanEval+"])
    return


@app.cell(hide_code=True)
def _(datasets, sample_ind):
    render_example(
        "HumanEval Pro", int(sample_ind.value or 0), datasets["HumanEval Pro"]
    )
    return


@app.cell(column=2, hide_code=True)
def _(datasets, sample_ind):
    render_example("MBPP Pro", int(sample_ind.value or 0), datasets["MBPP Pro"])
    return


@app.cell(hide_code=True)
def _(datasets, sample_ind):
    render_example(
        "BigCodeBench Lite Pro",
        int(sample_ind.value or 0),
        datasets["BigCodeBench Lite Pro"],
    )
    return


@app.cell(column=3, hide_code=True)
def _(datasets, sample_ind):
    render_example("ClassEval", int(sample_ind.value or 0), datasets["ClassEval"])
    return


@app.cell(column=4, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
