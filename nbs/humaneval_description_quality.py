"""Explore HumanEval+ description quality using nl_code primitives.

Loads the HumanEval+ dataset, computes length and lexical overlap
metrics treating docstrings as descriptions of the ground-truth code,
and visualizes distributions across the dataset.
"""

import marimo

__generated_with = "0.23.1"
app = marimo.App(width="columns")

with app.setup:
    import marimo as mo
    import altair as alt

    from nl_code.datasets.humaneval_dataset import HumanEvalDataset
    from nl_code.evaluation.length import compression_ratio, measure_length
    from nl_code.evaluation.overlap import lexical_overlap


@app.cell(hide_code=True)
def _():
    mo.md("""
    # HumanEval+ Description Quality

    This notebook explores how well HumanEval+ docstrings describe
    their corresponding code, using length and lexical overlap primitives
    from `nl_code`.
    """)
    return


@app.cell(hide_code=True)
def _():
    ds = HumanEvalDataset()
    ds.load()
    mo.md(f"""
    **Loaded {len(ds.raw_samples)} valid tasks**
    ({len(ds.flawed_raw_samples)} flawed, skipped)
    """)
    return (ds,)


@app.cell(column=1, hide_code=True)
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
    _comments = _raw.prompt_comments or "_(none)_"
    mo.vstack(
        [
            mo.md(
                f"""
                ## `{task_selector.value}` — `{_raw.entry_point}`

                ---
                """
            ),
            mo.accordion(
                {"**Docstring**": mo.plain(_raw.prompt_docstring.split("\n"))}
            ),
            mo.accordion({"**Comments**": mo.md(f"```\n{_comments}\n```")}),
            mo.accordion(
                {
                    "**Source code** (with docstring)": mo.md(
                        f"```python\n{_raw.gt_solution}```"
                    )
                }
            ),
            mo.accordion(
                {
                    "**Source code** (stripped)": mo.md(
                        f"```python\n{_raw.gt_solution_without_comments}```"
                    )
                }
            ),
            mo.accordion({"**Test harness**": mo.md(f"```python\n{_raw.test}```")}),
        ]
    )
    return


@app.cell(hide_code=True)
def _(ds, task_selector):
    _raw = ds.raw_samples[task_selector.value]
    _docstring = _raw.prompt_docstring
    _code = _raw.gt_solution_without_comments

    doc_length = measure_length(_docstring)
    code_length = measure_length(_code)
    cr = compression_ratio(_docstring, _code)
    overlap = lexical_overlap(_raw.prompt_docstring, _raw.gt_solution_without_comments)
    return code_length, cr, doc_length, overlap


@app.cell(hide_code=True)
def _(code_length, cr, doc_length):
    mo.md(f"""
    ### Length metrics

    | | chars | tokens |
    |---|---|---|
    | **Docstring** | {doc_length.char_count} | {doc_length.token_count} |
    | **Code** | {code_length.char_count} | {code_length.token_count} |
    | **Ratio** (doc/code) | {cr.char_ratio:.2f} | {cr.token_ratio:.2f} |
    """)
    return


@app.cell(hide_code=True)
def _(overlap):
    mo.md(f"""
    ### Lexical overlap (docstring vs code)

    | Metric | Value |
    |---|---|
    | **Jaccard** | {overlap.jaccard:.3f} |
    | **Overlap (doc → code)** | {overlap.overlap_a:.3f} |
    | **Overlap (code → doc)** | {overlap.overlap_b:.3f} |

    **Shared tokens**: {", ".join(f"`{t}`" for t in sorted(overlap.shared)) or "_(none)_"}

    **Only in docstring**: {", ".join(f"`{t}`" for t in sorted(overlap.only_a)) or "_(none)_"}

    **Only in code**: {", ".join(f"`{t}`" for t in sorted(overlap.only_b)) or "_(none)_"}
    """)
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md("""
    ## Dataset-wide distributions
    """)
    return


@app.cell(hide_code=True)
def _(ds):
    all_metrics = []
    for _tid, _raw in ds.raw_samples.items():
        _doc = _raw.prompt_docstring
        _code = _raw.gt_solution_without_comments
        _cr = compression_ratio(_doc, _code)
        _ov = lexical_overlap(_doc, _code)
        all_metrics.append(
            {
                "task_id": _tid,
                "entry_point": _raw.entry_point,
                "doc_chars": _cr.description.char_count,
                "code_chars": _cr.code.char_count,
                "char_ratio": _cr.char_ratio,
                "doc_tokens": _cr.description.token_count,
                "code_tokens": _cr.code.token_count,
                "token_ratio": _cr.token_ratio,
                "jaccard": _ov.jaccard,
                "overlap_doc_to_code": _ov.overlap_a,
                "overlap_code_to_doc": _ov.overlap_b,
                "shared_count": len(_ov.shared),
            }
        )
    metrics_table = mo.ui.table(
        all_metrics,
        selection=None,
        page_size=15,
        label="All tasks — description quality metrics",
    )
    metrics_table
    return (all_metrics,)


@app.cell(hide_code=True)
def _(all_metrics):
    _chart_data = alt.Data(values=all_metrics)

    char_ratio_hist = (
        alt.Chart(_chart_data)
        .mark_bar()
        .encode(
            alt.X(
                "char_ratio:Q",
                bin=alt.Bin(maxbins=30),
                title="Char ratio (doc / code)",
            ),
            alt.Y("count()", title="Count"),
        )
        .properties(
            title="Compression ratio distribution (chars)", width=400, height=250
        )
    )

    jaccard_hist = (
        alt.Chart(_chart_data)
        .mark_bar()
        .encode(
            alt.X("jaccard:Q", bin=alt.Bin(maxbins=30), title="Jaccard similarity"),
            alt.Y("count()", title="Count"),
        )
        .properties(
            title="Lexical overlap distribution (Jaccard)", width=400, height=250
        )
    )

    char_ratio_hist | jaccard_hist
    return


@app.cell(hide_code=True)
def _(all_metrics):
    _chart_data = alt.Data(values=all_metrics)

    overlap_scatter = (
        alt.Chart(_chart_data)
        .mark_circle(size=40, opacity=0.6)
        .encode(
            alt.X("overlap_doc_to_code:Q", title="Overlap: doc → code"),
            alt.Y("overlap_code_to_doc:Q", title="Overlap: code → doc"),
            alt.Tooltip(["task_id:N", "entry_point:N", "jaccard:Q"]),
        )
        .properties(
            title="Directional overlap: docstring vs code",
            width=400,
            height=400,
        )
    )
    overlap_scatter
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md("""
    ## Description vs Description Overlap
    """)
    return


@app.cell(hide_code=True)
def _(task_ids):
    task_a_selector = mo.ui.dropdown(
        options=task_ids,
        value=task_ids[0],
        label="Task A",
    )
    task_b_selector = mo.ui.dropdown(
        options=task_ids,
        value=task_ids[1] if len(task_ids) > 1 else task_ids[0],
        label="Task B",
    )
    mo.hstack([task_a_selector, task_b_selector])
    return task_a_selector, task_b_selector


@app.cell(hide_code=True)
def _(ds, task_a_selector, task_b_selector):
    _doc_a = ds.raw_samples[task_a_selector.value].prompt_docstring
    _doc_b = ds.raw_samples[task_b_selector.value].prompt_docstring
    doc_overlap = lexical_overlap(_doc_a, _doc_b)
    return (doc_overlap,)


@app.cell(hide_code=True)
def _(doc_overlap, task_a_selector, task_b_selector):
    mo.md(f"""
    ### Docstring overlap: `{task_a_selector.value}` vs `{task_b_selector.value}`

    | Metric | Value |
    |---|---|
    | **Jaccard** | {doc_overlap.jaccard:.3f} |
    | **Overlap A → B** | {doc_overlap.overlap_a:.3f} |
    | **Overlap B → A** | {doc_overlap.overlap_b:.3f} |

    **Shared**: {", ".join(f"`{t}`" for t in sorted(doc_overlap.shared)) or "_(none)_"}
    """)
    return


@app.cell(column=4, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
