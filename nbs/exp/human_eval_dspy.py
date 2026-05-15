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
    import os
    from litellm import completion
    import dspy

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


@app.cell
def _(OPENROUTER_API_KEY, OPENROUTER_BASE_URL, model):
    lm = dspy.LM(model, api_key=OPENROUTER_API_KEY, api_base=OPENROUTER_BASE_URL)
    dspy.configure(lm=lm)
    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
    )
    return (lm,)


@app.cell
def _(lm):
    lm("Say this is a test!")
    return


@app.cell
def _(lm, messages, reasoning):
    lm(messages=messages, reasoning=reasoning)
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    ### Test out LiteLLM Direct
    """)
    return


@app.cell
def _():
    # Configure with environment variables
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.getenv(
        "OPENROUTER_API_BASE", "https://openrouter.ai/api/v1"
    )

    # Set environment for LiteLLM
    os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENROUTER_API_BASE"] = OPENROUTER_BASE_URL
    return OPENROUTER_API_KEY, OPENROUTER_BASE_URL


@app.cell
def _():
    model = "openrouter/openai/gpt-5-nano"
    return (model,)


@app.cell
def _():
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    return (messages,)


@app.cell
def _():
    reasoning = {
        "effort": "minimal",
    }
    return (reasoning,)


@app.cell
def _(OPENROUTER_BASE_URL, messages, model):
    response = completion(
        model=model,
        messages=messages,
        base_url=OPENROUTER_BASE_URL,
    )
    response
    return (response,)


@app.cell(hide_code=True)
def _(response):
    mo.vstack(
        [
            mo.md(f">{response.choices[0].message.content}"),
            dict(response),
            dict(response.choices[0].message),
            dict(response.usage),
        ]
    )
    return


@app.cell
def _():
    return


@app.cell(column=3, hide_code=True)
def _():
    mo.md(r"""
    (leave space)
    """)
    return


if __name__ == "__main__":
    app.run()
