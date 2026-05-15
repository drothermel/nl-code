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
    import json
    import random
    from datetime import datetime, timezone
    from pathlib import Path
    from litellm import completion
    import dspy

    from nl_code.code_execution.models import (
        CodeExecutionInfrastructureError,
        TestCase,
        TestCaseResult,
    )
    from nl_code.code_execution.runner import run_test_cases
    from nl_code.datasets import HumanEvalDataset

    NOTEBOOK_PATH = Path(__file__).resolve()
    REPO_ROOT = NOTEBOOK_PATH.parents[2]
    LOGS_DIR = REPO_ROOT / "logs"
    LOGS_DIR.mkdir(exist_ok=True)
    HUMAN_EVAL_DSPY_LOG_PATH = LOGS_DIR / (
        f"human_eval_dspy_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    )
    HUMAN_EVAL_DSPY_LOG_PATH.touch(exist_ok=True)


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

    CODE_GENERATION_INSTRUCTIONS = (
        "Implement the requested function using the provided specification. "
        "Return only executable Python code. Do not include explanations or Markdown."
    )
    ENCODER_INSTRUCTIONS = (
        "Provide a concise natural language description of the code. "
        "Do not output anything else."
    )
    model = "openrouter/openai/gpt-5-nano"
    reasoning = {"effort": "minimal"}
    return (
        CODE_GENERATION_INSTRUCTIONS,
        ENCODER_INSTRUCTIONS,
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        model,
        reasoning,
    )


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
def display_encdec_generation(data_sample, pred_output):
    display_out = [
        mo.md(f"""
    ### **Enc Dec**
    **Pass rate:** {pred_output["pass_rate"]:.0%}
    """),
        mo.md("#### **GT Code**"),
        mo.ui.code_editor(data_sample.gt_solution),
        mo.md("#### **Encoded specification**"),
        mo.md(pred_output["prediction"].code_spec),
        mo.md("#### **Extracted Generation**"),
        mo.ui.code_editor(pred_output["extracted"], language="python"),
    ]
    if pred_output["pass_rate"] < 1.0:
        display_out.append(
            mo.ui.table(
                [result.model_dump() for result in pred_output["results"]]
            )
        )

    return mo.vstack(display_out)


@app.function(hide_code=True)
def display_direct_generation(data_sample, pred_output):
    display_out = [
        mo.md(f"""
    ### **Direct Generation**
    **Pass rate:** {pred_output["pass_rate"]:.0%}
    """),
        mo.md("#### **Prompt**"),
        mo.ui.code_editor(data_sample.source__prompt),
        mo.md("#### **GT Code**"),
        mo.ui.code_editor(data_sample.gt_solution),
        mo.md("#### **Extracted Generation**"),
        mo.ui.code_editor(pred_output["extracted"], language="python"),
    ]
    if pred_output["pass_rate"] < 1.0:
        display_out.append(
            mo.ui.table(
                [result.model_dump() for result in pred_output["results"]]
            )
        )

    return mo.vstack(display_out)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Direct Generation Defs
    """)
    return


@app.cell
def _(CODE_GENERATION_INSTRUCTIONS):
    class CompleteCodeFromStub(dspy.Signature):
        code_stub: str = dspy.InputField(
            desc=(
                "Partial Python source containing imports, function signature, "
                "and any available docstring or comments to complete."
            )
        )
        completed_code: str = dspy.OutputField(
            desc="Complete executable Python source code."
        )


    CompleteCodeFromStub = CompleteCodeFromStub.with_instructions(
        CODE_GENERATION_INSTRUCTIONS
    )
    return (CompleteCodeFromStub,)


@app.cell
def _(CompleteCodeFromStub):
    class DirectCodeGenerator(dspy.Module):
        def __init__(self):
            super().__init__()
            self.complete = dspy.Predict(CompleteCodeFromStub)

        def forward(self, code_stub: str):
            return self.complete(code_stub=code_stub)

    return (DirectCodeGenerator,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Encoder-Decoder Defs
    """)
    return


@app.cell
def _(CODE_GENERATION_INSTRUCTIONS, ENCODER_INSTRUCTIONS):
    class EncodeCodeSpec(dspy.Signature):
        input_code: str = dspy.InputField(
            desc="Complete Python source code to describe."
        )
        code_spec: str = dspy.OutputField(
            desc="Concise natural-language behavior specification for the code."
        )


    EncodeCodeSpec = EncodeCodeSpec.with_instructions(ENCODER_INSTRUCTIONS)


    class DecodeCodeSpec(dspy.Signature):
        code_spec: str = dspy.InputField(
            desc="Natural-language behavior specification for the requested function."
        )
        function_stub: str = dspy.InputField(
            desc=(
                "Python imports and function signature to complete; comments and "
                "docstrings are intentionally omitted."
            )
        )
        completed_code: str = dspy.OutputField(
            desc="Complete executable Python source code."
        )


    DecodeCodeSpec = DecodeCodeSpec.with_instructions(CODE_GENERATION_INSTRUCTIONS)
    return DecodeCodeSpec, EncodeCodeSpec


@app.cell
def _(DecodeCodeSpec, EncodeCodeSpec):
    class CodeSpecEncoder(dspy.Module):
        def __init__(self):
            super().__init__()
            self.encode = dspy.Predict(EncodeCodeSpec)

        def forward(self, input_code: str):
            return self.encode(input_code=input_code)


    class CodeSpecDecoder(dspy.Module):
        def __init__(self):
            super().__init__()
            self.decode = dspy.Predict(DecodeCodeSpec)

        def forward(self, code_spec: str, function_stub: str):
            return self.decode(code_spec=code_spec, function_stub=function_stub)


    class EncoderDecoderCodeGenerator(dspy.Module):
        def __init__(self, encoder=None, decoder=None):
            super().__init__()
            self.encoder = encoder or CodeSpecEncoder()
            self.decoder = decoder or CodeSpecDecoder()

        def forward(self, input_code: str, function_stub: str):
            encoded = self.encoder(input_code=input_code)
            decoded = self.decoder(
                code_spec=encoded.code_spec,
                function_stub=function_stub,
            )
            return dspy.Prediction(
                code_spec=encoded.code_spec,
                completed_code=decoded.completed_code,
            )

    return (EncoderDecoderCodeGenerator,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Setup LLM and Run Gen Helpers
    """)
    return


@app.cell
def _(
    DirectCodeGenerator,
    EncoderDecoderCodeGenerator,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    model,
    reasoning,
):
    lm = dspy.LM(
        model,
        api_key=OPENROUTER_API_KEY,
        api_base=OPENROUTER_BASE_URL,
        reasoning=reasoning,
    )
    dspy.configure(lm=lm)
    dspy.configure_cache(
        enable_disk_cache=False,
        enable_memory_cache=False,
    )
    direct_generator = DirectCodeGenerator()
    encoder_decoder_generator = EncoderDecoderCodeGenerator()
    return direct_generator, encoder_decoder_generator, lm


@app.cell
def _():
    def _json_default(value):
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if hasattr(value, "toDict"):
            return value.toDict()
        if hasattr(value, "dict"):
            return value.dict()
        return str(value)


    def dump_latest_lm_history(lm_instance, log_file):
        if log_file is None:
            return None
        if not lm_instance.history:
            return None

        record = dict(lm_instance.history[-1])
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(record, default=_json_default, ensure_ascii=False)
                + "\n"
            )

        return log_path

    return (dump_latest_lm_history,)


@app.cell
def _(
    dump_latest_lm_history,
    evaluate_completed_code,
    extract_python_code,
    lm,
):
    def run_gen_eval(data_sample, generator, log_file=None):
        if data_sample.test_results is None:
            return {
                "prediction": None,
                "log_file": None,
                "extracted": "",
                "results": [],
                "pass_rate": 0.0,
                "skipped": True,
                "error": "sample does not provide expected test results",
                "task_id": data_sample.task_id,
            }

        output = {"skipped": False, "error": None, "task_id": data_sample.task_id}
        output["prediction"] = generator(
            code_stub=data_sample.source__prompt,
        )
        output["log_file"] = dump_latest_lm_history(lm, log_file)
        output["extracted"] = extract_python_code(
            output["prediction"].completed_code
        )
        output["results"], output["pass_rate"] = evaluate_completed_code(
            output["extracted"], data_sample
        )
        output_errors = [
            result.error or result.compile_error
            for result in output["results"]
            if result.error or result.compile_error
        ]
        if output_errors:
            output["error"] = output_errors[0]
        return output

    return (run_gen_eval,)


@app.cell
def _(
    dump_latest_lm_history,
    evaluate_completed_code,
    extract_python_code,
    lm,
):
    def run_encdec_eval(data_sample, generator, log_file=None):
        if data_sample.test_results is None:
            return {
                "prediction": None,
                "log_file": None,
                "extracted": "",
                "results": [],
                "pass_rate": 0.0,
                "skipped": True,
                "error": "sample does not provide expected test results",
                "task_id": data_sample.task_id,
            }

        output = {"skipped": False, "error": None, "task_id": data_sample.task_id}
        output["prediction"] = generator(
            input_code=data_sample.gt_solution,
            function_stub=data_sample.function_stub,
        )
        output["log_file"] = dump_latest_lm_history(lm, log_file)
        output["extracted"] = extract_python_code(
            output["prediction"].completed_code
        )
        output["results"], output["pass_rate"] = evaluate_completed_code(
            output["extracted"], data_sample
        )
        output_errors = [
            result.error or result.compile_error
            for result in output["results"]
            if result.error or result.compile_error
        ]
        if output_errors:
            output["error"] = output_errors[0]
        return output

    return (run_encdec_eval,)


@app.function
def sample_dataset_indices(dataset, val_num: int, seed: int) -> list[int]:
    if val_num < 1:
        raise ValueError("val_num must be positive")

    evaluable_indices = [
        index
        for index in range(len(dataset.raw_samples))
        if dataset.get_raw_sample_at_index(index).test_results is not None
    ]
    sample_n = min(val_num, len(evaluable_indices))
    return random.Random(seed).sample(evaluable_indices, k=sample_n)


@app.function(hide_code=True)
def timestamped_eval_log_path(eval_type: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return LOGS_DIR / f"human_eval_dspy_{eval_type}_eval_{timestamp}.json"


@app.function(hide_code=True)
def dump_full_eval_run(
    *,
    eval_type: str,
    dataset_indices: list[int],
    outputs: dict[int, dict],
    val_num: int,
    seed: int,
    log_file=None,
):
    def _eval_json_default(value):
        if isinstance(value, Path):
            return str(value)
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        if hasattr(value, "toDict"):
            return value.toDict()
        if hasattr(value, "dict"):
            return value.dict()
        return str(value)

    timestamp = datetime.now(timezone.utc).isoformat()
    log_path = (
        Path(log_file)
        if log_file is not None
        else timestamped_eval_log_path(eval_type)
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": timestamp,
        "eval_type": eval_type,
        "val_num": val_num,
        "seed": seed,
        "dataset_indices": dataset_indices,
        "outputs": [
            {
                "dataset_index": dataset_index,
                "output": outputs[dataset_index],
            }
            for dataset_index in dataset_indices
        ],
    }
    log_path.write_text(
        json.dumps(
            payload, default=_eval_json_default, ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    return log_path


@app.function(hide_code=True)
def run_full_eval_loop(
    *,
    dataset,
    val_num: int,
    seed: int,
    eval_type: str,
    run_eval,
    generator,
    generation_log_file=HUMAN_EVAL_DSPY_LOG_PATH,
    run_log_file=None,
):
    dataset_indices = sample_dataset_indices(dataset, val_num, seed)
    outputs = {}
    for dataset_index in dataset_indices:
        data_sample = dataset.get_raw_sample_at_index(dataset_index)
        outputs[dataset_index] = run_eval(
            data_sample,
            generator,
            log_file=generation_log_file,
        )
    run_log_path = dump_full_eval_run(
        eval_type=eval_type,
        dataset_indices=dataset_indices,
        outputs=outputs,
        val_num=val_num,
        seed=seed,
        log_file=run_log_file,
    )
    return outputs, run_log_path


@app.cell(hide_code=True)
def _(direct_generator, run_gen_eval):
    def run_full_direct_eval_loop(
        dataset,
        val_num: int,
        seed: int,
        generation_log_file=HUMAN_EVAL_DSPY_LOG_PATH,
        run_log_file=None,
    ):
        return run_full_eval_loop(
            dataset=dataset,
            val_num=val_num,
            seed=seed,
            eval_type="direct",
            run_eval=run_gen_eval,
            generator=direct_generator,
            generation_log_file=generation_log_file,
            run_log_file=run_log_file,
        )

    return (run_full_direct_eval_loop,)


@app.cell(hide_code=True)
def _(encoder_decoder_generator, run_encdec_eval):
    def run_full_encdec_eval_loop(
        dataset,
        val_num: int,
        seed: int,
        generation_log_file=HUMAN_EVAL_DSPY_LOG_PATH,
        run_log_file=None,
    ):
        return run_full_eval_loop(
            dataset=dataset,
            val_num=val_num,
            seed=seed,
            eval_type="encdec",
            run_eval=run_encdec_eval,
            generator=encoder_decoder_generator,
            generation_log_file=generation_log_file,
            run_log_file=run_log_file,
        )

    return (run_full_encdec_eval_loop,)


@app.function(hide_code=True)
def summarize_full_eval_outputs(
    outputs: dict[int, dict],
) -> dict[str, float | int]:
    evaluated_outputs = [
        output
        for output in outputs.values()
        if not output.get("skipped", False)
    ]
    pass_rates = [output["pass_rate"] for output in evaluated_outputs]
    sample_pass_count = sum(pass_rate == 1.0 for pass_rate in pass_rates)
    evaluated_samples = len(pass_rates)
    total_outputs = len(outputs)
    skipped_count = total_outputs - evaluated_samples
    return {
        "total_outputs": total_outputs,
        "evaluated_samples": evaluated_samples,
        "skipped_count": skipped_count,
        "sample_pass_count": sample_pass_count,
        "binary_sample_pass_rate": (
            sample_pass_count / evaluated_samples if evaluated_samples else 0.0
        ),
        "average_test_pass_rate": (
            sum(pass_rates) / evaluated_samples if evaluated_samples else 0.0
        ),
    }


@app.cell(hide_code=True)
def _(run_full_direct_eval_loop, run_full_encdec_eval_loop):
    full_eval_loop_helpers = {
        "sample_dataset_indices": sample_dataset_indices,
        "run_full_direct_eval_loop": run_full_direct_eval_loop,
        "run_full_encdec_eval_loop": run_full_encdec_eval_loop,
        "summarize_full_eval_outputs": summarize_full_eval_outputs,
    }

    mo.md(r"""
    Full eval loop helpers are available.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Eval Helpers
    """)
    return


@app.cell
def _():
    RUN_SINGLE_TEST_CASE_FUNCTION = "run_single_test_case"


    def extract_python_code(text: str) -> str:
        stripped = text.strip()
        if "```" not in stripped:
            return stripped + "\n"

        fenced = stripped.split("```", maxsplit=1)[1]
        if fenced.startswith("python"):
            fenced = fenced[len("python") :].lstrip()
        return fenced.split("```", maxsplit=1)[0].strip() + "\n"


    def build_single_test_case_solution(code: str, entry_point: str) -> str:
        return "\n".join(
            [
                code.rstrip(),
                "",
                "",
                f"def {RUN_SINGLE_TEST_CASE_FUNCTION}(input_value):",
                f"    return {entry_point}(*input_value)",
                "",
            ]
        )


    def build_test_cases(sample) -> list[TestCase]:
        if sample.test_results is None:
            raise ValueError("sample does not provide expected test results")
        return [
            TestCase(input_value=input_value, expected_output=expected_output)
            for input_value, expected_output in zip(
                sample.test_inputs,
                sample.test_results,
                strict=True,
            )
        ]


    def failed_eval_results(
        test_cases: list[TestCase],
        error: str,
    ) -> list[TestCaseResult]:
        return [
            TestCaseResult(
                input_value=test_case.input_value,
                expected_output=test_case.expected_output,
                actual_output=None,
                passed=False,
                error=error,
                compile_success=False,
                compile_error=error,
            )
            for test_case in test_cases
        ]


    def evaluate_completed_code(completed_code: str, sample):
        test_cases = build_test_cases(sample)
        eval_code = build_single_test_case_solution(
            extract_python_code(completed_code),
            sample.entry_point,
        )
        try:
            return run_test_cases(
                code=eval_code,
                function_name=RUN_SINGLE_TEST_CASE_FUNCTION,
                test_cases=test_cases,
            )
        except CodeExecutionInfrastructureError as exc:
            error = str(exc)
            return failed_eval_results(test_cases, error), 0.0

    return evaluate_completed_code, extract_python_code


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
def _():
    n_samples = 163
    seed = 42
    return n_samples, seed


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Direct Generation Loop
    """)
    return


@app.cell(hide_code=True)
def _(ds, n_samples, run_full_direct_eval_loop, seed):
    direct_full_eval_outputs, direct_full_eval_log_path = (
        run_full_direct_eval_loop(
            ds,
            val_num=n_samples,
            seed=seed,
        )
    )
    direct_full_eval_summary = summarize_full_eval_outputs(
        direct_full_eval_outputs
    )

    direct_full_eval_rows = [
        {
            "dataset_index": dataset_index,
            "task_id": output.get("task_id"),
            "skipped": output.get("skipped", False),
            "passed": output["pass_rate"] == 1.0,
            "test_pass_rate": output["pass_rate"],
            "error": output.get("error"),
            "log_file": output.get("log_file"),
        }
        for dataset_index, output in direct_full_eval_outputs.items()
    ]
    return (
        direct_full_eval_log_path,
        direct_full_eval_rows,
        direct_full_eval_summary,
    )


@app.cell(hide_code=True)
def _(
    direct_full_eval_log_path,
    direct_full_eval_rows,
    direct_full_eval_summary,
    n_samples,
    seed,
):
    mo.vstack(
        [
            mo.md(f"""
    ### Direct full eval loop

    **Seed:** {seed}  
    **Requested samples:** {n_samples}  
    **Evaluated samples:** {direct_full_eval_summary["evaluated_samples"]}  
    **Skipped samples:** {direct_full_eval_summary["skipped_count"]}  
    **Binary sample pass rate:** {direct_full_eval_summary["binary_sample_pass_rate"]:.1%}  
    **Average test pass rate:** {direct_full_eval_summary["average_test_pass_rate"]:.3%}  
    **Run log:** `{direct_full_eval_log_path}`
    """),
            mo.ui.table(direct_full_eval_rows),
        ]
    )
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _():
    mo.md(r"""
    ### Encoder-Decoder Loop
    """)
    return


@app.cell(hide_code=True)
def _(ds, n_samples, run_full_encdec_eval_loop, seed):
    encdec_full_eval_outputs, encdec_full_eval_log_path = (
        run_full_encdec_eval_loop(
            ds,
            val_num=n_samples,
            seed=seed,
        )
    )
    encdec_full_eval_summary = summarize_full_eval_outputs(
        encdec_full_eval_outputs
    )

    encdec_full_eval_rows = [
        {
            "dataset_index": dataset_index,
            "task_id": output.get("task_id"),
            "skipped": output.get("skipped", False),
            "passed": output["pass_rate"] == 1.0,
            "test_pass_rate": output["pass_rate"],
            "error": output.get("error"),
            "log_file": output.get("log_file"),
        }
        for dataset_index, output in encdec_full_eval_outputs.items()
    ]
    return


@app.cell(hide_code=True)
def _(
    direct_full_eval_log_path,
    direct_full_eval_rows,
    direct_full_eval_summary,
    n_samples,
    seed,
):
    mo.vstack(
        [
            mo.md(f"""
    ### Direct full eval loop

    **Seed:** {seed}  
    **Requested samples:** {n_samples}  
    **Evaluated samples:** {direct_full_eval_summary["evaluated_samples"]}  
    **Skipped samples:** {direct_full_eval_summary["skipped_count"]}  
    **Binary sample pass rate:** {direct_full_eval_summary["binary_sample_pass_rate"]:.1%}  
    **Average test pass rate:** {direct_full_eval_summary["average_test_pass_rate"]:.3%}  
    **Run log:** `{direct_full_eval_log_path}`
    """),
            mo.ui.table(direct_full_eval_rows),
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


@app.cell(hide_code=True)
def _(direct_generator, ds, run_gen_eval, val_ids):
    if False:
        _ind = 2
        _out = run_gen_eval(
            ds.get_raw_sample_at_index(val_ids[_ind]),
            direct_generator,
            log_file=HUMAN_EVAL_DSPY_LOG_PATH,
        )
        display_direct_generation(ds.get_raw_sample_at_index(val_ids[_ind]), _out)
    return


@app.cell(hide_code=True)
def _(ds, encoder_decoder_generator, run_encdec_eval, val_ids):
    if False:
        _ind = 2
        _out = run_encdec_eval(
            ds.get_raw_sample_at_index(val_ids[_ind]),
            encoder_decoder_generator,
            log_file=HUMAN_EVAL_DSPY_LOG_PATH,
        )
        display_encdec_generation(ds.get_raw_sample_at_index(val_ids[_ind]), _out)
    return


if __name__ == "__main__":
    app.run()
