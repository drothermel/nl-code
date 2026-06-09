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
    from datetime import datetime, timezone
    from pathlib import Path
    import dspy
    import pandas as pd

    from nl_code.code_analysis import extract_from_code_fences
    from nl_code.code_execution.models import CodeExecutionInfrastructureError
    from nl_code.datasets import HumanEvalDataset
    from nl_code.optim.dspy_generators import (
        DEFAULT_DSPY_MODEL,
        DEFAULT_OPENROUTER_BASE_URL,
        DEFAULT_REASONING_EFFORT,
        DirectCodeGenerator,
        EncoderDecoderCodeGenerator,
        configure_dspy_lm,
    )
    from nl_code.optim.humaneval_dspy_eval import (
        GenerationType,
        HumanEvalDspyEvalConfig,
        build_test_cases,
        dump_lm_history_since,
        evaluate_completed_code as evaluate_completed_code_from_library,
        lm_history_length,
        run_humaneval_dspy_eval,
        select_dataset_indices,
        _failed_eval_results as failed_eval_results_from_library,
        _timestamped_log_path as timestamped_log_path_from_library,
    )
    from nl_code.optim.humaneval_dspy_logs import (
        HumanEvalDspyAttemptSnapshot,
        load_humaneval_dspy_log_snapshot,
        parse_humaneval_dspy_logs,
        token_count,
    )

    NOTEBOOK_PATH = Path(__file__).resolve()
    REPO_ROOT = NOTEBOOK_PATH.parents[2]
    LOGS_DIR = REPO_ROOT / "logs"
    LOGS_DIR.mkdir(exist_ok=True)
    HUMAN_EVAL_DSPY_LOG_PATH = LOGS_DIR / (
        f"human_eval_dspy_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    )
    HUMAN_EVAL_DSPY_SNAPSHOT_PATH = (
        LOGS_DIR / "human_eval_dspy_snapshot_latest.json"
    )
    HUMAN_EVAL_DSPY_LOG_PATH.touch(exist_ok=True)


@app.cell
def _():
    # Configure with environment variables
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.getenv(
        "OPENROUTER_API_BASE",
        DEFAULT_OPENROUTER_BASE_URL,
    )

    # Set environment for LiteLLM when an API key is available.
    if OPENROUTER_API_KEY is not None:
        os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
    os.environ["OPENROUTER_API_BASE"] = OPENROUTER_BASE_URL

    model = DEFAULT_DSPY_MODEL
    reasoning_effort = DEFAULT_REASONING_EFFORT
    return OPENROUTER_API_KEY, OPENROUTER_BASE_URL, model, reasoning_effort


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


@app.cell
def _(OPENROUTER_API_KEY, OPENROUTER_BASE_URL, model, reasoning_effort):
    lm = None
    if OPENROUTER_API_KEY is not None:
        try:
            lm = configure_dspy_lm(
                model=model,
                api_key=OPENROUTER_API_KEY,
                api_base=OPENROUTER_BASE_URL,
                reasoning_effort=reasoning_effort,
            )
        except RuntimeError as exc:
            if "same async task" not in str(exc):
                raise
            # Live marimo sessions may already have DSPy configured from an earlier task.
            lm = dspy.settings.lm
    direct_generator = DirectCodeGenerator()
    encoder_decoder_generator = EncoderDecoderCodeGenerator()
    return (lm,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Parsed Eval Snapshot
    """)
    return


@app.cell(hide_code=True)
def _():
    if HUMAN_EVAL_DSPY_SNAPSHOT_PATH.exists():
        human_eval_log_snapshot = load_humaneval_dspy_log_snapshot(
            HUMAN_EVAL_DSPY_SNAPSHOT_PATH
        )
        snapshot_source = HUMAN_EVAL_DSPY_SNAPSHOT_PATH
    else:
        human_eval_log_snapshot = parse_humaneval_dspy_logs(LOGS_DIR)
        snapshot_source = LOGS_DIR

    mo.vstack(
        [
            mo.md(
                f"""
    Loaded **{len(human_eval_log_snapshot.runs)}** runs, **{len(human_eval_log_snapshot.attempts)}** attempts, **{len(human_eval_log_snapshot.failed_attempts)}** failed attempts, and **{len(human_eval_log_snapshot.generation_calls)}** generation calls from `{snapshot_source.relative_to(REPO_ROOT)}`.
    """
            ),
            mo.ui.table(
                pd.DataFrame.from_records(human_eval_log_snapshot.pipeline_rows()),
                page_size=10,
            ),
        ]
    )
    return (human_eval_log_snapshot,)


@app.cell(hide_code=True)
def _(human_eval_log_snapshot):
    eval_runs_df = pd.DataFrame.from_records(human_eval_log_snapshot.run_rows())
    eval_attempts_df = pd.DataFrame.from_records(
        human_eval_log_snapshot.attempt_rows()
    )
    generation_calls_df = pd.DataFrame.from_records(
        human_eval_log_snapshot.generation_call_rows()
    )

    if eval_attempts_df.empty:
        aggregate_pass_df = pd.DataFrame()
        failed_attempts_df = pd.DataFrame()
    else:
        evaluated_attempts_df = eval_attempts_df.loc[
            ~eval_attempts_df["skipped"].fillna(False)
        ].copy()
        aggregate_attempt_df = (
            evaluated_attempts_df.groupby("generation_type", dropna=False)
            .agg(
                evaluated_attempts=("passed", "size"),
                full_pass_attempts=("passed", "sum"),
                average_test_pass_rate=("pass_rate", "mean"),
                mean_failed_tests=("failed_test_count", "mean"),
            )
            .reset_index()
        )
        sample_best_df = (
            evaluated_attempts_df.groupby(
                ["generation_type", "task_id"], dropna=False
            )["passed"]
            .max()
            .reset_index()
            .groupby("generation_type", dropna=False)
            .agg(
                unique_samples=("task_id", "size"),
                best_of_n_passes=("passed", "sum"),
            )
            .reset_index()
        )
        aggregate_pass_df = aggregate_attempt_df.merge(
            sample_best_df,
            on="generation_type",
            how="left",
        )
        aggregate_pass_df["attempt_pass_rate"] = (
            aggregate_pass_df["full_pass_attempts"]
            / aggregate_pass_df["evaluated_attempts"]
        )
        aggregate_pass_df["best_of_n_sample_pass_rate"] = (
            aggregate_pass_df["best_of_n_passes"]
            / aggregate_pass_df["unique_samples"]
        )
        failed_attempts_df = (
            evaluated_attempts_df.loc[evaluated_attempts_df["pass_rate"] < 1.0]
            .sort_values(
                ["task_id", "generation_type", "timestamp", "run_log_name"],
                na_position="last",
            )
            .reset_index(drop=True)
        )
    return (
        aggregate_pass_df,
        eval_runs_df,
        failed_attempts_df,
        generation_calls_df,
    )


@app.cell(hide_code=True)
def _(aggregate_pass_df, eval_runs_df, generation_calls_df):
    if eval_runs_df.empty:
        log_summary_display = mo.md("No parsed HumanEval DSPy eval logs found.")
    else:
        generation_usage_display = (
            mo.md("_No generation call records found._")
            if generation_calls_df.empty
            else mo.ui.table(
                generation_calls_df[
                    [
                        "generation_log_name",
                        "record_index",
                        "prompt_kind",
                        "model",
                        "total_tokens",
                        "cost",
                    ]
                ].head(100),
                page_size=10,
            )
        )
        log_summary_display = mo.vstack(
            [
                mo.md("#### Run Summary"),
                mo.ui.table(eval_runs_df, page_size=10),
                mo.md("#### Aggregate Pass Rates"),
                mo.ui.table(aggregate_pass_df, page_size=10),
                # mo.md("#### Generation Call Usage"),
                # generation_usage_display,
            ]
        )

    log_summary_display
    return


@app.cell(hide_code=True)
def _(failed_attempts_df, human_eval_log_snapshot):
    failed_task_ids = human_eval_log_snapshot.failed_task_ids
    if not failed_task_ids:
        failed_task_control = None
        failed_task_selector_display = mo.md(
            "No failed tasks found in the parsed snapshot."
        )
    else:
        failed_task_control = mo.ui.dropdown(
            options=failed_task_ids,
            value=failed_task_ids[0],
            label="Failed task",
        )
        overview_columns = [
            "generation_type",
            "dataset_index",
            "task_id",
            "repeat_index",
            "pass_rate",
            "failed_test_count",
            "first_failed_error",
            "run_log_name",
        ]
        failed_task_selector_display = mo.vstack(
            [
                mo.md(f"#### Failed Tasks ({len(failed_task_ids)})"),
                failed_task_control,
                mo.ui.table(
                    failed_attempts_df[overview_columns].head(100),
                    page_size=10,
                    wrapped_columns=["first_failed_error", "run_log_name"],
                ),
            ]
        )

    failed_task_selector_display
    return (failed_task_control,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Helpers
    """)
    return


@app.function(hide_code=True)
def render_attempt_generation_calls(generation_calls):
    if not generation_calls:
        return mo.md("_No per-call LM history is available for this attempt._")
    return mo.vstack(
        [
            mo.accordion(
                {
                    f"{call.attempt.call_index if call.attempt else call.record_index}: {call.prompt_kind or 'unknown prompt'}": mo.vstack(
                        [
                            mo.md("##### Messages"),
                            mo.ui.code_editor(
                                json.dumps(call.messages, indent=2),
                                language="json",
                            ),
                            mo.md("##### Output"),
                            mo.ui.code_editor(
                                "\n\n".join(call.outputs),
                                language="text",
                            ),
                            mo.md("##### Usage"),
                            mo.ui.table(
                                pd.DataFrame(
                                    [
                                        {
                                            "model": call.model,
                                            "response_model": call.response_model,
                                            "prompt_tokens": token_count(
                                                call.usage,
                                                "prompt_tokens",
                                                "input_tokens",
                                            ),
                                            "completion_tokens": token_count(
                                                call.usage,
                                                "completion_tokens",
                                                "output_tokens",
                                            ),
                                            "total_tokens": token_count(
                                                call.usage,
                                                "total_tokens",
                                            ),
                                            "cost": call.cost,
                                        }
                                    ]
                                )
                            ),
                        ]
                    )
                }
            )
            for call in generation_calls
        ]
    )


@app.function(hide_code=True)
def render_encdec_attempt_sequence(sample, attempt, generation_calls=None):
    if attempt is None:
        return mo.md(
            "_No encoder-decoder attempt is available for this task._"
        )
    return mo.vstack(
        [
            mo.md("### Encoder-Decoder Generation"),
            attempt_metadata_table(attempt),
            mo.md("#### 1. Ground-truth code sent to encoder"),
            mo.ui.code_editor(sample.gt_solution, language="python"),
            mo.md("#### 2. Encoded natural-language specification"),
            mo.ui.code_editor(attempt.code_spec or "", language="markdown"),
            mo.md("#### 3. Raw LM calls logged for this attempt"),
            render_attempt_generation_calls(generation_calls or []),
            mo.md("#### 4. Function stub sent to decoder"),
            mo.ui.code_editor(sample.function_stub, language="python"),
            mo.md("#### 5. Raw decoded completion"),
            mo.ui.code_editor(attempt.raw_completed_code, language="python"),
            mo.md("#### 6. Extracted code evaluated"),
            mo.ui.code_editor(attempt.extracted_code, language="python"),
            mo.md("#### 7. Test execution results"),
            attempt_results_table(attempt),
        ]
    )


@app.function(hide_code=True)
def render_direct_attempt_sequence(sample, attempt, generation_calls=None):
    if attempt is None:
        return mo.md("_No direct attempt is available for this task._")
    return mo.vstack(
        [
            mo.md("### Direct Generation"),
            attempt_metadata_table(attempt),
            mo.md("#### 1. Prompt sent to the model"),
            mo.ui.code_editor(sample.source__prompt, language="python"),
            mo.md("#### 2. Raw LM calls logged for this attempt"),
            render_attempt_generation_calls(generation_calls or []),
            mo.md("#### 3. Raw model completion"),
            mo.ui.code_editor(attempt.raw_completed_code, language="python"),
            mo.md("#### 4. Extracted code evaluated"),
            mo.ui.code_editor(attempt.extracted_code, language="python"),
            mo.md("#### 5. Ground-truth reference"),
            mo.ui.code_editor(sample.gt_solution, language="python"),
            mo.md("#### 6. Test execution results"),
            attempt_results_table(attempt),
        ]
    )


@app.function(hide_code=True)
def attempt_metadata_table(attempt: HumanEvalDspyAttemptSnapshot):
    return mo.ui.table(
        pd.DataFrame(
            [
                {
                    "run": attempt.source_file.name,
                    "task_id": attempt.task_id,
                    "dataset_index": attempt.dataset_index,
                    "repeat_index": attempt.repeat_index,
                    "pass_rate": attempt.test_pass_rate,
                    "failed_tests": attempt.failed_test_count,
                    "error": attempt.error,
                }
            ]
        ),
        selection=None,
        wrapped_columns=["run", "error"],
    )


@app.function(hide_code=True)
def latest_attempt_for_generation(
    human_eval_log_snapshot,
    task_id: str,
    generation_type: str,
) -> HumanEvalDspyAttemptSnapshot | None:
    attempts = [
        attempt
        for attempt in human_eval_log_snapshot.attempts_for_task(task_id)
        if attempt.generation_type == generation_type
    ]
    if not attempts:
        return None
    return sorted(
        attempts,
        key=lambda attempt: (
            attempt.timestamp or datetime.min.replace(tzinfo=timezone.utc),
            attempt.run_id,
            attempt.repeat_index or 0,
        ),
    )[-1]


@app.function(hide_code=True)
def attempt_results_table(attempt: HumanEvalDspyAttemptSnapshot):
    results_df = pd.DataFrame.from_records(
        [result.model_dump() for result in attempt.test_case_results]
    )
    if results_df.empty:
        return mo.md("_No test case results recorded._")
    return mo.ui.table(
        results_df,
        page_size=10,
        wrapped_columns=["error", "compile_error"],
    )


@app.cell
def _():
    def load_human_eval_dspy_eval_logs(logs_dir):
        run_rows = []
        output_rows = []
        for _path in sorted(logs_dir.glob("human_eval_dspy_*_eval_*.json")):
            _payload = _read_eval_json(_path)
            if _payload is not None:
                _run_row, _output_rows = _legacy_eval_log_rows(_path, _payload)
                run_rows.append(_run_row)
                output_rows.extend(_output_rows)
        for _path in sorted(logs_dir.glob("human_eval_dspy_run_*.json")):
            _payload = _read_eval_json(_path)
            if _payload is not None:
                _run_row, _output_rows = _package_eval_log_rows(_path, _payload)
                run_rows.append(_run_row)
                output_rows.extend(_output_rows)

        return pd.DataFrame.from_records(run_rows), pd.DataFrame.from_records(
            output_rows
        )


    def load_human_eval_dspy_generation_history(logs_dir):
        rows = []
        for _path in sorted(logs_dir.glob("human_eval_dspy_*.jsonl")):
            for _record_index, _line in enumerate(
                _path.read_text(encoding="utf-8").splitlines()
            ):
                if not _line.strip():
                    continue
                try:
                    _record = json.loads(_line)
                except json.JSONDecodeError as _exc:
                    rows.append(
                        {
                            "generation_log_file": str(_path),
                            "generation_log_name": _path.name,
                            "record_index": _record_index,
                            "parse_error": str(_exc),
                        }
                    )
                    continue
                _usage = _record.get("usage") or {}
                rows.append(
                    {
                        "generation_log_file": str(_path),
                        "generation_log_name": _path.name,
                        "record_index": _record_index,
                        "timestamp": _record.get("timestamp"),
                        "model": _record.get("model"),
                        "response_model": _record.get("response_model"),
                        "cost": _record.get("cost"),
                        "prompt_tokens": _usage.get("prompt_tokens")
                        or _usage.get("input_tokens"),
                        "completion_tokens": _usage.get("completion_tokens")
                        or _usage.get("output_tokens"),
                        "total_tokens": _usage.get("total_tokens"),
                        "uuid": _record.get("uuid"),
                        "parse_error": None,
                    }
                )
        return pd.DataFrame.from_records(rows)


    def _read_eval_json(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None


    def _legacy_eval_log_rows(path, payload):
        outputs = payload.get("outputs") or []
        run_row = {
            "run_log_path": str(path),
            "run_log_name": path.name,
            "run_format": "notebook",
            "timestamp": payload.get("timestamp"),
            "generation_type": payload.get("eval_type"),
            "val_num": payload.get("val_num"),
            "seed": payload.get("seed"),
            "num_outputs": len(outputs),
        }
        output_rows = []
        for _item in outputs:
            _output = _item.get("output") or {}
            _prediction = _output.get("prediction") or {}
            _results = _output.get("results") or []
            _pass_rate = float(_output.get("pass_rate") or 0.0)
            _first_failed = _first_failed_result(_results)
            output_rows.append(
                _base_output_row(
                    path=path,
                    run_format="notebook",
                    timestamp=payload.get("timestamp"),
                    generation_type=payload.get("eval_type"),
                    seed=payload.get("seed"),
                    dataset_index=_item.get("dataset_index"),
                    task_id=_output.get("task_id"),
                    repeat_index=0,
                    skipped=bool(_output.get("skipped", False)),
                    pass_rate=_pass_rate,
                    error=_output.get("error"),
                    code_spec=_prediction.get("code_spec"),
                    completed_code=_prediction.get("completed_code", ""),
                    extracted_code=_output.get("extracted", ""),
                    generation_log_file=_output.get("log_file"),
                    results=_results,
                    first_failed=_first_failed,
                )
            )
        return run_row, output_rows


    def _package_eval_log_rows(path, payload):
        config = payload.get("config") or {}
        attempts = payload.get("attempts") or []
        run_row = {
            "run_log_path": str(path),
            "run_log_name": path.name,
            "run_format": "package",
            "timestamp": payload.get("timestamp"),
            "generation_type": config.get("generation_type"),
            "val_num": config.get("n_samples"),
            "seed": config.get("seed"),
            "num_outputs": len(attempts),
        }
        output_rows = []
        for _attempt in attempts:
            _results = _attempt.get("test_case_results") or []
            _pass_rate = float(_attempt.get("test_pass_rate") or 0.0)
            _first_failed = _first_failed_result(_results)
            output_rows.append(
                _base_output_row(
                    path=path,
                    run_format="package",
                    timestamp=payload.get("timestamp"),
                    generation_type=_attempt.get("generation_type"),
                    seed=config.get("seed"),
                    dataset_index=_attempt.get("dataset_index"),
                    task_id=_attempt.get("task_id"),
                    repeat_index=_attempt.get("repeat_index"),
                    skipped=bool(_attempt.get("skipped", False)),
                    pass_rate=_pass_rate,
                    error=_attempt.get("error"),
                    code_spec=_attempt.get("code_spec"),
                    completed_code=_attempt.get("raw_completed_code", ""),
                    extracted_code=_attempt.get("extracted_code", ""),
                    generation_log_file=_attempt.get("generation_log_file"),
                    results=_results,
                    first_failed=_first_failed,
                )
            )
        return run_row, output_rows


    def _base_output_row(
        *,
        path,
        run_format,
        timestamp,
        generation_type,
        seed,
        dataset_index,
        task_id,
        repeat_index,
        skipped,
        pass_rate,
        error,
        code_spec,
        completed_code,
        extracted_code,
        generation_log_file,
        results,
        first_failed,
    ):
        failed_count = sum(
            not bool(_result.get("passed", False)) for _result in results
        )
        return {
            "run_log_path": str(path),
            "run_log_name": path.name,
            "run_format": run_format,
            "timestamp": timestamp,
            "generation_type": generation_type,
            "seed": seed,
            "dataset_index": dataset_index,
            "task_id": task_id,
            "repeat_index": repeat_index,
            "skipped": skipped,
            "pass_rate": pass_rate,
            "passed": (not skipped) and pass_rate == 1.0,
            "error": error,
            "code_spec": code_spec,
            "completed_code": completed_code,
            "extracted_code": extracted_code,
            "generation_log_file": generation_log_file,
            "test_case_results": results,
            "test_count": len(results),
            "failed_test_count": failed_count,
            "first_failed_input": _json_cell_value(
                first_failed.get("input_value")
            ),
            "first_failed_expected": _json_cell_value(
                first_failed.get("expected_output")
            ),
            "first_failed_actual": _json_cell_value(
                first_failed.get("actual_output")
            ),
            "first_failed_error": first_failed.get("error")
            or first_failed.get("compile_error"),
        }


    def _first_failed_result(results):
        for _result in results:
            if not bool(_result.get("passed", False)):
                return _result
        return {}


    def _json_cell_value(value):
        if value is None:
            return None
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    return


@app.function
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


@app.function
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


@app.cell
def _(lm):
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
        history_start_index = lm_history_length(lm)
        output["prediction"] = generator(
            code_stub=data_sample.source__prompt,
        )
        output["log_file"] = dump_lm_history_since(
            lm,
            Path(log_file) if log_file is not None else None,
            start_index=history_start_index,
            attempt_metadata={
                "generation_type": "direct",
                "dataset_index": None,
                "task_id": data_sample.task_id,
                "repeat_index": 0,
            },
        )
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

    return


@app.cell
def _(lm):
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
        history_start_index = lm_history_length(lm)
        output["prediction"] = generator(
            input_code=data_sample.gt_solution,
            function_stub=data_sample.function_stub,
        )
        output["log_file"] = dump_lm_history_since(
            lm,
            Path(log_file) if log_file is not None else None,
            start_index=history_start_index,
            attempt_metadata={
                "generation_type": "encdec",
                "dataset_index": None,
                "task_id": data_sample.task_id,
                "repeat_index": 0,
            },
        )
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

    return


@app.function
def sample_dataset_indices(dataset, val_num: int, seed: int) -> list[int]:
    HumanEvalDspyEvalConfig(n_samples=val_num, seed=seed)
    return select_dataset_indices(
        dataset,
        n_samples=val_num,
        seed=seed,
    )


@app.function
def timestamped_eval_log_path(eval_type: str) -> Path:
    return timestamped_log_path_from_library(
        LOGS_DIR,
        f"{eval_type}_eval",
        suffix=".json",
    )


@app.cell
def _(lm):
    def output_from_attempt(attempt):
        prediction = {
            "code_spec": attempt.code_spec,
            "completed_code": attempt.raw_completed_code,
        }
        return {
            "prediction": prediction,
            "log_file": attempt.generation_log_file,
            "extracted": attempt.extracted_code,
            "results": attempt.test_case_results,
            "pass_rate": attempt.test_pass_rate,
            "skipped": attempt.skipped,
            "error": attempt.error,
            "task_id": attempt.task_id,
        }


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
        generation_type = GenerationType(eval_type)
        config = HumanEvalDspyEvalConfig(
            generation_type=generation_type,
            n_samples=val_num,
            seed=seed,
            output_dir=LOGS_DIR,
            generation_log_file=(
                Path(generation_log_file)
                if generation_log_file is not None
                else None
            ),
            run_log_file=Path(run_log_file) if run_log_file is not None else None,
        )
        run = run_humaneval_dspy_eval(
            config,
            dataset=dataset,
            direct_generator=(
                generator if generation_type == GenerationType.DIRECT else None
            ),
            encoder_decoder_generator=(
                generator if generation_type == GenerationType.ENCDEC else None
            ),
            lm=lm,
        )
        outputs = {
            attempt.dataset_index: output_from_attempt(attempt)
            for attempt in run.attempts
        }
        return outputs, run.run_log_file

    return


@app.function
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


@app.function
def extract_python_code(text: str) -> str:
    extracted_code, _had_fences = extract_from_code_fences(text)
    return extracted_code.rstrip() + "\n"


@app.function
def evaluate_completed_code(completed_code: str, sample):
    try:
        _extracted_code, results, pass_rate = (
            evaluate_completed_code_from_library(
                completed_code=completed_code,
                sample=sample,
            )
        )
    except CodeExecutionInfrastructureError as exc:
        error = str(exc)
        return failed_eval_results_from_library(
            build_test_cases(sample),
            error,
        ), 0.0
    return results, pass_rate


@app.function
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


@app.cell(column=1, hide_code=True)
def _(ds, failed_task_control, human_eval_log_snapshot):
    if failed_task_control is None:
        encdec_attempt = None
        encdec_generation_calls = []
        selected_sample = None
        failed_task_detail_display = mo.md(
            "Select a failed task to inspect the full direct and encoder-decoder sequences."
        )
    else:
        selected_task_id = failed_task_control.value
        selected_sample = ds.raw_samples.get(selected_task_id)
        if selected_sample is None:
            task_attempts = human_eval_log_snapshot.attempts_for_task(
                selected_task_id
            )
            selected_sample = ds.get_raw_sample_at_index(
                int(task_attempts[0].dataset_index)
            )
        direct_attempt = latest_attempt_for_generation(
            human_eval_log_snapshot,
            selected_task_id,
            "direct",
        )
        encdec_attempt = latest_attempt_for_generation(
            human_eval_log_snapshot,
            selected_task_id,
            "encdec",
        )
        direct_generation_calls = human_eval_log_snapshot.generation_calls_for_attempt(
            direct_attempt
        )
        encdec_generation_calls = human_eval_log_snapshot.generation_calls_for_attempt(
            encdec_attempt
        )
        failed_task_detail_display = mo.vstack(
            [
                mo.md(f"## Failed Case: `{selected_task_id}`"),
                mo.hstack(
                    [
                        render_direct_attempt_sequence(
                            selected_sample,
                            direct_attempt,
                            direct_generation_calls,
                        ),
                    ]
                ),
            ]
        )

    failed_task_detail_display
    return encdec_attempt, encdec_generation_calls, selected_sample


@app.cell(column=2, hide_code=True)
def _(encdec_attempt, encdec_generation_calls, selected_sample):
    (
        mo.md("Select a failed task to inspect the encoder-decoder sequence.")
        if selected_sample is None
        else render_encdec_attempt_sequence(
            selected_sample,
            encdec_attempt,
            encdec_generation_calls,
        )
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
