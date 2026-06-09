# Changelog

## 0.7.0 - 2026-05-17

This release formalizes the current task schema as `v3` and bumps parsed dataset caches to schema version 3.

- Redesigned derived `Task` objects around `target: TaskTarget` and `source: TaskSource` instead of flat `entry_point_name`, `description`, and `gt_solution` fields.
- Nested raw-task `source` objects replace flat `source__...` serialized fields across HumanEval, Pro, and ClassEval task families.
- Added HumanEval serialized test suites (`HumanEvalTest`, `HumanEvalTestCase`) with per-case source rewriting and `HumanEvalDataset.get_test_cases_at_index()`.
- Added shared Pro-task modeling in `pro_task.py` and `GTSolution` for HumanEval ground-truth execution.
- Expanded `code_parsing.py` with AST span editing and test-list parsing helpers.
- Migrated HumanEval DSPy eval/optimize workflows to v3 sample accessors in `humaneval_dspy_sample.py`.
- Restored HumanEval `function_stub` semantics: docstrings stripped for enc/dec decoder inputs while `code_stub` keeps the full prompt.
- Unified DSPy LM configuration around `resolve_dspy_lm_settings()` and catalog-first default `DEFAULT_LLM_CONFIG_ID`.
- Removed OpenRouter catalog id `openrouter/openai/gpt-oss-20b/low/v1`.
- Removed `Task.description`, `DatasetSlice.get_code_stub()`, and `DatasetSlice.get_code_stub_with_comments()`.
- Rebuilt dataset caches against schema version 3.

### Migration notes (v2/v1 → v3)

| Removed | Replacement |
| --- | --- |
| `Task.entry_point_name` | `Task.target.name` |
| `Task.description` | Family-specific raw fields (for example `raw.description` on Pro tasks) |
| `Task.gt_solution` / flat `gt_solution` on derived tasks | `Task.source.code` or `raw.gt_solution.code` |
| `DatasetSlice.get_code_stub()` | `DatasetSlice.get_official_prompt()` or `raw.function_stub` |
| `DatasetSlice.get_code_stub_with_comments()` | `raw.code_stub` or `raw.source.prompt` |
| Flat `source__*` serialized fields | Nested `source.*` on raw task models |

## 0.6.0 - 2026-04-16

This release formalizes the current task schema as `v2` and defines the prior raw-task/task design as `v1`.

- Added `version: Literal["v1", "v2"] = "v2"` to raw task models and derived `Task` objects.
- Added task-level validation that rejects derived tasks whose `version` does not match the raw task they were built from.
- Propagated raw task versions into dataset `_to_task()` conversions so version metadata stays explicit instead of relying on a default.
- Redesigned the HumanEval, HumanEval-Pro, MBPP-Pro, BigCodeBench-Lite-Pro, and ClassEval raw task models around preserved `source__...` inputs plus richer derived prompt/code artifacts.
- Standardized commented vs stripped code semantics so plain exported fields remove comments and docstrings, while explicit `with_comments` or `with_docstrings_and_comments` fields preserve them.
- Updated Pro-task models with explicit original/new function views, docstring-and-comment extraction fields, and separated original/new prompt and stub artifacts.
- Added `new_code_stub` and `new_code_stub_with_comments` aliases across task families so prompt/stub access is named consistently.
- Standardized official prompt generation:
  - Pro tasks now expose instruction-wrapped `original_official_prompt` and `new_official_prompt` fields.
  - HumanEval-plus `official_prompt` / `new_official_prompt` now use an instruction wrapper around the source prompt.
  - ClassEval `new_official_prompt` now uses the class name plus official skeleton and wraps the code in a fenced `python` block.
- Wrapped code content inside all `new_official_prompt` fields with fenced `python` code blocks.
- Split runnable and commented ground-truth code consistently across task families, and kept rebuild-time validation on the self-contained commented forms where required.
- Updated Pro-task ground-truth assembly so original and self-invoking functions are separated by two blank lines in both stripped and comment-preserving forms.
- Added validation notebooks for the redesigned MBPP-Pro, BigCodeBench-Lite-Pro, and ClassEval task models to mirror the HumanEval inspection workflow.
- Extended `DatasetSlice` with `get_official_prompt()`, `get_code_stub()`, and `get_code_stub_with_comments()` accessors parallel to `get_source_code()`.
- Rebuilt dataset caches against the redesigned task schemas.
