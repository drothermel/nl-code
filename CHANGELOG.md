# Changelog

## 0.6.0 - 2026-04-16

This release formalizes the current task schema as `v2` and defines the prior raw-task/task design as `v1`.

- Added `version: Literal["v1", "v2"] = "v2"` to raw task models and derived `Task` objects.
- Added task-level validation that rejects derived tasks whose `version` does not match the raw task they were built from.
- Propagated raw task versions into dataset `_to_task()` conversions so version metadata stays explicit instead of relying on a default.
- Redesigned the HumanEval, HumanEval-Pro, MBPP-Pro, BigCodeBench-Lite-Pro, and ClassEval raw task models around preserved `source__...` inputs plus richer derived prompt/code artifacts.
- Standardized commented vs stripped code semantics so plain exported fields remove comments and docstrings, while explicit `with_comments` or `with_docstrings_and_comments` fields preserve them.
- Updated Pro-task models with explicit original/new function views, docstring-and-comment extraction fields, and separated original/new prompt and stub artifacts.
- Split runnable and commented ground-truth code consistently across task families, and kept rebuild-time validation on the self-contained commented forms where required.
- Added validation notebooks for the redesigned MBPP-Pro, BigCodeBench-Lite-Pro, and ClassEval task models to mirror the HumanEval inspection workflow.
- Rebuilt dataset caches against the redesigned task schemas.
