# HumanEval Enc-Dec Prompt Optimization Figures

This note documents the plotting scripts and figure outputs used to summarize
the effect of prompt optimization on the DSPy HumanEval encoder-decoder setup.

## Data Sources

The plots use two related but distinct evidence sources.

`ui/dspy-eval-static-viewer/exports/task_variation_full5x.csv` is the preferred
source for final presentation figures. It contains the full-5x saved-prompt
eval export for the GPT-5 nano HumanEval runs. Each row is one HumanEval task,
with baseline enc-dec pass rate and saved-prompt pass rates for:

- MIPRO encoder
- MIPRO decoder
- GEPA encoder
- GEPA decoder

This export does not include a both-stage enc-dec saved-prompt condition.

`data/humaneval-dspy-sample-performance/humaneval_sample_performance_evidence.csv`
contains a broader evidence table. The optimizer-summary plots filter this
table to GPT-5 nano enc-dec rows where:

```text
source kind = optimizer_summary_task_score
model name = openrouter/openai/gpt-5-nano
direct or enc-dec = enc-dec
phase = baseline or optimized
```

These optimizer-summary rows are useful for inspecting optimizer-internal
target comparisons, including both-stage optimization. They should not be
presented as equivalent to the full-5x saved-prompt eval export.

## Scripts

### Eval-only figures

Use this script for presentation plots based only on heldout eval-style saved
prompt results:

```bash
MPLBACKEND=Agg uv run --with matplotlib python \
  scripts/plot_humaneval_encdec_prompt_optimization_eval_only.py
```

The default input is:

```text
ui/dspy-eval-static-viewer/exports/task_variation_full5x.csv
```

The default output directory is:

```text
figures/humaneval-encdec-prompt-optimization-eval-only
```

The script writes both `.png` and `.svg` versions of each figure.

### Optimizer-summary figures

This script is retained for exploratory optimizer-summary comparisons:

```bash
MPLBACKEND=Agg uv run --with matplotlib python \
  scripts/plot_humaneval_encdec_prompt_optimization.py
```

The default output directory is:

```text
figures/humaneval-encdec-prompt-optimization
```

Use these figures when the question explicitly concerns optimizer-summary score
maps or target availability, especially the both-stage enc-dec condition. Do
not mix these numbers with eval-only figures without labeling the evidence
source.

## Eval-only Results

The eval-only script prints the main table:

```text
MIPRO encoder: baseline=0.854 saved_prompt=0.912 delta=+0.058 n=163
GEPA  encoder: baseline=0.854 saved_prompt=0.891 delta=+0.037 n=163
MIPRO decoder: baseline=0.854 saved_prompt=0.863 delta=+0.009 n=163
GEPA  decoder: baseline=0.854 saved_prompt=0.865 delta=+0.011 n=163
```

The safest interpretation is:

- Encoder optimization is the main lever in the enc-dec setup.
- MIPRO has the larger encoder gain in the full-5x saved-prompt eval.
- Decoder-only optimization is much smaller for both optimizers.
- The eval-only data does not support saying that GEPA made no progress. GEPA
  encoder improves over baseline, but less than MIPRO encoder.
- The eval-only data cannot evaluate both-stage enc-dec optimization because
  that saved-prompt condition is not present in the full-5x export.

## Eval-only Figure Inventory

`eval_only_mipro_target_delta_bars` compares MIPRO encoder versus MIPRO decoder
saved-prompt deltas. It is the cleanest single figure for the claim that the
encoder is the main optimization target.

`eval_only_mipro_target_dumbbell` shows baseline and saved-prompt mean pass
rates for the MIPRO encoder and decoder prompts. Use this when the absolute
performance levels matter more than just the delta.

`eval_only_mipro_target_delta_distribution` shows per-task MIPRO deltas for
encoder and decoder saved prompts. Use this for a technical audience when the
task-local nature of the gains is important.

`eval_only_encoder_method_bars` compares MIPRO encoder and GEPA encoder
saved-prompt deltas. Use this for the narrower claim that MIPRO has the larger
encoder gain.

`eval_only_method_target_heatmap` shows the 2x2 method-by-target matrix for
MIPRO/GEPA and encoder/decoder. This is the most compact figure for the
combined message: encoder beats decoder for both optimizers, and MIPRO encoder
is the strongest cell.

## Optimizer-summary Figure Inventory

`target_delta_bars` compares MIPRO encoder, decoder, and both-stage deltas from
optimizer-summary task scores. It supports the exploratory read that MIPRO
encoder improves, MIPRO decoder is flat, and MIPRO both-stage regresses in the
summary score maps.

`mipro_target_dumbbell` shows baseline and optimized mean pass rates for the
same MIPRO target comparison.

`mipro_target_delta_distribution` shows paired optimizer-summary row deltas for
MIPRO targets.

`encoder_method_bars` compares MIPRO encoder and GEPA encoder using
optimizer-summary task scores. In this evidence slice, MIPRO encoder improves
and GEPA encoder regresses.

`method_target_heatmap` shows the optimizer-summary method-by-target matrix.
It marks GEPA both-stage as low coverage because it has only three paired rows.

`encoder_evidence_source_comparison` compares encoder deltas from
optimizer-summary rows against full-5x saved-prompt eval rows. It is useful as
a cautionary figure: the GEPA encoder conclusion depends on evidence source.

## Recommended Presentation Pair

For presentation using eval scores only:

1. Use `eval_only_mipro_target_delta_bars` for "encoder optimization is the
   real lever."
2. Use `eval_only_method_target_heatmap` for "encoder optimization dominates
   decoder optimization, and MIPRO encoder is the strongest saved-prompt
   condition."

Avoid the stronger wording "GEPA made no progress" when using eval-only plots.
The full-5x eval data shows positive GEPA encoder movement; it is just smaller
than MIPRO encoder movement.
