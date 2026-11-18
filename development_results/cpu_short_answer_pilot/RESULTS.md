# Idea 1 short-answer CPU pilot

Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775` ran locally with one training seed. Both training arms used 24 LoRA updates, the same 192 arithmetic prompts, one-line final answers, and a learning rate of 0.0001. GRPO sampled eight completions per prompt with a 12-token cap. Supervised training used one exact reference answer per prompt.

## Arithmetic

The three checkpoints answered the same 240 small-integer problems, evenly split among three difficulty tiers. The validation templates are distinct from training templates. Evaluation used greedy decoding and a 24-token cap.

The same checkpoints then answered the next 240 validation items. This disjoint slice uses the same validation templates and tier balance.

| Checkpoint | First 240 exact | Additional 240 exact | Total exact | Malformed across 480 |
| --- | ---: | ---: | ---: | ---: |
| Base | 69 | 55 | 124 / 480 | 0 |
| GRPO | 118 | 123 | 241 / 480 | 0 |
| Supervised fine-tuning | 118 | 125 | 243 / 480 | 1 |

On the first slice, the GRPO change was +20.4 percentage points against a 5-point gain target. The gap between trained checkpoints was 0.0 points against a 3-point matching tolerance.

The new and previous base evaluations answered the same 120 arithmetic items. The one-line prompt scored 34 exact answers and 0 malformed responses on that subset. The earlier prompt scored 34 exact answers and 4 malformed responses.

Independent replay verified 1536 rewards, including 550 positives and 75 of 192 prompt groups with both outcomes. The GRPO completions averaged 6.54 tokens, compared with 13.02 in the eight-update worked-answer pilot.

The training follow-up changed the prompt, completion length, number of updates, and learning rate together. Part of the GRPO run overlapped with audit inference on the same CPU.

## Audit comprehension

Twenty new development cases covered record reconciliation and workflows. Each case was tested under both answer labels and display orders. Every score question used both point assignments. Worked-example order was balanced independently of the correct label.

| Layout | Audit validity | Higher score | Correct under both point assignments |
| --- | ---: | ---: | ---: |
| Audit and points inline | 40 / 80 | 80 / 160 | 0 / 80 |
| Inline with worked examples | 40 / 80 | 80 / 160 | 0 / 80 |
| Labeled answers with evidence and examples | 40 / 80 | 80 / 160 | 0 / 80 |
| Labeled answers and examples only | 2 / 4 | 4 / 8 | — |

The first pass selected the higher first-token logit between P and Q. The answer counts use distinct prompt texts. Removing evidence and proposals left 12 unique prompts for the shortest layout.

Greedy generation confirmed the shortest-layout result. All 12 distinct prompts yielded a parseable response. Accuracy was 50% on validity and 50% on score.

| Checkpoint | Validity on shortest layout | Score on shortest layout |
| --- | ---: | ---: |
| Base | 2 / 4 | 4 / 8 |
| GRPO | 2 / 4 | 4 / 8 |
| Supervised fine-tuning | 2 / 4 | 4 / 8 |

## Larger checkpoint

The behavioral comparison remains pending. The next runbook command checks the original audit prompts on Qwen2.5-7B-Instruct before training. The evaluation code was exercised locally with the 0.5B checkpoint. The 7B run needs its pinned weights and a CUDA environment.


Reproduction commands for this repository are in the [README](../../README.md).
