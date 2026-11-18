# Idea 1 local Qwen pilot

Qwen2.5-0.5B-Instruct, pinned revision `7ae557604adf67be50417f59c2c2f167def9a775`. Both training arms started from the base model and ran eight LoRA updates on the same 64 small-integer arithmetic prompts. They used seed 17 and the same LoRA targets. GRPO sampled eight completions per prompt. Supervised fine-tuning used one verified worked solution per prompt.

## Arithmetic

All three checkpoints answered the same 120 validation questions from 16 templates excluded from training. There were 40 questions per difficulty tier. Decoding was greedy with a 96-token cap. Scoring required an exact integer in the expected final-answer format.

| Checkpoint | Exact answers | Malformed | Hit 96-token cap | Gained / lost vs. base |
| --- | ---: | ---: | ---: | ---: |
| Base | 34 / 120 | 4 | 0 | — |
| GRPO | 33 / 120 | 3 | 0 | 0 / 1 |
| Supervised fine-tuning | 33 / 120 | 11 | 6 | 0 / 1 |

GRPO changed exact-answer accuracy by -0.8 percentage points against the pilot's +5-point target. The gap between the two trained checkpoints was 0.0 points against a 3-point matching tolerance.

The SFT responses that reached 96 tokens were rerun with a 192-token cap. 6 of 6 preserved the original output prefix. 0 produced an exact answer at the higher cap, and 0 still reached the limit.

Independent replay verified 512 GRPO rewards. 85 were positive. 25 of 64 prompt groups contained both positive and zero rewards.

Measured training times were 387 seconds for GRPO and 21 seconds for supervised fine-tuning. The first 60 questions match the earlier four-update pilot's validation subset.

## Audit comprehension

The base checkpoint answered two separate questions about each of 20 development cases. One asked which option passed the audit and the other asked which option received more dashboard points. Every combination of valid label and display order was tested. Each score question was tested with points favoring each option in turn. The higher first-token logit among P and Q determined the answer.

| Prompt layout | Audit validity | Higher score | Both score assignments correct |
| --- | ---: | ---: | ---: |
| Original | 43 / 80 | 80 / 160 | 0 / 80 |
| Audit and points inline | 40 / 80 | 81 / 160 | 1 / 80 |
| Audit and points only | 2 / 4 | 5 / 8 | — |

The original layout kept the same choice after the point values swapped in 80 of 80 pairs. The inline layout kept the same choice in 79 of 80 pairs.

Validity and score counts use distinct prompt texts. The audit-only prompts collapse to four unique validity texts and eight unique score texts after removing case evidence and proposals. The comprehension target was 90% on each diagnostic. Logit comparisons measure a forced two-option choice.

## Scope

This development pilot uses one training seed and a 120-question validation subset. The behavioral work measures comprehension on development prompts. The behavioral holdout and the training-induced choice comparison remain pending.


Reproduction commands for this repository are in the [README](../../README.md).
