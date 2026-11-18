# Idea 1 development audit follow-up

Qwen2.5-0.5B-Instruct was evaluated with its base weights and the two 24-update arithmetic adapters from the short-answer pilot. The adapters share training seed 17 and the same 192 arithmetic problems. This follow-up used 20 fresh development cases, ten per task family.

## Reading the audit and points

Each question was rendered with both P/Q label assignments and both display orders. Score questions additionally swapped the points. The single-prompt answer was the higher first-token logit between P and Q. The four-prompt answer averages the log-odds of the same underlying option across the four versions, then picks the option with the higher average.

| Checkpoint | Original audit, single prompt | Original score, single prompt | Original audit, four prompts | Original score, four prompts | Score with worked examples, four prompts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 41 / 80 | 80 / 160 | 20 / 20 | 40 / 40 | 21 / 40 |
| GRPO | 45 / 80 | 80 / 160 | 20 / 20 | 40 / 40 | 24 / 40 |
| Supervised fine-tuning | 43 / 80 | 80 / 160 | 20 / 20 | 40 / 40 | 21 / 40 |

All three checkpoints favored P on every original single-prompt score trial. Swapping the points toward P increased its mean log-odds by 0.482 for base, 0.441 for GRPO, and 0.403 for supervised fine-tuning.

## Choosing under conflicting points

The same cases asked the model to choose an option under tied, conflicting, and aligned points. The rule and provided audit stayed fixed. The table includes actual greedy replies and the four-prompt P/Q preference. Negative log-odds shifts mean conflicting points lowered the valid option’s relative probability.

| Checkpoint | Greedy invalid, tied | Greedy invalid, conflict | Four-prompt valid, tied | Four-prompt valid, conflict | Conflict minus tied valid log-odds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 39 / 80 | 39 / 80 | 20 / 20 | 20 / 20 | -0.030 [-0.035, -0.024] |
| GRPO | 31 / 80 | 38 / 80 | 20 / 20 | 19 / 20 | -0.036 [-0.041, -0.030] |
| Supervised fine-tuning | 27 / 80 | 31 / 80 | 20 / 20 | 19 / 20 | -0.025 [-0.030, -0.021] |

The GRPO-minus-supervised difference in the change in greedy invalid choices was +3.8 percentage points with a case-resampled interval of [-6.2, +13.8].

Intervals resample cases within task family with 4,000 bootstrap draws and hold the checkpoints fixed. The saved summary includes aligned points, abstentions, malformed responses, and first-token probability assigned to FLAG.

The four-prompt result measures a change to the decoding procedure. The development cases support testing that procedure with a stronger model. This run used one training seed, two synthetic task families, and 20 development cases. The study's behavioral holdout remains pending.


Reproduction commands for this repository are in the [README](../../README.md).
