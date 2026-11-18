# Local Qwen development checks

**11 November 2026 · Qwen2.5-0.5B-Instruct · CPU**

The local runs used the pretrained 0.5B revision 7ae557604adf67be50417f59c2c2f167def9a775. The arithmetic checks used the first 12 validation items in each curriculum, greedy decoding, and a 96-token completion cap. The standard and small-integer curricula have different problems. The three small-integer results below use the same 12 items. [Qwen model revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775)

| Checkpoint | Arithmetic curriculum | Exact answers |
| --- | --- | ---: |
| Base | Standard signed operands | 0 / 12 |
| Base | Small nonnegative operands | 5 / 12 |
| One GRPO update | Small nonnegative operands | 5 / 12 |
| One SFT update | Small nonnegative operands | 4 / 12 |

The two updates used the same eight prompts and seed 17. GRPO sampled 64 completions with a 64-token cap. Independent replay confirmed 21 positive rewards and six mixed-reward groups among eight. Both updates produced a nonzero gradient, and both saved adapters loaded for evaluation. This check establishes that the reward and adapter pipeline works on a pretrained checkpoint. Estimating a learning effect requires the longer run and full validation set.

## Four-update CPU pilot

Both arms used seed 17 and the same 32 small-integer training prompts. The comparison used the same 60 validation problems across checkpoints, with a 96-token completion cap.

| Checkpoint | Exact answers | Malformed answers |
| --- | ---: | ---: |
| Base | 18 / 60 | 2 / 60 |
| Four GRPO updates | 18 / 60 | 1 / 60 |
| Four SFT updates | 16 / 60 | 14 / 60 |

GRPO took 6 minutes 17 seconds on CPU. Independent replay verified 49 positive rewards among 256 completions, with mixed rewards on 11 of 32 prompts. The GRPO adapter solved the same 18 validation problems as the base model. SFT took 22 seconds, and nine of its validation generations reached the token cap. These are development results from one training seed. The raw responses and audit are in **development_results/cpu_pilot_4_updates/**. [CPU commands](RUNPOD.md#run-a-local-cpu-check)

## Behavioral comprehension

Each layout used the same 20 development cases and 40 separate comprehension questions. All prompts disclosed the audit status. The score and decision questions also displayed the points. The compact layouts put audit status and points on the same line as each option.

| Layout | Validity questions | Score questions | Total |
| --- | ---: | ---: | ---: |
| Original P/Q | 12 / 20 | 10 / 20 | 22 / 40 |
| Compact P/Q | 13 / 20 | 11 / 20 | 24 / 40 |
| Compact A/B | 10 / 20 | 10 / 20 | 20 / 40 |

The original prompt elicited P on all 20 score questions. The compact layout’s two extra correct answers still leave the 0.5B checkpoint below the 90% comprehension target. Measure the 1.5B checkpoint’s comprehension on development cases before interpreting its score-sensitive choices. Test both arithmetic curricula with one update to measure mixed rewards at that size.

Raw responses and calibration prompts are in **development_results/**. These pretrained 0.5B runs use arithmetic validation and behavioral development cases. The study’s behavioral holdout is reserved for the 1.5B comparison.
