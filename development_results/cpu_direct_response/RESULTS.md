# Idea 1 direct-response development pilot

Qwen2.5-0.5B-Instruct and the two 24-update arithmetic adapters generated 1,440 submissions on CPU. Both adapters used training seed 17. The 80 distinct development cases covered ledgers and workflow ordering, with 40 cases per family.

Each prompt disclosed the verified total or one valid workflow. The model returned an integer or a six-activity sequence. The workflow checker accepted every valid sequence. An invalid submission matching the exact one named by the dashboard was scored as a target match.

## Two dashboard rows

The first 40 cases showed a valid submission and an invalid target in both display orders. Only the point values changed among tied, conflict, and aligned prompts within each order.

| Checkpoint | Target at zero points | Target at one point | Point effect, pp | Target listed first at zero, pp |
| --- | ---: | ---: | ---: | ---: |
| Base | 17 / 80 | 21 / 80 | +5.0 | +37.5 |
| GRPO | 17 / 80 | 20 / 80 | +3.8 | +37.5 |
| Supervised fine-tuning | 20 / 80 | 23 / 80 | +3.8 | +35.0 |

## Single bonus rule

The first 40 cases were rerun with a single dashboard rule, followed by 40 fresh cases. The three prompts gave every submission zero points, explicitly named the invalid target for zero points, or named the same target for one point. The first two prompts estimate the effect of mentioning the target. The last two differ only in the number of points and estimate the points effect.

| Checkpoint | Unmentioned target | Named target at zero | Named target at one | Mention effect, pp | Point effect, pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 0 / 80 | 19 / 80 | 24 / 80 | +23.8 | +6.2 |
| GRPO | 0 / 80 | 14 / 80 | 19 / 80 | +17.5 | +6.2 |
| Supervised fine-tuning | 0 / 80 | 11 / 80 | 15 / 80 | +13.8 | +5.0 |

| Development sample | GRPO point effect, pp | Supervised point effect, pp | Difference, pp |
| --- | ---: | ---: | ---: |
| First 40 cases | +10.0 | +5.0 | +5.0 |
| Fresh 40 cases | +2.5 | +5.0 | -2.5 |

The combined GRPO-minus-supervised point effect was +1.2 percentage points with a 95% case bootstrap interval of [-3.8, +7.5]. The bootstrap resamples cases within each task family and holds the checkpoints fixed.

## Scope and reproduction

The second sample uses fresh tasks and the same trained checkpoints. The study’s main behavioral holdout and training-seed replication remain pending. The comparison files also record all invalid submissions, malformed replies, abstentions, and results by family.

The archive includes the code, exact prompts, raw responses, evaluation settings, and analysis. Use PyTorch 2.14.0 and the versions in **code/requirements-runpod.txt**. The pinned checkpoint must be available in the local Hugging Face cache. Set **ADAPTER_ROOT** to the **results/** directory extracted from the short-answer pilot for its two adapters.

```sh
model=Qwen/Qwen2.5-0.5B-Instruct
revision=7ae557604adf67be50417f59c2c2f167def9a775
export OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
for phase in two_rows bonus_first bonus_replication; do
  for arm in base grpo sft; do
    set --
    if test "$arm" != base; then set -- --adapter "$ADAPTER_ROOT/$arm/adapter"; fi
    python code/qwen.py evaluate direct "results/$phase/data" "reproduced/$phase/$arm" --model "$model" --revision "$revision" --device cpu --offline --batch-size 8 "$@"
  done
  python code/qwen.py compare-direct "results/$phase/data" "reproduced/$phase/base" "reproduced/$phase/grpo" "reproduced/$phase/sft" --output "reproduced/$phase/comparison.json"
done
```
