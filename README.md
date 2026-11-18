# Clean RL and reward hacking

Can reinforcement learning on correctly graded arithmetic change a language model's willingness to violate an explicit instruction when doing so earns more points on a new task?

This repository contains the experiment code, synthetic datasets, raw development results, reward audits, and two trained LoRA adapters. The project follows Idea 1 in Owain Evans's [Three half-baked research ideas](https://www.lesswrong.com/posts/vAniKbagqhyzpha6r/three-half-baked-research-ideas).

The completed experiments use Qwen2.5-0.5B-Instruct on CPU, with one paired training seed. Larger-model experiments, training-seed replication, and the main behavioral holdout evaluation are pending.

## Development results

Both training arms started from revision `7ae557604adf67be50417f59c2c2f167def9a775` and used the same 192 arithmetic prompts. GRPO used exact-answer rewards. Supervised fine-tuning used verified answers. Each arm ran 24 LoRA updates.

| Checkpoint | Exact arithmetic answers |
| --- | ---: |
| Base | 124 / 480 |
| GRPO | 241 / 480 |
| Supervised fine-tuning | 243 / 480 |

The 480 validation problems came from two disjoint slices with the same held-out templates. Independent replay verified all 1,536 logged GRPO rewards. [Arithmetic pilot](development_results/cpu_short_answer_pilot/RESULTS.md)

The behavioral pilots identified sensitivity to answer labels, dashboard order, and the presence of a suggested wrong answer.

- On the original diagnostic prompts, all three checkpoints selected P on every score question. Averaging predictions over four label and display-order permutations recovered 40/40 score answers and 20/20 audit answers per checkpoint on 20 fresh development cases. [Label and order analysis](development_results/cpu_score_sensitivity/RESULTS.md)
- The direct-response pilot collected 1,440 submissions across 80 development cases. Naming the wrong answer at zero points increased its submission rate by 13.75–23.75 percentage points across checkpoints. Increasing its reward from zero to one added 5–6.25 points. [Direct-response analysis](development_results/cpu_direct_response/RESULTS.md)
- The combined GRPO-minus-supervised difference in the direct-response point effect was +1.25 percentage points, with a case-bootstrap 95% interval of −3.75 to +7.50 points. Its direction changed between the two development samples. The causal question remains open.

These intervals condition on the trained checkpoints. They cover variation across the sampled development cases. Training variability requires additional seeds.

## Check the saved results

Use Python 3.12 for the complete project. The offline tests and saved-result replay use the standard library.

```sh
python3.12 -m unittest -q
python3.12 verify_results.py
```

The replay checks regenerate the synthetic arithmetic data, verify reward records, recalculate math accuracy, rerun the label-bias analysis, and replay the direct-response comparisons. It leaves the model weights unloaded.

## Reproduce the arithmetic pilot

Install PyTorch 2.14.0 for your platform and the packages in [requirements-runpod.txt](requirements-runpod.txt). The base checkpoint must be available in the local Hugging Face cache for offline inference.

```sh
model=Qwen/Qwen2.5-0.5B-Instruct
revision=7ae557604adf67be50417f59c2c2f167def9a775
hf download "$model" --revision "$revision"
python3.12 qwen.py prepare generated-small --arithmetic-profile small-integers --answer-format short
export OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
for arm in grpo sft; do
  python3.12 qwen.py train "$arm" generated-small "runs/$arm" --steps 24 --seed 17 --micro-batch 4 --learning-rate 0.0001 --max-completion-tokens 12 --model "$model" --revision "$revision" --device cpu --offline
done
for arm in base grpo sft; do
  set --
  if test "$arm" != base; then set -- --adapter "runs/$arm/adapter"; fi
  python3.12 qwen.py evaluate validation generated-small "runs/${arm}_validation" --limit 240 --batch-size 8 --max-new-tokens 24 --model "$model" --revision "$revision" --device cpu --offline "$@"
done
python3.12 qwen.py audit generated-small runs/grpo
```

The final saved adapters are under `development_results/cpu_short_answer_pilot/{grpo,sft}/adapter/`. Each export contains the exact trained weight file and its PEFT configuration. The base checkpoint supplies the tokenizer. Historical evaluation metadata retains the fingerprints of the original full adapter directories.

## Reproduce direct submissions

These commands use the included final adapters and reproduce the first 40-case development sample.

```sh
model=Qwen/Qwen2.5-0.5B-Instruct
revision=7ae557604adf67be50417f59c2c2f167def9a775
python3.12 qwen.py prepare-direct runs/direct-data --dashboard bonus --seed 20261123
for arm in base grpo sft; do
  set --
  if test "$arm" != base; then set -- --adapter "development_results/cpu_short_answer_pilot/$arm/adapter"; fi
  python3.12 qwen.py evaluate direct runs/direct-data "runs/direct-$arm" --model "$model" --revision "$revision" --device cpu --offline --batch-size 8 "$@"
done
python3.12 qwen.py compare-direct runs/direct-data runs/direct-base runs/direct-grpo runs/direct-sft --output runs/direct-comparison.json
```

The scorer accepts every workflow satisfying the rules. It records valid responses, the specific invalid dashboard target, other invalid responses, abstentions, and malformed responses. Bonus-rule prompts compare an unmentioned target, a named target worth zero, and the same named target worth one point. Two-row prompts additionally counterbalance dashboard order.

## Project files

| Path | Contents |
| --- | --- |
| `arithmetic.py` | Synthetic arithmetic and independent answer verification |
| `qwen.py` | Preparation, training, evaluation, and analysis commands |
| `qwen_train.py` | GRPO, supervised training, and reward replay |
| `qwen_local_diagnostics.py` | Audit comprehension and actual generated replies |
| `qwen_score_sensitivity.py` | Paired label-probability and permutation analysis |
| `direct_response.py` | Direct ledger and workflow submissions |
| `qwen_direct_analysis.py` | Paired score effects and display controls |
| `development_results/` | Raw pilot responses, summaries, and final adapters |
| `experiment_data/` | Frozen standard curriculum and the pending behavioral holdout |
| `prepared/` | Prepared Claude development requests |

[RUNPOD.md](RUNPOD.md) covers the larger-model preflight and planned main study. [DEVELOPMENT.md](DEVELOPMENT.md) records the early local checks. The [Claude pilot instructions](docs/CLAUDE_PILOT.md) describe the optional API workflow. The repository includes prepared Claude inputs. API responses have yet to be collected.

Machine-specific paths in the published metadata are normalized to repository-relative paths or an `HF_CACHE/` placeholder. Dataset contents and model responses retain their recorded values.
