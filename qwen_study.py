from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pilot import parse_jsonl, run_lock, sha256, verified_file
from qwen_analysis import compare_holdout, select_validation, write_analysis
from qwen_model import model_reference


MAIN_SEEDS = (17, 29, 43)


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def study(args) -> None:
    data, output = args.data, args.output
    pilot = args.profile == "pilot"
    steps = args.steps if args.steps is not None else (64 if pilot else 512)
    seeds = (17,) if pilot else MAIN_SEEDS
    reference = model_reference(args.model, args.revision)
    train_data = verified_file(data, "math_train.jsonl")
    if not 0 < steps * 8 <= len(parse_jsonl(train_data)):
        raise ValueError("The study needs eight arithmetic training prompts for each update.")
    if args.batch_size < 1 or args.max_completion_tokens < 1:
        raise ValueError("Batch size and completion length must be positive.")
    train_hash = sha256(train_data)
    cli = [sys.executable, str(Path(__file__).with_name("qwen.py"))]
    model_args = ["--model", args.model, "--device", args.device]
    if args.revision:
        model_args.extend(["--revision", args.revision])
    if args.offline:
        model_args.append("--offline")

    def evaluate(task: str, target: Path, adapter: Path | None = None, source: Path = data) -> None:
        command = [*cli, "evaluate", task, str(source), str(target), *model_args, "--batch-size", str(args.batch_size)]
        if adapter is not None:
            command.extend(["--adapter", str(adapter)])
        subprocess.run(command, check=True)

    def fit(arm: str, seed: int, target: Path) -> None:
        if (target / "training_summary.json").is_file():
            config = read_json(target / "training_config.json")
            expected = {"arm": arm, "seed": seed, "steps": steps, "model": reference, "device": args.device,
                        "max_completion_tokens": args.max_completion_tokens, "training_corpus_sha256": train_hash}
            if any(config[key] != value for key, value in expected.items()):
                raise ValueError(f"Existing training run uses different settings at {target}.")
            return
        command = [*cli, "train", arm, str(data), str(target), *model_args, "--seed", str(seed), "--steps", str(steps),
                   "--max-completion-tokens", str(args.max_completion_tokens)]
        if (target / "training_config.json").is_file():
            command.append("--resume")
        subprocess.run(command, check=True)

    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output / "study.lock", "a study run"):
        base_validation = output / "base_validation"
        evaluate("validation", base_validation)
        if pilot:
            evaluate("development", output / "base_development", source=args.development)
        selections = []
        for seed in seeds:
            seed_path = output / f"seed-{seed}"
            grpo, sft = seed_path / "grpo", seed_path / "sft"
            for arm, target in (("grpo", grpo), ("sft", sft)):
                fit(arm, seed, target)
            grpo_validation = seed_path / "grpo_validation"
            evaluate("validation", grpo_validation, grpo / "adapter")
            candidates = [sft / "adapter"]
            if not pilot:
                intermediate = [path for path in sft.glob("checkpoint-*") if (path / "adapter_model.safetensors").is_file()]
                candidates.extend(sorted(path for path in intermediate if int(path.name.removeprefix("checkpoint-")) < steps))
            sft_validation = [seed_path / f"sft_validation_{candidate.name}" for candidate in candidates]
            for candidate, target in zip(candidates, sft_validation):
                evaluate("validation", target, candidate)
            selection = select_validation(data, base_validation, grpo_validation, sft_validation)
            write_analysis(seed_path / "selection.json", selection)
            selections.append(selection)

        if pilot:
            training = read_json(output / "seed-17" / "grpo" / "training_summary.json")
            report = {
                "profile": "pilot", "selection": selections[0], "reward_audit": training["reward_audit"],
                "grpo_seconds_per_update": training["metrics"]["train_runtime"] / steps,
                "base_development": read_json(output / "base_development" / "summary.json"),
            }
        else:
            for task in ("test", "holdout", "diagnostics"):
                evaluate(task, output / f"base_{task}")
            comparisons = []
            for selection in selections:
                seed_path = output / f"seed-{selection['seed']}"
                adapters = {"grpo": selection["grpo_adapter"], "sft": selection["selected_sft"]["adapter"]}
                for arm, adapter in adapters.items():
                    for task in ("test", "holdout", "diagnostics"):
                        evaluate(task, seed_path / f"{arm}_{task}", Path(adapter["path"]))
                comparison = compare_holdout(data, output / "base_holdout", seed_path / "grpo_holdout",
                                             seed_path / "sft_holdout", seed_path / "selection.json")
                write_analysis(seed_path / "comparison.json", comparison)
                comparisons.append(comparison)
            report = {"profile": "main", "selections": selections, "seed_comparisons": comparisons}
        write_analysis(output / "study_summary.json", report)
