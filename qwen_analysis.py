from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from pilot import atomic_write, json_bytes, parse_jsonl, sha256, verified_file
from qwen_data import BEHAVIOR_FILE, verified_math
from qwen_eval import score_math
from scoring import index_results, observation


EVAL_SETTINGS = ("model", "greedy", "device", "batch_size", "max_new_tokens", "packages")
TRAIN_SETTINGS = ("seed", "steps", "training_corpus_sha256", "used_prompt_ids_sha256", "lora")


def read_evaluation(directory: Path, task: str, digest: str, ids: set[str]) -> tuple[dict, list[dict]]:
    config = json.loads((directory / "evaluation_config.json").read_bytes())
    results = parse_jsonl((directory / "results.jsonl").read_bytes())
    if config["task"] != task or config["dataset_sha256"] != digest:
        raise ValueError(f"Evaluation dataset does not match {directory}.")
    if not config["full_dataset"] or config["planned"] != len(ids) or len(results) != len(ids):
        raise ValueError(f"Evaluation is incomplete at {directory}.")
    indexed = index_results(results, ids)
    if any(result["type"] != "succeeded" for result in indexed.values()):
        raise ValueError(f"Evaluation contains missing model responses at {directory}.")
    return config, results


def require_comparable(reference: dict, candidate: dict) -> None:
    for key in EVAL_SETTINGS:
        if reference[key] != candidate[key]:
            raise ValueError(f"Incompatible evaluation settings for {key}.")


def require_arm(config: dict, arm: str) -> None:
    if arm == "base":
        if config["adapter"] is not None:
            raise ValueError("The base evaluation must use the unmodified checkpoint.")
    elif config.get("training", {}).get("arm") != arm:
        raise ValueError(f"Evaluation must identify its {arm} training run.")


def require_training_pair(grpo: dict, sft: dict) -> None:
    for key in TRAIN_SETTINGS:
        if grpo["training"][key] != sft["training"][key]:
            raise ValueError(f"The training pair differs in {key}.")


def adapter_step(config: dict) -> int:
    name = Path(config["adapter"]["path"]).name
    return int(name.removeprefix("checkpoint-")) if name.startswith("checkpoint-") else config["training"]["steps"]


def select_validation(data: Path, base: Path, grpo: Path, sft: list[Path]) -> dict:
    filename = "math_validation.jsonl"
    rows = verified_math(data, "validation")
    digest = sha256(verified_file(data, filename))
    ids = {row["case_id"] for row in rows}

    def validation(directory, arm):
        config, results = read_evaluation(directory, "validation", digest, ids)
        require_arm(config, arm)
        return config, score_math(rows, results)["all"]["accuracy"]

    base_config, base_accuracy = validation(base, "base")
    grpo_config, grpo_accuracy = validation(grpo, "grpo")
    require_comparable(base_config, grpo_config)
    candidates = []
    for path in sft:
        config, accuracy = validation(path, "sft")
        require_comparable(grpo_config, config)
        require_training_pair(grpo_config, config)
        candidates.append({
            "evaluation": str(path.resolve()), "adapter": config["adapter"],
            "step": adapter_step(config), "accuracy": accuracy,
            "absolute_accuracy_gap": abs(accuracy - grpo_accuracy),
        })
    selected = min(candidates, key=lambda row: (round(row["absolute_accuracy_gap"] * len(rows)), row["step"], row["evaluation"]))
    return {
        "model": grpo_config["model"], "seed": grpo_config["training"]["seed"],
        "validation_dataset_sha256": digest, "n": len(rows),
        "base_accuracy": base_accuracy, "grpo_accuracy": grpo_accuracy, "grpo_adapter": grpo_config["adapter"],
        "grpo_accuracy_gain": grpo_accuracy - base_accuracy,
        "meets_five_point_gain_target": grpo_accuracy - base_accuracy >= 0.05 - 1e-12,
        "selected_sft": selected, "sft_within_three_points": selected["absolute_accuracy_gap"] <= 0.03 + 1e-12,
        "sft_candidates": candidates,
    }


def interval(draws: list[float]) -> list[float]:
    draws.sort()
    return [draws[int((len(draws) - 1) * probability)] for probability in (0.025, 0.975)]


def paired_contrast(design: list[dict], rl: dict[str, dict], control: dict[str, dict], audit: str) -> dict:
    pairs = defaultdict(dict)
    for row in design:
        condition = row["condition"]
        if row["audit"] == audit and condition in {"tied", "conflict"}:
            pair = tuple(int(observation(row, group[row["custom_id"]])["outcome"] == "invalid") for group in (rl, control))
            pairs[(row["family"], row["template"], row["case_id"])][condition] = pair

    clusters = defaultdict(lambda: defaultdict(list))
    for (family, template, _), conditions in pairs.items():
        rl_tied, control_tied = conditions["tied"]
        rl_conflict, control_conflict = conditions["conflict"]
        effect = rl_conflict - rl_tied - control_conflict + control_tied
        clusters[family][template].append((effect, rl_conflict, rl_tied, control_conflict, control_tied))

    family_results = {}
    rng = random.Random(20261111)
    family_draws = []
    for family, templates in sorted(clusters.items()):
        records = [row for cluster in templates.values() for row in cluster]
        deltas, rl_conflict, rl_tied, control_conflict, control_tied = map(mean, zip(*records))
        summaries = [(sum(row[0] for row in cluster), len(cluster)) for cluster in templates.values()]
        draws = []
        for _ in range(2000):
            total, n = map(sum, zip(*rng.choices(summaries, k=len(summaries))))
            draws.append(total / n)
        family_draws.append(draws)
        family_results[family] = {
            "cases": len(records), "templates": len(templates), "effect": deltas,
            "rl_conflict": rl_conflict, "rl_tied": rl_tied,
            "control_conflict": control_conflict, "control_tied": control_tied,
            "absolute_conflict_rate_change": rl_conflict - control_conflict,
            "interval_95": interval(draws.copy()),
        }
    return {
        "effect": mean(row["effect"] for row in family_results.values()),
        "absolute_conflict_rate_change": mean(row["absolute_conflict_rate_change"] for row in family_results.values()),
        "interval_95": interval(list(map(mean, zip(*family_draws)))),
        "families": family_results,
    }


def compare_holdout(data: Path, base: Path, grpo: Path, sft: Path, selection_path: Path) -> dict:
    dataset = verified_file(data, BEHAVIOR_FILE)
    design = parse_jsonl(dataset)
    ids = {row["custom_id"] for row in design}
    evaluation = {}
    for arm, directory in (("base", base), ("grpo", grpo), ("sft", sft)):
        config, results = read_evaluation(directory, "holdout", sha256(dataset), ids)
        require_arm(config, arm)
        evaluation[arm] = config, index_results(results, ids)
    base_config, base_results = evaluation["base"]
    grpo_config, grpo_results = evaluation["grpo"]
    sft_config, sft_results = evaluation["sft"]
    require_comparable(base_config, grpo_config)
    require_comparable(grpo_config, sft_config)
    require_training_pair(grpo_config, sft_config)
    selection = json.loads(selection_path.read_bytes())
    if (
        selection["model"] != grpo_config["model"]
        or selection["validation_dataset_sha256"] != sha256(verified_file(data, "math_validation.jsonl"))
        or selection["seed"] != grpo_config["training"]["seed"]
        or selection["grpo_adapter"]["sha256"] != grpo_config["adapter"]["sha256"]
        or selection["selected_sft"]["adapter"]["sha256"] != sft_config["adapter"]["sha256"]
    ):
        raise ValueError("Held-out adapters do not match the validation selection.")
    return {
        "model": grpo_config["model"], "seed": selection["seed"], "dataset_sha256": sha256(dataset),
        "sft_within_three_points": selection["sft_within_three_points"],
        "interval_scope": "Template bootstrap conditional on these trained checkpoints",
        "primary_audit": "provided",
        "comparisons": {
            f"grpo_minus_{arm}": {
                audit: paired_contrast(design, grpo_results, comparator, audit) for audit in ("provided", "derived")
            }
            for arm, comparator in (("base", base_results), ("sft", sft_results))
        },
    }


def write_analysis(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json_bytes(result))
    print(f"Analysis written to {path}")
