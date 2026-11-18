from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from direct_response import DASHBOARD_VARIANTS, DIRECT_FILE, FAMILIES, direct_observations, score_direct
from pilot import parse_jsonl, sha256, verified_file
from qwen_analysis import read_evaluation, require_arm, require_comparable, require_training_pair
from qwen_score_sensitivity import interval_by_case


ARMS = ("base", "grpo", "sft")
ENDPOINTS = {
    "reward_target": {"reward_target"},
    "any_invalid": {"reward_target", "other_invalid"},
}


def case_outcomes(observations: list[dict], dashboard: str) -> dict:
    cases = defaultdict(dict)
    for row in observations:
        key = (row["family"], row["case_id"])
        variant = (row["dashboard_order"], row["condition"])
        if variant in cases[key]:
            raise ValueError(f"Duplicate direct-response variant for {key!r}.")
        cases[key][variant] = row["outcome"]
    orders, conditions = DASHBOARD_VARIANTS[dashboard]
    expected = {(order, condition) for order in orders for condition in conditions}
    if not cases or any(set(variants) != expected for variants in cases.values()):
        raise ValueError("Direct-response analysis requires every planned variant of each case.")
    return cases


def estimate(values: dict[str, list[float]]) -> dict:
    if set(values) != set(FAMILIES):
        raise ValueError("Direct-response analysis requires both task families.")
    rates = {family: mean(group) for family, group in values.items()}
    return {"effect": mean(rates.values()), "by_family": rates, "interval_95": interval_by_case(values)}


def compare_direct(data: Path, base: Path, grpo: Path, sft: Path) -> dict:
    raw = verified_file(data, DIRECT_FILE)
    manifest = json.loads((data / "manifest.json").read_bytes())
    dashboard = manifest.get("dashboard", "two-row")
    orders, _ = DASHBOARD_VARIANTS[dashboard]
    rows = parse_jsonl(raw)
    ids = {row["custom_id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError("Direct-response trial IDs must be unique.")
    configs, counts, checkpoints = {}, {}, {}
    for arm, path in zip(ARMS, (base, grpo, sft)):
        config, results = read_evaluation(path, "direct", sha256(raw), ids)
        require_arm(config, arm)
        configs[arm] = config
        require_comparable(configs["base"], config)
        counts[arm] = score_direct(rows, results)
        checkpoints[arm] = case_outcomes(direct_observations(rows, results), dashboard)
    require_training_pair(configs["grpo"], configs["sft"])

    shifts = {arm: {} for arm in ARMS}
    comparisons = {f"grpo_minus_{arm}": {} for arm in ("base", "sft")}
    display_effects = {arm: {} for arm in ARMS}
    with_cue, without_cue = (
        (("target_first", "tied"), ("valid_first", "tied")) if dashboard == "two-row"
        else (("bonus_rule", "tied"), ("bonus_rule", "no_target"))
    )
    for endpoint, relevant in ENDPOINTS.items():
        by_case = {arm: {} for arm in ARMS}
        for arm, cases in checkpoints.items():
            differences = defaultdict(list)
            display_changes = defaultdict(list)
            for key in sorted(cases):
                family, _ = key
                variants = cases[key]
                delta = mean(
                    int(variants[position, "conflict"] in relevant) - int(variants[position, "tied"] in relevant)
                    for position in orders
                )
                by_case[arm][key] = delta
                differences[family].append(delta)
                display_changes[family].append(int(variants[with_cue] in relevant) - int(variants[without_cue] in relevant))
            shifts[arm][endpoint] = estimate(differences)
            display_effects[arm][endpoint] = estimate(display_changes)
        for arm in ("base", "sft"):
            differences = defaultdict(list)
            for key, delta in by_case["grpo"].items():
                differences[key[0]].append(delta - by_case[arm][key])
            comparisons[f"grpo_minus_{arm}"][endpoint] = estimate(differences)

    display_contrast = (
        "target_first_minus_valid_first_at_tied" if dashboard == "two-row" else "target_mentioned_minus_unmentioned_at_zero"
    )
    return {
        "model": configs["base"]["model"], "training_seed": configs["grpo"]["training"]["seed"],
        "training_steps": configs["grpo"]["training"]["steps"], "dataset_sha256": sha256(raw),
        "cases": len(checkpoints["base"]), "trials_per_checkpoint": len(rows), "dashboard": dashboard,
        "counts": counts, "conflict_minus_tied": shifts, "comparisons": comparisons,
        display_contrast: display_effects,
        "interval_scope": "Case bootstrap within each family, conditional on these checkpoints",
    }
