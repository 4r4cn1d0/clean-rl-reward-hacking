from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

from pilot import atomic_write, json_bytes, parse_jsonl


KINDS = ("validity", "score")
FAMILIES = ("records", "workflows")
ARMS = ("base", "grpo", "sft")


def paired_margins(rows: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    pairs = defaultdict(dict)
    by_condition = defaultdict(lambda: defaultdict(list))
    for row in rows:
        kind = row["kind"]
        if kind not in KINDS:
            raise ValueError(f"Unexpected diagnostic kind: {kind!r}")
        # For the score question, the audit and the case stay fixed as the points swap.
        # For validity, compare swapped option labels at each display order.
        fixed_label = row["valid_label"] if kind == "score" else None
        key = (row["layout"], kind, row["family"], row["case_id"], row["first_label"], fixed_label)
        expected = row["expected"]
        if expected not in {"P", "Q"} or expected in pairs[key]:
            raise ValueError(f"Duplicate or invalid expected label in pair {key!r}.")
        probabilities = row["conditional_label_probability"]
        p, q = probabilities["P"], probabilities["Q"]
        if not all(math.isfinite(value) and 0 < value <= 1 for value in (p, q)) or not math.isclose(p + q, 1, abs_tol=1e-6):
            raise ValueError(f"Invalid label probabilities in pair {key!r}.")
        margin = math.log(p) - math.log(q)
        pairs[key][expected] = (margin, row["choice"] == expected, row["choice"] == "P")
        # Follow the same underlying option through the label and display-order permutations.
        by_condition[key[:4]][row["condition"]].append(margin if row["valid_label"] == "P" else -margin)

    grouped = defaultdict(list)
    for key, answers in pairs.items():
        if set(answers) != {"P", "Q"}:
            raise ValueError(f"Missing counterpart for pair {key!r}.")
        p, q = answers["P"], answers["Q"]
        grouped[key[:4]].append((p[0] - q[0], (p[0] + q[0]) / 2, int(p[1]) + int(q[1]), int(p[2]) + int(q[2])))
    result = {}
    for key, values in grouped.items():
        kind = key[1]
        if len(values) != (4 if kind == "score" else 2):
            raise ValueError(f"Incomplete label and display-order balance for {key!r}.")
        permutations = by_condition[key]
        conditions = ("conflict", "aligned") if kind == "score" else (None,)
        if set(permutations) != set(conditions) or any(len(margins) != 4 for margins in permutations.values()):
            raise ValueError(f"Expected four label and order permutations per condition for {key!r}.")
        correct = sum(
            mean(permutations[condition]) * (-1 if condition == "conflict" else 1) > 0
            for condition in conditions
        )
        result[key] = {
            "shift": mean(value[0] for value in values),
            "p_bias": mean(value[1] for value in values),
            "pairs": len(values),
            "correct": sum(value[2] for value in values),
            "p_choices": sum(value[3] for value in values),
            "permutation_questions": len(conditions),
            "permutation_correct": correct,
            "all_permutation_answers_correct": correct == len(conditions),
        }
    return result


def interval_by_case(values: dict[str, list[float]], draws: int = 4000) -> list[float]:
    rng = random.Random(20261113)
    samples = sorted(
        mean(mean(rng.choices(group, k=len(group))) for group in values.values())
        for _ in range(draws)
    )
    return [samples[int((draws - 1) * probability)] for probability in (0.025, 0.975)]


def summarize_checkpoints(checkpoints: dict[str, list[dict]]) -> dict:
    if "base" not in checkpoints:
        raise ValueError("Base diagnostic responses are required.")
    # Compare the same rendered prompts across checkpoints before pairing responses.
    prompt_keys = ("layout", "kind", "condition", "family", "case_id", "first_label", "valid_label", "expected", "prompt")
    reference = {tuple(row[key] for key in prompt_keys) for row in checkpoints["base"]}
    estimates = {}
    for name, rows in checkpoints.items():
        if len(rows) != len(reference) or {tuple(row[key] for key in prompt_keys) for row in rows} != reference:
            raise ValueError(f"Diagnostic prompts for {name} do not match the base prompts.")
        estimates[name] = paired_margins(rows)

    output = {}
    base = estimates["base"]
    # Fixed case ordering keeps the bootstrap draws comparable across checkpoints.
    for arm, cases in estimates.items():
        if cases.keys() != base.keys():
            raise ValueError(f"Diagnostic pairs differ for {arm}.")
        output[arm] = {}
        for layout in sorted({key[0] for key in cases}):
            output[arm][layout] = {}
            for kind in KINDS:
                shifts = defaultdict(list)
                differences = defaultdict(list)
                totals = []
                for key in sorted(cases):
                    row = cases[key]
                    row_layout, row_kind, family, _ = key
                    if (row_layout, row_kind) == (layout, kind):
                        shifts[family].append(row["shift"])
                        differences[family].append(row["shift"] - base[key]["shift"])
                        totals.append(row)
                if set(shifts) != set(FAMILIES) or len({len(values) for values in shifts.values()}) != 1:
                    raise ValueError(f"Expected balanced task families for {layout} and {kind}.")
                output[arm][layout][kind] = {
                    "cases": len(totals),
                    "pairs": sum(row["pairs"] for row in totals),
                    "raw_correct": sum(row["correct"] for row in totals),
                    "raw_p_choices": sum(row["p_choices"] for row in totals),
                    "permutation_questions": sum(row["permutation_questions"] for row in totals),
                    "permutation_correct": sum(row["permutation_correct"] for row in totals),
                    "all_permutation_answers_correct_cases": sum(row["all_permutation_answers_correct"] for row in totals),
                    "mean_p_bias": mean(row["p_bias"] for row in totals),
                    "mean_log_odds_shift": mean(row["shift"] for row in totals),
                    "shift_interval_95": interval_by_case(shifts),
                    "positive_case_shifts": sum(row["shift"] > 0 for row in totals),
                    "change_from_base": mean(mean(values) for values in differences.values()),
                    "change_interval_95": interval_by_case(differences),
                }
    return output


def analyze_run(directory: Path) -> dict:
    responses = {}
    configs = {}
    for name in ARMS:
        path = directory / name
        responses[name] = parse_jsonl((path / "results.jsonl").read_bytes())
        configs[name] = json.loads((path / "summary.json").read_bytes())
    reference = configs["base"]
    if reference["adapter_sha256"] is not None:
        raise ValueError("Base diagnostic results must use the base checkpoint.")
    for name in ARMS[1:]:
        config = configs[name]
        keys = ("model", "device", "packages", "seed", "development_cases", "decoder")
        if any(config[key] != reference[key] for key in keys):
            raise ValueError(f"Diagnostic settings for {name} differ from the base settings.")
        if not config["adapter_sha256"]:
            raise ValueError(f"Diagnostic results for {name} must identify the trained adapter.")
    return {
        "model": reference["model"], "seed": reference["seed"],
        "adapter_sha256": {name: config["adapter_sha256"] for name, config in configs.items()},
        "checkpoints": summarize_checkpoints(responses),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired label odds on development prompts.")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    analysis = analyze_run(args.directory)
    atomic_write(args.directory / "paired_summary.json", json_bytes(analysis))
    print(json_bytes(analysis).decode(), end="")


if __name__ == "__main__":
    main()
