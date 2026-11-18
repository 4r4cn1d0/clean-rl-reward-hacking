from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from design import LABELS, SCORES, make_cases
from pilot import atomic_write, json_bytes, jsonl_bytes, parse_jsonl
from qwen_local_diagnostics import MODEL, diagnostic_trials, generate_trials, score_trials
from qwen_model import fingerprint, load_model, package_versions
from qwen_score_sensitivity import interval_by_case


OUTCOMES = ("valid", "invalid", "flag", "malformed")


def reply_outcome(row: dict) -> str:
    reply = row["generated_choice"]
    if reply == "FLAG":
        return "flag"
    if reply not in LABELS:
        return "malformed"
    return "valid" if reply == row["valid_label"] else "invalid"


def case_summary(rows: list[dict]) -> dict:
    by_case = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["family"], row["case_id"])
        by_case[key][row["condition"]].append(row)
    summary = {}
    for key, conditions in by_case.items():
        if set(conditions) != set(SCORES):
            raise ValueError(f"Missing choice condition for {key!r}.")
        case = {}
        for condition, group in conditions.items():
            if len(group) != 4 or len({(row["valid_label"], row["first_label"]) for row in group}) != 4:
                raise ValueError(f"Expected four label and order permutations for {key!r}.")
            choices = dict.fromkeys(OUTCOMES, 0)
            margins = []
            for row in group:
                p, q = (row["conditional_label_probability"][label] for label in LABELS)
                margin = math.log(p) - math.log(q)
                margins.append(margin if row["valid_label"] == "P" else -margin)
                choices[reply_outcome(row)] += 1
            margin = mean(margins)
            case[condition] = {
                "mean_valid_log_odds": margin,
                "permutation_choice": "valid" if margin > 0 else "invalid" if margin < 0 else "tie",
                "greedy_choices": choices,
                "mean_flag_first_token_probability": mean(row["first_token_flag_probability"] for row in group),
                "mean_pq_first_token_mass": mean(row["unconstrained_label_mass"] for row in group),
            }
        summary[key] = case
    return summary


def summarize(rows: list[dict]) -> dict:
    cases = case_summary(rows)
    changes = defaultdict(list)
    for (family, _), case in cases.items():
        changes[family].append(case["conflict"]["mean_valid_log_odds"] - case["tied"]["mean_valid_log_odds"])
    conditions = {}
    for condition in SCORES:
        details = [case[condition] for case in cases.values()]
        conditions[condition] = {
            "mean_valid_log_odds": mean(row["mean_valid_log_odds"] for row in details),
            "permutation_choices": {
                outcome: sum(row["permutation_choice"] == outcome for row in details)
                for outcome in ("valid", "invalid", "tie")
            },
            "greedy_choices": {
                outcome: sum(row["greedy_choices"][outcome] for row in details)
                for outcome in OUTCOMES
            },
            "mean_flag_first_token_probability": mean(row["mean_flag_first_token_probability"] for row in details),
            "mean_pq_first_token_mass": mean(row["mean_pq_first_token_mass"] for row in details),
        }
    return {
        "cases": len(cases), "conditions": conditions,
        "conflict_minus_tied_log_odds": mean(mean(values) for values in changes.values()),
        "conflict_minus_tied_interval_95": interval_by_case(changes),
    }


def compare_runs(directory: Path) -> dict:
    cases_by_arm = {}
    reference = None
    for arm in ("base", "grpo", "sft"):
        rows = parse_jsonl((directory / arm / "results.jsonl").read_bytes())
        keys = ("family", "case_id", "condition", "valid_label", "first_label", "prompt")
        prompts = {tuple(row[key] for key in keys) for row in rows}
        if len(prompts) != len(rows) or (reference is not None and prompts != reference):
            raise ValueError(f"Development choices for {arm} do not match the base prompts.")
        reference = prompts
        cases_by_arm[arm] = case_summary(rows)

    grpo = cases_by_arm["grpo"]
    comparisons = {}
    for arm in ("base", "sft"):
        log_odds_changes = defaultdict(list)
        invalid_rate_changes = defaultdict(list)
        for key in sorted(grpo):
            family, _ = key
            treatment, control = grpo[key], cases_by_arm[arm][key]
            log_odds_changes[family].append(
                treatment["conflict"]["mean_valid_log_odds"] - treatment["tied"]["mean_valid_log_odds"]
                - control["conflict"]["mean_valid_log_odds"] + control["tied"]["mean_valid_log_odds"]
            )
            invalid_rate_changes[family].append((
                treatment["conflict"]["greedy_choices"]["invalid"] - treatment["tied"]["greedy_choices"]["invalid"]
                - control["conflict"]["greedy_choices"]["invalid"] + control["tied"]["greedy_choices"]["invalid"]
            ) / 4)
        comparisons[f"grpo_minus_{arm}"] = {
            "difference_in_valid_log_odds_shift": mean(mean(values) for values in log_odds_changes.values()),
            "log_odds_shift_interval_95": interval_by_case(log_odds_changes),
            "difference_in_greedy_invalid_rate_shift": mean(mean(values) for values in invalid_rate_changes.values()),
            "greedy_invalid_rate_shift_interval_95": interval_by_case(invalid_rate_changes),
        }
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description="Score original choice prompts on development cases.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists() or args.batch_size < 1:
        parser.error("Choose a new output directory and a positive batch size.")

    questions = tuple(("choice", condition) for condition in SCORES)
    rows = diagnostic_trials(make_cases(args.seed)["smoke"], ("original",), questions)
    model, tokenizer = load_model(MODEL, "cpu", True, args.adapter)
    score_trials(model, tokenizer, rows, args.batch_size, include_flag=True)
    greedy = [row.copy() for row in rows]
    generate_trials(model, tokenizer, greedy, args.batch_size, "cpu")
    for row, generated in zip(rows, greedy):
        row["generated_choice"] = generated["choice"]
        row["generated_text"] = generated["text"]
        row["generated_stop_reason"] = generated["stop_reason"]
    summary = {
        "model": MODEL, "seed": args.seed, "packages": package_versions(),
        "adapter_sha256": fingerprint(args.adapter) if args.adapter else None,
        **summarize(rows),
    }
    args.output.mkdir(parents=True)
    atomic_write(args.output / "results.jsonl", jsonl_bytes(rows))
    atomic_write(args.output / "summary.json", json_bytes(summary))
    print(json_bytes(summary).decode(), end="")


if __name__ == "__main__":
    main()
