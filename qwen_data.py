from __future__ import annotations

from collections import Counter
from pathlib import Path

from arithmetic import check_item, make_arithmetic
from holdout import make_diagnostics, make_holdout
from pilot import jsonl_bytes, parse_jsonl, verified_file, write_manifested_directory


DEFAULT_DATA_SEED = 20261111
BEHAVIOR_FILE = "behavior_holdout.jsonl"
DIAGNOSTICS_FILE = "behavior_diagnostics.jsonl"


def prepare_experiment(
    directory: Path, seed: int, arithmetic_profile: str = "standard", answer_format: str = "worked",
) -> None:
    arithmetic = make_arithmetic(seed, profile=arithmetic_profile, answer_format=answer_format)
    behavior = make_holdout(seed + 1)
    templates = {split: {row["template"] for row in rows} for split, rows in arithmetic.items()}
    if sum(map(len, templates.values())) != len(set.union(*templates.values())):
        raise ValueError("Arithmetic splits share templates.")
    files = {f"math_{split}.jsonl": jsonl_bytes(rows) for split, rows in arithmetic.items()}
    files[BEHAVIOR_FILE] = jsonl_bytes(behavior)
    diagnostics = make_diagnostics(seed + 1)
    files[DIAGNOSTICS_FILE] = jsonl_bytes(diagnostics)
    metadata = {
        "schema_version": 1,
        "arithmetic_seed": seed,
        "arithmetic_profile": arithmetic_profile,
        "behavior_seed": seed + 1,
        "arithmetic_counts": {split: len(rows) for split, rows in arithmetic.items()},
        "arithmetic_template_counts": {split: len(ids) for split, ids in templates.items()},
        "behavior_cases": len({row["case_id"] for row in behavior}),
        "behavior_trials": len(behavior),
        "diagnostic_trials": len(diagnostics),
        "behavior_cases_by_family": dict(Counter(row["family"] for row in behavior if row["audit"] == "provided" and row["condition"] == "tied")),
    }
    if answer_format != "worked":
        metadata["arithmetic_answer_format"] = answer_format
    write_manifested_directory(directory, files, metadata)
    print(f"Prepared {metadata['arithmetic_counts']} arithmetic prompts and {len(behavior):,} held-out trials in {directory}")


def verified_math(directory: Path, split: str) -> list[dict]:
    rows = parse_jsonl(verified_file(directory, f"math_{split}.jsonl"))
    if len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError(f"Arithmetic {split} has duplicate case IDs.")
    for row in rows:
        check_item(row)
    return rows
