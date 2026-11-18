from __future__ import annotations

import json
import tempfile
from pathlib import Path

from arithmetic import make_arithmetic
from direct_response import DIRECT_FILE, development_rows, score_direct
from pilot import jsonl_bytes, parse_jsonl, sha256, verified_file, write_manifested_directory
from qwen_development_choice import compare_runs, summarize
from qwen_direct_analysis import compare_direct
from qwen_eval import score_math
from qwen_score_sensitivity import analyze_run
from qwen_train import reward_audit


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "development_results"
ARMS = ("base", "grpo", "sft")


def read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def require_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(message)


def verify_arithmetic() -> None:
    run = RESULTS / "cpu_short_answer_pilot"
    manifest = read_json(run / "data_manifest.json")
    data = make_arithmetic(
        manifest["arithmetic_seed"], profile=manifest["arithmetic_profile"],
        answer_format=manifest["arithmetic_answer_format"],
    )
    for split, rows in data.items():
        require_equal(
            sha256(jsonl_bytes(rows)), manifest["sha256"][f"math_{split}.jsonl"],
            f"Arithmetic {split} differs from its recorded checksum.",
        )
    for arm in ARMS:
        for prefix, items in ((Path(), data["validation"][:240]), (Path("confirmation"), data["validation"][240:480])):
            directory = run / prefix / f"{arm}_validation"
            responses = parse_jsonl((directory / "results.jsonl").read_bytes())
            require_equal(score_math(items, responses), read_json(directory / "summary.json"), f"Arithmetic replay differs for {directory}.")
    with tempfile.TemporaryDirectory(prefix="idea1-reward-replay-") as scratch:
        directory = Path(scratch) / "data"
        write_manifested_directory(directory, {"math_train.jsonl": jsonl_bytes(data["train"])}, {})
        require_equal(
            reward_audit(directory, run / "grpo"), read_json(run / "grpo/training_summary.json")["reward_audit"],
            "GRPO reward replay differs from the saved summary.",
        )
    print("Arithmetic accuracy and all 1,536 GRPO rewards verified.")


def verify_labels() -> None:
    run = RESULTS / "cpu_score_sensitivity"
    require_equal(analyze_run(run), read_json(run / "paired_summary.json"), "Label probability analysis differs from the saved results.")
    choices = read_json(run / "choice_summary.json")
    require_equal(compare_runs(run / "choice"), choices["comparisons"], "Choice comparisons differ from the saved results.")
    for arm in ARMS:
        rows = parse_jsonl((run / "choice" / arm / "results.jsonl").read_bytes())
        saved = choices["checkpoints"][arm]
        for key, value in summarize(rows).items():
            require_equal(value, saved[key], f"Choice summary differs for {arm}.")
    print("Label permutations and development choices verified.")


def verify_direct() -> None:
    root = RESULTS / "cpu_direct_response"
    for run in (root, root / "bonus", root / "bonus_replication"):
        directory = run / "data"
        manifest = read_json(directory / "manifest.json")
        raw = verified_file(directory, DIRECT_FILE)
        require_equal(
            raw, jsonl_bytes(development_rows(manifest["seed"], manifest.get("dashboard", "two-row"))),
            f"Direct-response data differs at {run}.",
        )
        rows = parse_jsonl(raw)
        for arm in ARMS:
            responses = parse_jsonl((run / arm / "results.jsonl").read_bytes())
            require_equal(score_direct(rows, responses), read_json(run / arm / "summary.json"), f"Direct-response scoring differs for {arm} at {run}.")
        require_equal(
            compare_direct(directory, *(run / arm for arm in ARMS)),
            read_json(run / "comparison.json"), f"Direct-response comparison differs at {run}.",
        )
    print("All 1,440 direct submissions and paired comparisons verified.")


def main() -> None:
    verify_arithmetic()
    verify_labels()
    verify_direct()


if __name__ == "__main__":
    main()
