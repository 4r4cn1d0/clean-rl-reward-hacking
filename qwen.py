from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from arithmetic import ANSWER_FORMATS, ARITHMETIC_PROFILES
from direct_response import DASHBOARD_VARIANTS, DIRECT_SEED, prepare_direct
from pilot import json_bytes
from qwen_analysis import compare_holdout, select_validation, write_analysis
from qwen_data import DEFAULT_DATA_SEED, prepare_experiment
from qwen_direct_analysis import compare_direct
from qwen_eval import EVAL_FILES, evaluate
from qwen_model import MODEL
from qwen_study import study
from qwen_train import reward_audit, train


def model_options(command) -> None:
    command.add_argument("--model", default=MODEL)
    command.add_argument("--revision", help="Exact Hugging Face commit for a different model.")
    command.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    command.add_argument("--offline", action="store_true", help="Load weights only from local files or cache.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, train, and evaluate the Idea 1 Qwen experiment.")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("data", type=Path)
    prepare.add_argument("--seed", type=int, default=DEFAULT_DATA_SEED)
    prepare.add_argument("--arithmetic-profile", choices=ARITHMETIC_PROFILES, default="standard")
    prepare.add_argument("--answer-format", choices=ANSWER_FORMATS, default="worked")
    direct = commands.add_parser("prepare-direct")
    direct.add_argument("data", type=Path)
    direct.add_argument("--seed", type=int, default=DIRECT_SEED)
    direct.add_argument("--dashboard", choices=DASHBOARD_VARIANTS, default="two-row")
    training = commands.add_parser("train")
    training.add_argument("arm", choices=("grpo", "sft"))
    training.add_argument("data", type=Path)
    training.add_argument("output", type=Path)
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--steps", type=int, default=64)
    training.add_argument("--micro-batch", type=int, default=4)
    training.add_argument("--max-completion-tokens", type=int, default=256)
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--resume", action="store_true")
    model_options(training)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("task", choices=EVAL_FILES)
    evaluation.add_argument("data", type=Path)
    evaluation.add_argument("output", type=Path)
    evaluation.add_argument("--adapter", type=Path)
    evaluation.add_argument("--batch-size", type=int, default=8)
    evaluation.add_argument("--max-new-tokens", type=int)
    evaluation.add_argument("--limit", type=int)
    model_options(evaluation)
    audit = commands.add_parser("audit")
    audit.add_argument("data", type=Path)
    audit.add_argument("output", type=Path)
    select = commands.add_parser("select-sft")
    select.add_argument("data", type=Path)
    select.add_argument("base", type=Path)
    select.add_argument("grpo", type=Path)
    select.add_argument("sft", type=Path, nargs="+")
    select.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("data", type=Path)
    compare.add_argument("base", type=Path)
    compare.add_argument("grpo", type=Path)
    compare.add_argument("sft", type=Path)
    compare.add_argument("selection", type=Path)
    compare.add_argument("--output", type=Path, required=True)
    compare_submissions = commands.add_parser("compare-direct")
    compare_submissions.add_argument("data", type=Path)
    compare_submissions.add_argument("base", type=Path)
    compare_submissions.add_argument("grpo", type=Path)
    compare_submissions.add_argument("sft", type=Path)
    compare_submissions.add_argument("--output", type=Path, required=True)
    run_study = commands.add_parser("study")
    run_study.add_argument("profile", choices=("pilot", "main"))
    run_study.add_argument("data", type=Path)
    run_study.add_argument("output", type=Path)
    run_study.add_argument("--development", type=Path, default=Path("prepared"))
    run_study.add_argument("--steps", type=int, help="Override the profile's number of training updates.")
    run_study.add_argument("--batch-size", type=int, default=8)
    run_study.add_argument("--max-completion-tokens", type=int, default=256)
    model_options(run_study)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_experiment(args.data, args.seed, args.arithmetic_profile, args.answer_format)
        elif args.command == "prepare-direct":
            prepare_direct(args.data, args.seed, args.dashboard)
        elif args.command == "train":
            train(args)
        elif args.command == "evaluate":
            evaluate(args)
        elif args.command == "audit":
            print(json_bytes(reward_audit(args.data, args.output)).decode(), end="")
        elif args.command == "select-sft":
            write_analysis(args.output, select_validation(args.data, args.base, args.grpo, args.sft))
        elif args.command == "compare":
            write_analysis(args.output, compare_holdout(args.data, args.base, args.grpo, args.sft, args.selection))
        elif args.command == "compare-direct":
            write_analysis(args.output, compare_direct(args.data, args.base, args.grpo, args.sft))
        else:
            study(args)
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
