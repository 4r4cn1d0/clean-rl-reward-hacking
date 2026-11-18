from __future__ import annotations

import io
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from arithmetic import audit_reward_rows, independently_evaluate, make_arithmetic, parse_answer
from design import make_cases
from holdout import holdout_cases, make_holdout
from pilot import json_bytes, jsonl_bytes, parse_jsonl, sha256, write_manifested_directory
from qwen_analysis import paired_contrast, select_validation
from qwen_eval import saved_results
from qwen_train import exact_reward


def result(custom_id: str, text: str) -> dict:
    return {"custom_id": custom_id, "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": text}]}}}


class ArithmeticTests(unittest.TestCase):
    def test_full_math_splits_are_disjoint_and_replay(self):
        corpus = make_arithmetic(20261111)
        self.assertEqual({split: len(rows) for split, rows in corpus.items()}, {"train": 4096, "validation": 1000, "test": 1000})
        templates = [set(row["template"] for row in rows) for rows in corpus.values()]
        self.assertEqual(len(set.union(*templates)), sum(map(len, templates)))
        self.assertEqual(len({row["expression"] for rows in corpus.values() for row in rows}), 6096)
        for rows in corpus.values():
            for row in rows:
                self.assertEqual(independently_evaluate(row["expression"]), row["answer"])

    def test_reward_replay_rejects_an_invalid_positive(self):
        self.assertEqual(parse_answer("4 - 9 = -5\nFinal answer: -5\n"), -5)
        for text in ("Final answer: 5\nFinal answer: 5", "Final answer: 5\nMore text", "Final answer: 05", "Final answer: ５"):
            self.assertIsNone(parse_answer(text))
        self.assertEqual(independently_evaluate("((4 - 9) * (-2))"), 10)
        with self.assertRaises(ValueError):
            independently_evaluate("2 ** 3")
        item = {"case_id": "fixture", "expression": "(4 - 9)", "answer": -5}
        stream = io.BytesIO()
        reward = exact_reward(stream)
        outputs = ["Final answer: -5", "Final answer: 5", "Final answer: -5\nExtra text"]
        self.assertEqual(reward(outputs, [item["case_id"]] * 3, [item["expression"]] * 3, [item["answer"]] * 3), [1., 0., 0.])
        rows = parse_jsonl(stream.getvalue())
        self.assertEqual(audit_reward_rows(rows, {"fixture": item})["positive_rewards"], 1)
        rows[1]["reward"] = 1
        with self.assertRaisesRegex(ValueError, "Reward audit mismatch"):
            audit_reward_rows(rows, {"fixture": item})


class HoldoutTests(unittest.TestCase):
    def test_holdout_preserves_pairing_and_workflow_constraints(self):
        seed = 20261112
        cases = holdout_cases(seed)
        self.assertEqual(Counter(case.scenario.family for case in cases), {"records": 400, "workflows": 400})
        self.assertEqual(set(Counter(case.scenario.template for case in cases).values()), {20})
        self.assertEqual(len({case.scenario.template for case in cases}), 40)
        development = make_cases(20261111)
        self.assertFalse({case.scenario.evidence for case in cases} & {case.scenario.evidence for group in development.values() for case in group})
        for case in cases:
            if case.scenario.family == "records":
                evidence = case.scenario.evidence.splitlines()[2:]
                first, second = [list(map(int, re.findall(r"\d+", line))) for line in evidence]
                mode = (int(case.scenario.template.rsplit("-", 1)[1]) - 1) % 3
                if mode == 0:
                    expected = [first[0] + first[1] - first[2], second[0] + second[1] - second[2]]
                elif mode == 1:
                    expected = [first[0] - second[0] + second[1], first[1] - second[1] + second[0]]
                else:
                    expected = [first[0] * first[1] + first[2], second[0] * second[1] + second[2]]
                valid = list(map(int, re.findall(r"\d+", case.scenario.valid_option)))
                invalid = list(map(int, re.findall(r"\d+", case.scenario.invalid_option)))
                self.assertEqual(valid, expected)
                self.assertEqual(sum(a != b for a, b in zip(valid, invalid)), 1)
                continue
            valid = case.scenario.valid_option.split(" → ")
            invalid = case.scenario.invalid_option.split(" → ")
            rules = case.scenario.evidence.splitlines()[3:]
            self.assertEqual(sorted(valid), sorted(invalid))

            def violated(sequence):
                failures = 0
                for line in rules:
                    activity, constraint = line[2:-1].split(" must ", 1)
                    slot = re.fullmatch(r"be activity number (\d+)", constraint)
                    if slot:
                        failures += sequence.index(activity) != int(slot[1]) - 1
                    else:
                        immediate = constraint.startswith("occur immediately before ")
                        other = constraint.split("before ", 1)[1].split(",")[0]
                        distance = sequence.index(other) - sequence.index(activity)
                        failures += distance != 1 if immediate else distance < (2 if "," in constraint else 1)
                return failures

            self.assertEqual(violated(valid), 0)
            self.assertEqual(violated(invalid), 1)
        trials = make_holdout(seed)
        self.assertEqual(len(trials), 4800)
        self.assertEqual(len({trial["custom_id"] for trial in trials}), len(trials))

    def test_paired_analysis_keeps_audit_conditions_separate(self):
        design = make_holdout(20261112)
        control = {}
        treatment = {}
        for row in design:
            custom_id, valid = row["custom_id"], row["valid_label"]
            conflict = row["condition"] == "conflict" and row["audit"] == "provided"
            alternative = "Q" if valid == "P" else "P"
            control[custom_id] = result(custom_id, valid)["result"]
            treatment[custom_id] = result(custom_id, alternative if conflict else valid)["result"]
        primary = paired_contrast(design, treatment, control, "provided")
        self.assertEqual(primary["effect"], 1)
        self.assertEqual(primary["interval_95"], [1, 1])
        self.assertEqual(paired_contrast(design, treatment, control, "derived")["effect"], 0)


class AnalysisTests(unittest.TestCase):
    def test_validation_selects_earlier_sft_checkpoint_on_exact_tie(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = make_arithmetic(13, {"train": 0, "validation": 4, "test": 0})["validation"]
            data = root / "data"
            raw = jsonl_bytes(rows)
            write_manifested_directory(data, {"math_validation.jsonl": raw}, {})
            common = {
                "task": "validation", "dataset_sha256": sha256(raw), "model": {"name": "fixture"}, "greedy": True,
                "device": "cpu", "batch_size": 4, "max_new_tokens": 64, "packages": {}, "planned": 4, "full_dataset": True,
            }
            training = {"seed": 17, "steps": 4, "training_corpus_sha256": "abc", "used_prompt_ids_sha256": "def", "lora": {}}
            for name, arm, adapter, correct in (("base", "base", None, 1), ("grpo", "grpo", "adapter", 2),
                                               ("late", "sft", "checkpoint-3", 3), ("early", "sft", "checkpoint-1", 1)):
                directory = root / name
                directory.mkdir()
                config = {**common, "adapter": None if adapter is None else {"path": adapter, "sha256": name}}
                if arm != "base":
                    config["training"] = {**training, "arm": arm}
                replies = [result(row["case_id"], f"Final answer: {row['answer'] + (index >= correct)}") for index, row in enumerate(rows)]
                (directory / "evaluation_config.json").write_bytes(json_bytes(config))
                (directory / "results.jsonl").write_bytes(jsonl_bytes(replies))
            report = select_validation(data, root / "base", root / "grpo", [root / "late", root / "early"])
            self.assertEqual(report["selected_sft"]["step"], 1)
            self.assertEqual(report["grpo_accuracy_gain"], 0.25)
            self.assertFalse(report["sft_within_three_points"])

    def test_evaluation_resumption_recovers_a_partial_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            valid = jsonl_bytes([result("fixture", "P")])
            path.write_bytes(valid + b'{"incomplete"')
            self.assertEqual(saved_results(path), parse_jsonl(valid))
            self.assertEqual(path.read_bytes(), valid)


if __name__ == "__main__":
    unittest.main()
