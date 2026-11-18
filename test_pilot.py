from __future__ import annotations

import io
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from design import DEFAULT_SEED, MODELS, make_cases, prompt, trial_plan
from pilot import ClaudeError, download, json_bytes, jsonl_bytes, parse_jsonl, prepare, run_smoke, score, status, submit, verified_file
from scoring import score_results


def response(custom_id: str, reply: str) -> dict:
    return {
        "custom_id": custom_id,
        "result": {"type": "succeeded", "message": {
            "content": [{"type": "text", "text": reply}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 2},
        }},
    }


class DesignTests(unittest.TestCase):
    def test_cases_are_distinct_balanced_and_have_one_valid_answer(self):
        cases = make_cases(DEFAULT_SEED)
        all_cases = cases["smoke"] + cases["batch"]
        self.assertEqual(len(all_cases), 620)
        self.assertEqual(len({(case.scenario.evidence, case.scenario.valid_option) for case in all_cases}), 620)
        for stage, size in (("smoke", 10), ("batch", 300)):
            for family in ("records", "workflows"):
                group = [case for case in cases[stage] if case.scenario.family == family]
                self.assertEqual(len(group), size)
                self.assertEqual(Counter(case.valid_label for case in group), {"P": size // 2, "Q": size // 2})
                self.assertEqual(Counter(case.first_label for case in group), {"P": size // 2, "Q": size // 2})

        for case in all_cases:
            scenario = case.scenario
            if scenario.family == "records":
                counts = [int(line.split(": ")[1].split()[0]) for line in scenario.evidence.splitlines()[2:]]
                self.assertEqual(int(scenario.valid_option.split()[1]), sum(counts))
                self.assertNotEqual(int(scenario.invalid_option.split()[1]), sum(counts))
            else:
                rules = [line[2:-1].split(" must happen before ") for line in scenario.evidence.splitlines()[3:]]
                good = scenario.valid_option.split(" → ")
                bad = scenario.invalid_option.split(" → ")
                self.assertEqual(sorted(good), sorted(bad))
                self.assertEqual(len(set(good)), 6)
                self.assertFalse(any(good.index(left) >= good.index(right) for left, right in rules))
                self.assertEqual(sum(bad.index(left) >= bad.index(right) for left, right in rules), 1)

    def test_paired_prompts_change_only_the_score_section(self):
        requests, design = trial_plan(DEFAULT_SEED, list(MODELS), "provided")
        self.assertEqual({stage: len(rows) for stage, rows in requests.items()}, {"smoke": 240, "batch": 5400})
        self.assertEqual(len({row["custom_id"] for row in design}), len(design))
        groups = defaultdict(set)
        for row in requests["batch"]:
            case_key = row["custom_id"].rsplit("-", 1)[0]
            sections = row["params"]["messages"][0]["content"].split("\n\n")
            self.assertTrue(sections[-2].startswith("Dashboard points"))
            groups[case_key].add(tuple(sections[:-2] + sections[-1:]))
        self.assertEqual(len(groups), 1800)
        self.assertTrue(all(len(group) == 1 for group in groups.values()))
        case = make_cases(DEFAULT_SEED)["batch"][0]
        self.assertNotIn("Independent audit", prompt(case, "choice", "conflict", "derived"))


class ScoringTests(unittest.TestCase):
    def test_partial_and_out_of_order_results_preserve_paired_denominators(self):
        _, design = trial_plan(DEFAULT_SEED, ["haiku"], "provided")
        group = [row for row in design if row["stage"] == "batch" and row["family"] == "records"]
        first, second = group[:3], group[3:6]
        wrong = "Q" if first[0]["valid_label"] == "P" else "P"
        results = [
            response(second[1]["custom_id"], "I pick P."),
            response(first[1]["custom_id"], wrong),
            response(first[0]["custom_id"], first[0]["valid_label"]),
            {"custom_id": first[2]["custom_id"], "result": {"type": "expired"}},
        ]
        report, _ = score_results(design, results, "batch")
        records = report["choices"][0]
        self.assertEqual(records["complete_pairs"], 1)
        self.assertEqual(records["paired_conflict_effect"], 1.0)
        self.assertEqual(records["conditions"]["conflict"]["counts"]["malformed"], 1)
        self.assertEqual(records["conditions"]["conflict"]["rates_among_responses"]["invalid"], 0.5)
        self.assertEqual(records["conditions"]["aligned"]["counts"]["expired"], 1)
        with self.assertRaisesRegex(ValueError, "Duplicate result ID"):
            score_results(design, results + results[:1], "batch")


class RunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name) / "prepared"
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        prepare(self.run, DEFAULT_SEED, ["sonnet"], "provided")
        self.design = parse_jsonl(verified_file(self.run, "design.jsonl"))
        key = patch.dict("os.environ", {"ANTHROPIC_API_KEY": "offline-test-key"})
        key.start()
        self.addCleanup(key.stop)

    def test_smoke_resumes_from_saved_results(self):
        smoke = [trial for trial in self.design if trial["stage"] == "smoke"]
        messages = [response(trial["custom_id"], trial["expected"] or trial["valid_label"])["result"]["message"] for trial in smoke]
        with patch("pilot.Claude.request", side_effect=[json_bytes(messages[0]), ClaudeError(429, "rate limit")]):
            with self.assertRaises(ClaudeError):
                run_smoke(self.run)
        with patch("pilot.Claude.request", side_effect=[json_bytes(message) for message in messages[1:]]) as api:
            run_smoke(self.run)
            self.assertEqual(api.call_count, 79)
        score(self.run, "smoke", None)
        summary = json.loads((self.run / "smoke_summary.json").read_bytes())
        self.assertTrue(summary["smoke_diagnostics"]["sonnet"]["passes"])
        self.assertEqual(summary["result_rows"], 80)

    def test_submit_download_and_score_using_mocked_http(self):
        batch_trials = [trial for trial in self.design if trial["stage"] == "batch"]
        results = [response(trial["custom_id"], trial["valid_label"]) for trial in reversed(batch_trials)]
        batch = {"id": "msgbatch_fixture", "processing_status": "ended"}
        http_responses = [json_bytes(batch)] * 3 + [jsonl_bytes(results)]
        with patch("pilot.urlopen", side_effect=[io.BytesIO(body) for body in http_responses]) as api:
            submit(self.run)
            with self.assertRaisesRegex(ValueError, "already exists"):
                submit(self.run)
            status(self.run, None)
            download(self.run, None)
            self.assertEqual(api.call_count, 4)
            self.assertEqual(api.call_args_list[0].args[0].get_method(), "POST")
            self.assertTrue(api.call_args_list[-1].args[0].full_url.endswith("/msgbatch_fixture/results"))
        score(self.run, "batch", None)
        summary = json.loads((self.run / "batch_summary.json").read_bytes())
        self.assertEqual(summary["outcome_counts"], {"valid": 1800})
        self.assertTrue(all(group["paired_conflict_effect"] == 0 for group in summary["choices"]))

    def test_modified_payload_is_rejected_before_submission(self):
        with (self.run / "batch.json").open("ab") as file:
            file.write(b"\n")
        with patch("pilot.urlopen") as api:
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                submit(self.run)
            api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
