from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from arithmetic import parse_answer
from direct_response import DIRECT_FILE, score_direct
from pilot import atomic_write, json_bytes, jsonl_bytes, parse_jsonl, run_lock, sha256, verified_file
from qwen_data import BEHAVIOR_FILE, DIAGNOSTICS_FILE, verified_math
from qwen_model import fingerprint, load_model, model_reference, package_versions
from scoring import index_results, score_results


EVAL_FILES = {
    "validation": "math_validation.jsonl", "test": "math_test.jsonl", "holdout": BEHAVIOR_FILE,
    "diagnostics": DIAGNOSTICS_FILE, "development": "smoke_requests.jsonl", "direct": DIRECT_FILE,
}


def score_math(rows: list[dict], results: list[dict]) -> dict:
    by_id = index_results(results, {row["case_id"] for row in rows})
    totals = defaultdict(Counter)
    for row in rows:
        result = by_id.get(row["case_id"])
        if result is None or result["type"] != "succeeded":
            outcome = "missing" if result is None else result["type"]
        else:
            answer = parse_answer(result["message"]["content"][0]["text"])
            outcome = "malformed" if answer is None else "correct" if answer == row["answer"] else "incorrect"
        totals["all"][outcome] += 1
        totals[row["tier"]][outcome] += 1
    return {tier: {"n": sum(counts.values()), "accuracy": counts["correct"] / sum(counts.values()), "outcomes": dict(counts)} for tier, counts in totals.items()}


def score_diagnostics(observations: list[dict]) -> list[dict]:
    groups = defaultdict(Counter)
    for row in observations:
        if row["kind"] in {"validity", "score"}:
            groups[(row["audit"], row["family"], row["kind"])][row["outcome"]] += 1
    return [
        {"audit": audit, "family": family, "kind": kind, "n": sum(counts.values()),
         "accuracy": counts["correct"] / sum(counts.values()), "outcomes": dict(counts)}
        for (audit, family, kind), counts in sorted(groups.items())
    ]


def inference_batches(model, tokenizer, rows: list[dict], device: str, batch_size: int, max_tokens: int):
    import torch
    from transformers import GenerationConfig

    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    stop_ids = {eos} if isinstance(eos, int) else set(eos or ())
    generation = GenerationConfig(
        do_sample=False, max_new_tokens=max_tokens, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
    )
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            group = rows[start:start + batch_size]
            prompts = [tokenizer.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True) for row in group]
            inputs = tokenizer(prompts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
            output = model.generate(**inputs, generation_config=generation)
            generated = output[:, inputs.input_ids.shape[1]:]
            completions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            results = []
            for trial, tokens, text in zip(group, generated.tolist(), completions):
                end = next((index + 1 for index, token in enumerate(tokens) if token in stop_ids), None)
                results.append({"custom_id": trial.get("custom_id", trial["case_id"]), "result": {
                    "type": "succeeded", "message": {
                        "content": [{"type": "text", "text": text}],
                        "stop_reason": "eos" if end is not None else "length",
                        "usage": {"generated_tokens": end if end is not None else len(tokens)},
                    },
                }})
            yield results


def saved_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[:data.rfind(b"\n") + 1]
        atomic_write(path, data)
    return parse_jsonl(data)


def development_trials(request_data: bytes, design_data: bytes) -> list[dict]:
    design = [row for row in parse_jsonl(design_data) if row["stage"] == "smoke"]
    if not design:
        raise ValueError("The development design has no smoke cases.")
    first_model = design[0]["model"]
    requests = {row["custom_id"]: row for row in parse_jsonl(request_data)}
    return [
        {**row, "stage": "development", "prompt": requests[row["custom_id"]]["params"]["messages"]}
        for row in design if row["model"] == first_model
    ]


def evaluate(args) -> None:
    task, directory, output = args.task, args.data, args.output
    filename = EVAL_FILES[task]
    data = verified_file(directory, filename)
    is_math = task in {"validation", "test"}
    if task == "development":
        design_data = verified_file(directory, "design.jsonl")
        rows = development_trials(data, design_data)
        digest = sha256(data + design_data)
    else:
        rows = verified_math(directory, task) if is_math else parse_jsonl(data)
        digest = sha256(data)
    default_max_tokens = 256 if is_math else 32 if task == "direct" else 16
    max_tokens = args.max_new_tokens if args.max_new_tokens is not None else default_max_tokens
    if max_tokens < 1 or args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Batch size, completion length, and evaluation limit must be positive.")
    expected = len(rows)
    if args.limit is not None:
        rows = rows[:args.limit]
    reference = model_reference(args.model, args.revision)
    adapter = None if args.adapter is None else args.adapter.resolve()
    metadata = {
        "task": task, "dataset_sha256": digest, "model": reference, "greedy": True,
        "batch_size": args.batch_size, "max_new_tokens": max_tokens, "device": args.device,
        "adapter": None if adapter is None else {"path": str(adapter), "sha256": fingerprint(adapter)},
        "planned": len(rows), "full_dataset": len(rows) == expected, "packages": package_versions(),
    }
    if adapter is not None:
        config_path = adapter.parent / "training_config.json"
        if config_path.is_file():
            training_config = json.loads(config_path.read_bytes())
            if training_config["model"] != reference:
                raise ValueError("Adapter was trained from a different model revision.")
            metadata["training"] = training_config
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output / "evaluation.lock", "an evaluation"):
        config_path, result_path = output / "evaluation_config.json", output / "results.jsonl"
        if config_path.exists():
            if json.loads(config_path.read_bytes()) != metadata:
                raise ValueError("The existing evaluation configuration differs from this request.")
        else:
            if result_path.exists():
                raise ValueError("Existing results have no evaluation configuration.")
            atomic_write(config_path, json_bytes(metadata))
        results = saved_results(result_path)
        planned_ids = {row.get("custom_id", row["case_id"]) for row in rows}
        if len(planned_ids) != len(rows):
            raise ValueError("Evaluation dataset contains duplicate IDs.")
        completed = index_results(results, planned_ids)
        pending = [row for row in rows if row.get("custom_id", row["case_id"]) not in completed]
        if pending:
            model, tokenizer = load_model(reference, args.device, args.offline, adapter)
            with result_path.open("ab") as stream:
                for group in inference_batches(model, tokenizer, pending, args.device, args.batch_size, max_tokens):
                    stream.write(jsonl_bytes(group))
                    stream.flush()
                    results.extend(group)
                    if len(results) == len(rows) or len(results) % (args.batch_size * 10) == 0:
                        print(f"Saved {len(results)}/{len(rows)} {task} responses", flush=True)
        if is_math:
            summary = score_math(rows, results)
        elif task == "direct":
            summary = score_direct(rows, results)
        else:
            design = [{key: value for key, value in row.items() if key != "prompt"} | {"model": output.name} for row in rows]
            summary, observations = score_results(design, results, task)
            if task in {"development", "diagnostics"}:
                summary["diagnostic_accuracy"] = score_diagnostics(observations)
        atomic_write(output / "summary.json", json_bytes(summary))
        print(f"Summary written to {output / 'summary.json'}")
