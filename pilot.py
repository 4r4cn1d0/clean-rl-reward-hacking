from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from design import DEFAULT_SEED, MAX_TOKENS, MODELS, batch_price, trial_plan
from scoring import index_results, score_results


API = "https://api.anthropic.com"
MESSAGES = "/v1/messages"
BATCHES = f"{MESSAGES}/batches"
REQUEST_FILES = {"smoke": "smoke_requests.jsonl", "batch": "batch.json"}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()


def parse_jsonl(data: bytes) -> list[dict]:
    rows = []
    for number, line in enumerate(data.splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError(f"Invalid JSONL at line {number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at line {number}.")
            rows.append(row)
    return rows


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(data)
            file.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def verified_file(run: Path, name: str) -> bytes:
    manifest = json.loads((run / "manifest.json").read_bytes())
    data = (run / name).read_bytes()
    if sha256(data) != manifest["sha256"][name]:
        raise ValueError(f"Checksum mismatch for {name}. Prepare a new run to change the design.")
    return data


def write_manifested_directory(run: Path, files: dict[str, bytes], metadata: dict) -> None:
    if run.exists():
        raise ValueError(f"Run directory already exists: {run}")
    manifest = {**metadata, "sha256": {name: sha256(data) for name, data in files.items()}}
    run.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=run.parent) as temporary:
        staged_run = Path(temporary) / "run"
        staged_run.mkdir()
        for name, data in files.items():
            (staged_run / name).write_bytes(data)
        (staged_run / "manifest.json").write_bytes(json_bytes(manifest))
        staged_run.rename(run)


def prepare(run: Path, seed: int, models: list[str], audit: str) -> None:
    if run.exists():
        raise ValueError(f"Run directory already exists: {run}")
    if len(models) != len(set(models)):
        raise ValueError("Choose each model once.")
    requests, design = trial_plan(seed, models, audit)
    files = {
        REQUEST_FILES["smoke"]: jsonl_bytes(requests["smoke"]),
        REQUEST_FILES["batch"]: (json.dumps({"requests": requests["batch"]}, ensure_ascii=False, separators=(",", ":")) + "\n").encode(),
        "design.jsonl": jsonl_bytes(design),
    }
    model_counts = {name: sum(row["model"] == name and row["stage"] == "batch" for row in design) for name in models}
    estimates = [
        round(sum(batch_price(name, count * inputs, count * outputs) for name, count in model_counts.items()), 2)
        for inputs, outputs in ((800, 100), (1000, MAX_TOKENS))
    ]
    manifest = {
        "schema_version": 1,
        "seed": seed,
        "audit": audit,
        "models": {name: MODELS[name].id for name in models},
        "request_counts": {stage: len(items) for stage, items in requests.items()},
        "batch_planning_estimate_usd": estimates,
    }
    write_manifested_directory(run, files, manifest)
    print(f"Prepared {len(requests['smoke']):,} smoke requests and {len(requests['batch']):,} batch requests in {run}")
    print(f"Batch planning estimate: ${estimates[0]:.2f}–${estimates[1]:.2f}")


class ClaudeError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Claude API HTTP {status}: {message}")
        self.status = status


class Claude:
    def __init__(self) -> None:
        self.key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not self.key:
            raise ValueError("Set ANTHROPIC_API_KEY in the terminal before making live requests.")

    def request(self, path: str, data: bytes | None = None) -> bytes:
        request = Request(API + path, data=data, headers={
            "x-api-key": self.key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        })
        try:
            with urlopen(request, timeout=120) as response:
                return response.read()
        except HTTPError as error:
            message = str(error.reason)
            try:
                message = json.loads(error.read(8192))["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            raise ClaudeError(error.code, str(message).replace(self.key, "[redacted]")) from error
        except (URLError, TimeoutError) as error:
            reason = str(getattr(error, "reason", error)).replace(self.key, "[redacted]")
            raise RuntimeError(f"Claude API connection failed: {reason}") from error


@contextmanager
def run_lock(path: Path, subject: str):
    with path.open("a") as file:
        try:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"This run already has {subject} in progress.") from error
        yield


def run_smoke(run: Path) -> None:
    requests = parse_jsonl(verified_file(run, REQUEST_FILES["smoke"]))
    result_path = run / "smoke_results.jsonl"
    claude = Claude()
    with run_lock(run / "smoke.lock", "a smoke test"):
        results = parse_jsonl(result_path.read_bytes()) if result_path.exists() else []
        completed = index_results(results, {request["custom_id"] for request in requests})
        for request in requests:
            custom_id = request["custom_id"]
            if custom_id in completed:
                continue
            message = json.loads(claude.request(MESSAGES, json_bytes(request["params"])))
            results.append({"custom_id": custom_id, "result": {"type": "succeeded", "message": message}})
            atomic_write(result_path, jsonl_bytes(results))
            if len(results) == 1 or len(results) % 20 == 0:
                print(f"Smoke responses saved: {len(results)}/{len(requests)}", flush=True)
    print(f"Smoke results ready at {result_path}")


def valid_batch_id(batch_id: str) -> str:
    if not isinstance(batch_id, str) or not re.fullmatch(r"msgbatch_[a-zA-Z0-9_-]+", batch_id):
        raise ValueError(f"Invalid Claude batch ID: {batch_id!r}")
    return batch_id


def submit(run: Path) -> None:
    payload = verified_file(run, REQUEST_FILES["batch"])
    claude = Claude()
    submission = run / "submission.json"
    try:
        with submission.open("x") as file:
            json.dump({"state": "submitting", "payload_sha256": sha256(payload)}, file)
    except FileExistsError as error:
        raise ValueError("A submission already exists or was interrupted. Check its status or recover with the batch ID.") from error

    try:
        batch = json.loads(claude.request(BATCHES, payload))
    except ClaudeError as error:
        if error.status in {400, 401, 403, 404, 413, 422, 429}:
            submission.unlink()
        raise
    batch_id = valid_batch_id(batch["id"])
    atomic_write(submission, json_bytes({"state": "submitted", "payload_sha256": sha256(payload), "batch": batch}))
    print(f"Submitted Claude batch {batch_id}")


def resolve_batch_id(run: Path, explicit: str | None) -> str:
    if explicit:
        return valid_batch_id(explicit)
    state = json.loads((run / "submission.json").read_bytes())
    if state["state"] != "submitted":
        raise ValueError("Submission status is uncertain. Check Claude Console and pass its batch ID with --batch-id.")
    return valid_batch_id(state["batch"]["id"])


def status(run: Path, explicit: str | None) -> None:
    batch_id = resolve_batch_id(run, explicit)
    batch = json.loads(Claude().request(f"{BATCHES}/{batch_id}"))
    print(json_bytes(batch).decode(), end="")


def download(run: Path, explicit: str | None) -> None:
    batch_id = resolve_batch_id(run, explicit)
    claude = Claude()
    batch = json.loads(claude.request(f"{BATCHES}/{batch_id}"))
    if batch["processing_status"] != "ended":
        raise ValueError(f"Batch {batch_id} is {batch['processing_status']}. Check status before downloading.")
    payload = json.loads(verified_file(run, REQUEST_FILES["batch"]))
    results = parse_jsonl(claude.request(f"{BATCHES}/{batch_id}/results"))
    planned_ids = {request["custom_id"] for request in payload["requests"]}
    received = index_results(results, planned_ids)
    if received.keys() != planned_ids:
        raise ValueError(f"Downloaded {len(received)}/{len(planned_ids)} expected result IDs.")
    data = jsonl_bytes(sorted(results, key=lambda row: row["custom_id"]))
    result_path = run / "batch_results.jsonl"
    if result_path.exists() and result_path.read_bytes() != data:
        raise ValueError(f"A different result file already exists at {result_path}.")
    atomic_write(result_path, data)
    print(f"Downloaded {len(received):,} Claude results to {result_path}")


def score(run: Path, stage: str, result_path: Path | None) -> None:
    design = parse_jsonl(verified_file(run, "design.jsonl"))
    source = result_path or (run / f"{stage}_results.jsonl")
    results = parse_jsonl(source.read_bytes())
    report, rows = score_results(design, results, stage)
    summary_path = run / f"{stage}_summary.json"
    atomic_write(summary_path, json_bytes(report))
    atomic_write(run / f"{stage}_observations.jsonl", jsonl_bytes(rows))
    print(f"Scored {len(results):,}/{report['planned_requests']:,} result rows. Summary saved to {summary_path}")
    for model, diagnostics in report.get("smoke_diagnostics", {}).items():
        verdict = "passed" if diagnostics["passes"] else "check results"
        print(f"{model}: {verdict} ({diagnostics['succeeded']}/{diagnostics['planned']} responses)")
    for comparison in report["choices"]:
        effect = comparison["paired_conflict_effect"]
        displayed = f"{effect * 100:+.1f} pp" if effect is not None else "unavailable"
        print(f"{comparison['model']} / {comparison['family']}: {displayed} across {comparison['complete_pairs']} complete pairs")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare, run, and score the Idea 1 Claude pilot.")
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare", help="Generate both request sets offline.")
    make.add_argument("run", type=Path)
    make.add_argument("--seed", type=int, default=DEFAULT_SEED)
    make.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    make.add_argument("--audit", choices=("provided", "derived"), default="provided")
    for name, description in (("smoke", "Run live smoke requests."), ("submit", "Submit the live Claude batch."),
                              ("status", "Check an existing batch."), ("download", "Download a completed batch.")):
        command = commands.add_parser(name, help=description)
        command.add_argument("run", type=Path)
        if name in {"status", "download"}:
            command.add_argument("--batch-id", help="Use an existing batch ID when recovering a submission.")
    analyze = commands.add_parser("score", help="Score saved responses offline.")
    analyze.add_argument("run", type=Path)
    analyze.add_argument("stage", choices=REQUEST_FILES)
    analyze.add_argument("--results", type=Path, help="Read results from a supplied JSONL file.")
    args = parser.parse_args()

    try:
        if args.command == "prepare":
            prepare(args.run, args.seed, args.models, args.audit)
        elif args.command == "smoke":
            run_smoke(args.run)
        elif args.command == "submit":
            submit(args.run)
        elif args.command == "status":
            status(args.run, args.batch_id)
        elif args.command == "download":
            download(args.run, args.batch_id)
        else:
            score(args.run, args.stage, args.results)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
