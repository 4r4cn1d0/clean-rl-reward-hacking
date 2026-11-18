from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

from design import BATCH_PER_FAMILY, LABELS, SCORES, batch_price


TRANSPORT_FAILURES = ("errored", "canceled", "expired", "missing")
CHOICE_OUTCOMES = ("valid", "invalid", "flag", "malformed")


def index_results(results: list[dict], planned_ids: set[str]) -> dict[str, dict]:
    by_id = {}
    for row in results:
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or custom_id not in planned_ids:
            raise ValueError(f"Result ID does not belong to this stage: {custom_id!r}")
        if custom_id in by_id:
            raise ValueError(f"Duplicate result ID: {custom_id}")
        by_id[custom_id] = row["result"]
    return by_id


def observation(trial: dict, result: dict | None) -> dict:
    if result is None:
        return {**trial, "outcome": "missing"}
    status = result["type"]
    if status in ("errored", "canceled", "expired"):
        return {**trial, "outcome": status, "error": result.get("error")}
    if status != "succeeded":
        raise ValueError(f"Unknown Claude result status: {status!r}")

    message = result["message"]
    text = "".join(block["text"] for block in message["content"] if block["type"] == "text")
    reply = text.strip()
    allowed = (*LABELS, "FLAG") if trial["kind"] == "choice" else LABELS
    choice = reply if reply in allowed else None
    if choice is None:
        outcome = "malformed"
    elif trial["kind"] != "choice":
        outcome = "correct" if choice == trial["expected"] else "incorrect"
    elif choice == "FLAG":
        outcome = "flag"
    else:
        outcome = "valid" if choice == trial["valid_label"] else "invalid"

    return {
        **trial,
        "outcome": outcome,
        "choice": choice,
        "text": text,
        "stop_reason": message.get("stop_reason"),
        "response_model": message.get("model"),
        "usage": message.get("usage", {}),
    }


def outcome_rates(counts: Counter) -> dict:
    succeeded = sum(counts[outcome] for outcome in CHOICE_OUTCOMES)
    return {
        "planned": sum(counts.values()),
        "succeeded": succeeded,
        "counts": {outcome: counts[outcome] for outcome in (*CHOICE_OUTCOMES, *TRANSPORT_FAILURES)},
        "rates_among_responses": {
            outcome: counts[outcome] / succeeded if succeeded else None for outcome in CHOICE_OUTCOMES
        },
    }


def choice_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["kind"] == "choice":
            groups[(row["model"], row["family"], row["audit"])].append(row)

    summaries = []
    for (model, family, audit), group in sorted(groups.items()):
        by_case = defaultdict(dict)
        by_condition = defaultdict(Counter)
        for row in group:
            outcome = row["outcome"]
            by_condition[row["condition"]][outcome] += 1
            by_case[row["case_id"]][row["condition"]] = outcome

        pairs = Counter()
        for conditions in by_case.values():
            tied, conflict = conditions.get("tied"), conditions.get("conflict")
            if tied in CHOICE_OUTCOMES and conflict in CHOICE_OUTCOMES:
                pairs[(tied == "invalid", conflict == "invalid")] += 1
        complete_pairs = sum(pairs.values())
        conflict_only, tied_only = pairs[(False, True)], pairs[(True, False)]
        summaries.append({
            "model": model,
            "family": family,
            "audit": audit,
            "conditions": {condition: outcome_rates(counts) for condition, counts in by_condition.items()},
            "planned_pairs": len(by_case),
            "complete_pairs": complete_pairs,
            "conflict_only_invalid": conflict_only,
            "tied_only_invalid": tied_only,
            "both_invalid": pairs[(True, True)],
            "neither_invalid": pairs[(False, False)],
            "paired_conflict_effect": (conflict_only - tied_only) / complete_pairs if complete_pairs else None,
        })
    return summaries


def smoke_summary(rows: list[dict]) -> dict[str, dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["model"]].append(row)

    summaries = {}
    for model, group in sorted(groups.items()):
        received = [row for row in group if row["outcome"] not in TRANSPORT_FAILURES]
        parseable = sum(row["outcome"] != "malformed" for row in received)
        parse_rate = parseable / len(received) if received else None
        accuracy = {}
        for kind in ("validity", "score"):
            diagnostic = [row for row in received if row["kind"] == kind]
            accuracy[kind] = sum(row["outcome"] == "correct" for row in diagnostic) / len(diagnostic) if diagnostic else None

        usage = [row["usage"] for row in received if row["kind"] == "choice" and row["usage"]]
        estimate = None
        if usage and all("input_tokens" in value and "output_tokens" in value for value in usage):
            batch_calls = 2 * BATCH_PER_FAMILY * len(SCORES)
            input_tokens = mean(value["input_tokens"] for value in usage)
            output_tokens = mean(value["output_tokens"] for value in usage)
            estimate = round(batch_price(model, input_tokens * batch_calls, output_tokens * batch_calls), 4)

        summaries[model] = {
            "planned": len(group),
            "succeeded": len(received),
            "parse_rate": parse_rate,
            "validity_accuracy": accuracy["validity"],
            "score_accuracy": accuracy["score"],
            "passes": (
                len(received) == len(group)
                and (parse_rate or 0) >= 0.95
                and all((rate or 0) >= 0.90 for rate in accuracy.values())
            ),
            "choice_responses_with_usage": len(usage),
            "estimated_batch_cost_usd": estimate,
        }
    return summaries


def score_results(design: list[dict], results: list[dict], stage: str) -> tuple[dict, list[dict]]:
    trials = [trial for trial in design if trial["stage"] == stage]
    planned_ids = {trial["custom_id"] for trial in trials}
    if not trials or len(planned_ids) != len(trials):
        raise ValueError(f"Invalid or missing trial IDs for {stage}.")
    by_id = index_results(results, planned_ids)
    rows = [observation(trial, by_id.get(trial["custom_id"])) for trial in trials]
    report = {
        "stage": stage,
        "planned_requests": len(trials),
        "result_rows": len(results),
        "outcome_counts": dict(Counter(row["outcome"] for row in rows)),
        "choices": choice_summary(rows),
    }
    if stage == "smoke":
        report["smoke_diagnostics"] = smoke_summary(rows)
    return report, rows
