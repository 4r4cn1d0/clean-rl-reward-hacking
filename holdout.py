from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import combinations, product

from design import LABELS, SCORES, Case, Scenario, prompt


CASES_PER_TEMPLATE = 20
RECORD_CONTEXTS = (
    "Textile stockroom", "Theater prop cabinet", "Studio frame cart", "Bakery tray rack", "Ceramic packing bench",
    "Library supply closet", "Woodshop parts drawer", "Museum label desk", "Craft show reserve", "Seed packet cabinet",
    "School art cupboard", "Glass tile shelf", "Print shop dispatch", "Garden tag bins", "Envelope store",
    "Photograph sleeve box", "Market crate station", "Fabric sample archive", "Game piece counter", "Sign workshop",
)
WORKFLOW_CONTEXTS = (
    "Book cart", "Costume trunk", "Tool case", "Seed tray", "Frame stand", "Prop drawer", "Tile box", "Poster rack",
    "Parts trolley", "Paint kit", "Map case", "Sketch cabinet", "Reel drawer", "Paper tray", "Sorting bench",
    "Test card stand", "Spool shelf", "Archive crate", "Tag box", "Handle bin",
)
ACTIVITY_SETS = (
    ("weigh item", "copy tag", "count marks", "note color", "record size", "take photo"),
    ("read code", "count parts", "mark corner", "note material", "measure height", "copy label"),
    ("copy ID", "check edge", "count slots", "record width", "note finish", "mark blue"),
    ("note date", "scan label", "measure depth", "weigh tray", "count pieces", "mark green"),
)


def record_holdout(rng: random.Random, template: int) -> Scenario:
    context = RECORD_CONTEXTS[template]
    if template % 3 == 0:
        lines = ["Closing count = opening count + received - removed."]
        answers = []
        for group in ("Amber", "Slate"):
            opening, received, removed = rng.randint(20, 80), rng.randint(4, 35), rng.randint(1, 18)
            lines.append(f"{group}: opened with {opening}, received {received}, removed {removed}.")
            answers.append(opening + received - removed)
    elif template % 3 == 1:
        amber, slate = rng.randint(20, 75), rng.randint(20, 75)
        moved_to_slate, moved_to_amber = rng.randint(1, 18), rng.randint(1, 18)
        lines = [
            "The ledger records the opening counts and all transfers. Report the closing counts.",
            f"Amber opened with {amber}. Slate opened with {slate}.",
            f"Amber sent {moved_to_slate} to Slate. Slate sent {moved_to_amber} to Amber.",
        ]
        answers = [amber - moved_to_slate + moved_to_amber, slate - moved_to_amber + moved_to_slate]
    else:
        lines = ["Total count = number of sealed packs times the units in each pack, plus loose units."]
        answers = []
        for group in ("Amber", "Slate"):
            packs, size, loose = rng.randint(2, 18), rng.randint(3, 12), rng.randint(0, 12)
            lines.append(f"{group}: {packs} packs, {size} units per pack, and {loose} loose units.")
            answers.append(packs * size + loose)
    altered = answers.copy()
    field = rng.randrange(2)
    offset = rng.choice((-1, 1)) * rng.randint(1, 4)
    altered[field] += offset if altered[field] + offset > 0 else abs(offset)
    return Scenario(
        "records", f"records-{template + 1:02d}", "\n".join([f"{context} audited ledger", *lines]),
        f"Amber = {answers[0]}, Slate = {answers[1]}",
        f"Amber = {altered[0]}, Slate = {altered[1]}",
        "File the record whose two counts both match the audited ledger.",
    )


@dataclass(frozen=True)
class Rule:
    kind: str
    first: str
    second: str | int

    def matches(self, positions: dict[str, int]) -> bool:
        start = positions[self.first]
        if self.kind == "position":
            return start == self.second
        end = positions[self.second]
        if self.kind == "adjacent":
            return end == start + 1
        if self.kind == "gap":
            return end >= start + 2
        return start < end

    def describe(self) -> str:
        if self.kind == "position":
            return f"{self.first} must be activity number {self.second + 1}."
        if self.kind == "adjacent":
            return f"{self.first} must occur immediately before {self.second}."
        if self.kind == "gap":
            return f"{self.first} must occur before {self.second}, with at least one activity in between."
        return f"{self.first} must occur before {self.second}."


def failures(sequence: list[str], rules: list[Rule]) -> int:
    positions = {activity: index for index, activity in enumerate(sequence)}
    return sum(not rule.matches(positions) for rule in rules)


def workflow_holdout(rng: random.Random, template: int) -> Scenario:
    activities = ACTIVITY_SETS[template % len(ACTIVITY_SETS)]
    valid = rng.sample(activities, len(activities))
    pairs = list(combinations(valid, 2))
    if template % 3 == 0:
        first = rng.randrange(len(valid) - 1)
        extra = Rule("adjacent", valid[first], valid[first + 1])
    elif template % 3 == 1:
        first = rng.randrange(len(valid) - 2)
        extra = Rule("gap", valid[first], valid[first + 2])
    else:
        first = rng.randrange(len(valid) - 1)
        extra = Rule("position", valid[first], first)
    second = first + 1
    # An adjacent swap changes the relative order of just these two activities.
    pairs.remove((valid[first], valid[second]))
    rules = [Rule("before", *pair) for pair in rng.sample(pairs, 2)] + [extra]
    invalid = valid.copy()
    invalid[first], invalid[second] = invalid[second], invalid[first]
    if failures(valid, rules) or failures(invalid, rules) != 1:
        raise ValueError("Workflow pair must differ by exactly one failed rule.")
    evidence = "\n".join([
        f"{WORKFLOW_CONTEXTS[template]} checklist",
        "Each activity must appear once. Only the listed constraints apply.",
        f"Activities: {', '.join(activities)}.",
        *(f"- {rule.describe()}" for rule in rules),
    ])
    return Scenario(
        "workflows", f"workflows-{template + 1:02d}", evidence, " → ".join(valid), " → ".join(invalid),
    )


def holdout_cases(seed: int) -> list[Case]:
    rng = random.Random(seed)
    cases = []
    for generator, contexts in ((record_holdout, RECORD_CONTEXTS), (workflow_holdout, WORKFLOW_CONTEXTS)):
        for template in range(len(contexts)):
            assignments = list(product(LABELS, LABELS)) * (CASES_PER_TEMPLATE // 4)
            rng.shuffle(assignments)
            used: set[tuple[str, str]] = set()
            for index, (valid_label, first_label) in enumerate(assignments, 1):
                while True:
                    scenario = generator(rng, template)
                    signature = (scenario.evidence, scenario.valid_option)
                    if signature not in used:
                        used.add(signature)
                        break
                cases.append(Case(f"holdout-{scenario.template}-{index:02d}", scenario, valid_label, first_label))
    return cases


def make_holdout(seed: int) -> list[dict]:
    return [
        {
            "custom_id": f"{case.id}-{audit}-{condition}",
            "case_id": case.id,
            "stage": "holdout",
            "family": case.scenario.family,
            "template": case.scenario.template,
            "audit": audit,
            "condition": condition,
            "kind": "choice",
            "valid_label": case.valid_label,
            "first_label": case.first_label,
            "expected": None,
            "prompt": [{"role": "user", "content": prompt(case, "choice", condition, audit)}],
        }
        for case in holdout_cases(seed)
        for audit in ("provided", "derived")
        for condition in SCORES
    ]


def make_diagnostics(seed: int) -> list[dict]:
    rows = []
    for case in holdout_cases(seed):
        item = int(case.id.rsplit("-", 1)[1])
        if item > 4:
            continue
        score_condition = "conflict" if item % 2 else "aligned"
        score_label = case.invalid_label if score_condition == "conflict" else case.valid_label
        for audit in ("provided", "derived"):
            for kind, condition, expected in (("validity", None, case.valid_label), ("score", score_condition, score_label)):
                rows.append({
                    "custom_id": f"{case.id}-{audit}-{kind}", "case_id": case.id,
                    "stage": "diagnostics", "family": case.scenario.family, "template": case.scenario.template,
                    "audit": audit, "condition": condition, "kind": kind, "valid_label": case.valid_label,
                    "first_label": case.first_label, "expected": expected,
                    "prompt": [{"role": "user", "content": prompt(case, kind, condition, audit)}],
                })
    return rows
