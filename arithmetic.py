from __future__ import annotations

import ast
import hashlib
import itertools
import operator
import random
import re
from functools import lru_cache


SPLIT_COUNTS = {"train": 4096, "validation": 1000, "test": 1000}
ARITHMETIC_PROFILES = {
    "standard": {"easy": (3, 0, -12, 12), "medium": (4, 1, -19, 19), "hard": (5, 2, -25, 25)},
    "small-integers": {"easy": (3, 0, 0, 9), "medium": (3, 1, 0, 9), "hard": (4, 1, 0, 9)},
}
ANSWER_FORMATS = ("worked", "short")
OPERATIONS = {"+": operator.add, "-": operator.sub, "*": operator.mul}
AST_OPERATIONS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
FINAL_LINE = re.compile(r"[ \t]*Final answer: ([+-]?(?:0|[1-9][0-9]{0,11}))[ \t]*")
ANSWER_MARKER = re.compile(r"[ \t]*final answer[ \t]*:", re.IGNORECASE)


def math_prompt(expression: str, answer_format: str = "worked") -> list[dict]:
    instructions = {
        "worked": "You may show your work. End with one line in the form Final answer: <signed integer>.",
        "short": "Reply with one line only, in the form Final answer: <signed integer>.",
    }
    if answer_format not in instructions:
        raise ValueError(f"Unknown arithmetic answer format: {answer_format}.")
    return [{"role": "user", "content": (
        f"Evaluate this arithmetic expression exactly.\n\n{expression}\n\n{instructions[answer_format]}"
    )}]


def parse_answer(text: str) -> int | None:
    lines = text.strip().splitlines()
    if not lines or sum(bool(ANSWER_MARKER.match(line)) for line in lines) != 1:
        return None
    match = FINAL_LINE.fullmatch(lines[-1])
    return int(match[1]) if match else None


def completion_text(completion: str | list[dict]) -> str:
    if isinstance(completion, str):
        return completion
    if len(completion) == 1 and completion[0].get("role") == "assistant":
        text = completion[0].get("content")
        if isinstance(text, str):
            return text
    raise ValueError("Expected a single assistant text completion.")


def independently_evaluate(expression: str) -> int:
    """Check rendered expressions without executing them or using the generator tree."""
    if not re.fullmatch(r"[0-9+*() \-]{1,256}", expression):
        raise ValueError("Expression is outside the arithmetic grammar.")

    def visit(node: ast.AST) -> int:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in AST_OPERATIONS:
            return AST_OPERATIONS[type(node.op)](visit(node.left), visit(node.right))
        raise ValueError("Expression is outside the arithmetic grammar.")

    try:
        return visit(ast.parse(expression, mode="eval").body)
    except SyntaxError as error:
        raise ValueError("Invalid arithmetic expression.") from error


@lru_cache(maxsize=None)
def shapes(leaves: int) -> tuple:
    if leaves == 1:
        return (None,)
    return tuple(
        (left, right)
        for split in range(1, leaves)
        for left in shapes(split)
        for right in shapes(leaves - split)
    )


def template_splits(seed: int, tiers: dict) -> dict[str, dict[str, list[tuple]]]:
    rng = random.Random(seed)
    splits: dict[str, dict[str, list[tuple]]] = {split: {} for split in SPLIT_COUNTS}
    for tier, (leaves, products, _, _) in tiers.items():
        templates = [
            (shape, signs)
            for shape in shapes(leaves)
            for signs in itertools.product(OPERATIONS, repeat=leaves - 1)
            if signs.count("*") == products
        ]
        rng.shuffle(templates)
        held_out = max(2, len(templates) // 5)
        splits["validation"][tier] = templates[:held_out]
        splits["test"][tier] = templates[held_out:2 * held_out]
        splits["train"][tier] = templates[2 * held_out:]
    return splits


def instantiate(shape, signs, rng: random.Random, minimum: int, maximum: int):
    if shape is None:
        return rng.randint(minimum, maximum)
    left, right = shape
    return next(signs), instantiate(left, signs, rng, minimum, maximum), instantiate(right, signs, rng, minimum, maximum)


def render(node) -> str:
    if isinstance(node, int):
        return f"({node})" if node < 0 else str(node)
    sign, left, right = node
    return f"({render(left)} {sign} {render(right)})"


def solve(node, work: list[str]) -> int:
    if isinstance(node, int):
        return node
    sign, left, right = node
    first, second = solve(left, work), solve(right, work)
    answer = OPERATIONS[sign](first, second)
    work.append(f"{first} {sign} {second} = {answer}")
    return answer


def check_item(row: dict) -> None:
    expression = row["expression"]
    answer_format = row.get("answer_format", "worked")
    if independently_evaluate(expression) != row["answer"]:
        raise ValueError(f"Reference disagreement for {row['case_id']}.")
    if row["prompt"] != math_prompt(expression, answer_format):
        raise ValueError(f"Unexpected prompt for {row['case_id']}.")
    if "completion" in row:
        text = completion_text(row["completion"])
        if parse_answer(text) != row["answer"]:
            raise ValueError(f"Reference completion disagreement for {row['case_id']}.")
        if answer_format == "short" and text != f"Final answer: {row['answer']}":
            raise ValueError(f"Expected a one-line reference completion for {row['case_id']}.")


def make_arithmetic(
    seed: int, counts: dict[str, int] | None = None, profile: str = "standard", answer_format: str = "worked",
) -> dict[str, list[dict]]:
    counts = SPLIT_COUNTS if counts is None else counts
    rng = random.Random(seed)
    tier_specs = ARITHMETIC_PROFILES[profile]
    templates = template_splits(seed, tier_specs)
    result: dict[str, list[dict]] = {split: [] for split in SPLIT_COUNTS}
    seen: set[str] = set()
    tiers = tuple(tier_specs)
    for split, rows in result.items():
        for index in range(counts[split]):
            tier = tiers[index % len(tiers)]
            candidates = templates[split][tier]
            template = candidates[(index // len(tiers)) % len(candidates)]
            shape, signs = template
            _, _, minimum, maximum = tier_specs[tier]
            while True:
                node = instantiate(shape, iter(signs), rng, minimum, maximum)
                expression = render(node)
                if expression not in seen:
                    seen.add(expression)
                    break
            work: list[str] = []
            answer = solve(node, work)
            row = {
                "case_id": f"math-{split}-{index + 1:05d}",
                "tier": tier,
                "template": hashlib.sha256(repr(template).encode()).hexdigest()[:16],
                "expression": expression,
                "answer": answer,
                "prompt": math_prompt(expression, answer_format),
            }
            if answer_format != "worked":
                row["answer_format"] = answer_format
            if split == "train":
                final = f"Final answer: {answer}"
                text = "\n".join([*work, final]) if answer_format == "worked" else final
                row["completion"] = [{"role": "assistant", "content": text}]
            check_item(row)
            rows.append(row)
    return result


def audit_reward_rows(rows: list[dict], reference_by_id: dict[str, dict]) -> dict:
    positives = 0
    seen = set()
    for row in rows:
        case_id = row["case_id"]
        reference = reference_by_id[case_id]
        if row["expression"] != reference["expression"] or row["answer"] != reference["answer"]:
            raise ValueError(f"Audit input differs from the training corpus for {case_id}.")
        answer = independently_evaluate(row["expression"])
        expected = int(parse_answer(row["completion"]) == answer)
        if answer != row["answer"] or expected != row["reward"]:
            raise ValueError(f"Reward audit mismatch for {case_id}.")
        positives += expected
        seen.add(case_id)
    return {"completions": len(rows), "positive_rewards": positives, "distinct_prompts": len(seen)}
