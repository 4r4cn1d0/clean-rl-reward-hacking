from __future__ import annotations

import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from arithmetic import audit_reward_rows, completion_text, parse_answer
from pilot import atomic_write, json_bytes, jsonl_bytes, parse_jsonl, run_lock, sha256, verified_file
from qwen_data import verified_math
from qwen_model import load_model, model_reference, package_versions


PROMPTS_PER_UPDATE = 8
GENERATIONS = 8
LORA = {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}


def exact_reward(stream):
    call = 0

    def exact_arithmetic(completions, case_id, expression, answer, **kwargs):
        nonlocal call
        call += 1
        rows = []
        if len({len(completions), len(case_id), len(expression), len(answer)}) != 1:
            raise ValueError("Reward inputs have different batch lengths.")
        for completion, item, task, reference in zip(completions, case_id, expression, answer):
            text = completion_text(completion)
            rows.append({
                "call": call, "case_id": item, "expression": task, "answer": reference,
                "completion": text, "reward": int(parse_answer(text) == reference),
            })
        stream.write(jsonl_bytes(rows))
        stream.flush()
        return [float(row["reward"]) for row in rows]

    return exact_arithmetic


def reward_audit(data: Path, run: Path) -> dict:
    train = {row["case_id"]: row for row in verified_math(data, "train")}
    trace_files = sorted(run.glob("reward_audit_*.jsonl"))
    if not trace_files:
        raise ValueError("This run has no reward audit records.")
    records = []
    groups = defaultdict(list)
    partial_traces = 0
    for path in trace_files:
        data_bytes = path.read_bytes()
        if data_bytes and not data_bytes.endswith(b"\n"):
            data_bytes = data_bytes[:data_bytes.rfind(b"\n") + 1]
            partial_traces += 1
        for row in parse_jsonl(data_bytes):
            groups[(path.name, row["call"], row["case_id"])].append(row["reward"])
            records.append(row)
    summary = audit_reward_rows(records, train)
    complete = [rewards for rewards in groups.values() if len(rewards) == GENERATIONS]
    mixed = sum(len(set(rewards)) > 1 for rewards in complete)
    return {
        **summary,
        "complete_groups": len(complete),
        "mixed_reward_groups": mixed,
        "mixed_reward_group_fraction": mixed / len(complete) if complete else None,
        "partial_trace_files": partial_traces,
    }


def last_checkpoint(directory: Path) -> str | None:
    complete = []
    for path in directory.glob("checkpoint-*"):
        step = path.name.removeprefix("checkpoint-")
        files = ("trainer_state.json", "adapter_config.json", "adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth")
        if step.isdigit() and all((path / name).is_file() for name in files):
            complete.append((int(step), path))
    return str(max(complete)[1]) if complete else None


def train(args) -> None:
    import pyarrow as pa
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

    def dataset(samples: list[dict]):
        fingerprint = sha256(jsonl_bytes(samples))[:16]
        return Dataset(pa.Table.from_pylist(samples), fingerprint=fingerprint)

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This training configuration uses a single process.")
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Expose one GPU for this training configuration, for example with CUDA_VISIBLE_DEVICES=0.")
    data = verified_math(args.data, "train")
    count = args.steps * PROMPTS_PER_UPDATE
    if not 0 < count <= len(data):
        raise ValueError(f"Choose 1 to {len(data) // PROMPTS_PER_UPDATE} training updates.")
    if not 0 < args.micro_batch <= PROMPTS_PER_UPDATE or PROMPTS_PER_UPDATE % args.micro_batch:
        raise ValueError("Microbatch size must divide the eight prompts in an update.")
    if args.max_completion_tokens < 1:
        raise ValueError("Completion length must be positive.")
    reference = model_reference(args.model, args.revision)
    run = args.output
    records = data[:count]
    rate = getattr(args, "learning_rate", None)
    if rate is None:
        rate = 1e-5 if args.arm == "grpo" else 5e-5
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    metadata = {
        "arm": args.arm, "model": reference, "seed": args.seed, "steps": args.steps,
        "device": args.device, "micro_batch": args.micro_batch, "learning_rate": rate,
        "max_completion_tokens": args.max_completion_tokens, "prompts_per_update": PROMPTS_PER_UPDATE,
        "generations_per_prompt": GENERATIONS if args.arm == "grpo" else None,
        "training_corpus_sha256": sha256(verified_file(args.data, "math_train.jsonl")),
        "used_prompt_ids_sha256": sha256(json_bytes([row["case_id"] for row in records])),
        "lora": LORA, "packages": package_versions(),
    }
    if args.resume and not (run / "training_config.json").is_file():
        raise ValueError("There is no existing training configuration to resume.")
    run.mkdir(parents=True, exist_ok=args.resume)
    with run_lock(run / "training.lock", "training"):
        if args.resume:
            if json.loads((run / "training_config.json").read_bytes()) != metadata:
                raise ValueError("The existing training configuration differs from this request.")
            if (run / "training_summary.json").exists():
                raise ValueError("This training run is complete.")
            checkpoint = last_checkpoint(run)
        else:
            atomic_write(run / "training_config.json", json_bytes(metadata))
            checkpoint = None

        set_seed(args.seed)
        model, tokenizer = load_model(reference, args.device, args.offline)
        lora = LoraConfig(**LORA, task_type="CAUSAL_LM")
        config = dict(
            output_dir=str(run), max_steps=args.steps, num_train_epochs=1,
            per_device_train_batch_size=args.micro_batch, learning_rate=rate,
            seed=args.seed, data_seed=args.seed, logging_steps=1, report_to="none",
            bf16=args.device == "cuda", use_cpu=args.device == "cpu", gradient_checkpointing=False,
            save_strategy="steps", save_steps=max(1, args.steps // 4), dataloader_drop_last=True,
        )
        if args.arm == "grpo":
            batch = PROMPTS_PER_UPDATE * GENERATIONS
            config.update(
                gradient_accumulation_steps=batch // args.micro_batch, generation_batch_size=batch,
                max_completion_length=args.max_completion_tokens, num_generations=GENERATIONS,
                num_iterations=1, temperature=0.8, beta=0.01, loss_type="grpo", scale_rewards="group",
                use_vllm=False, remove_unused_columns=False,
            )
            samples = [{key: row[key] for key in ("prompt", "case_id", "expression", "answer")} for row in records]
            with tempfile.NamedTemporaryFile(dir=run, prefix="reward_audit_", suffix=".jsonl", delete=False) as stream:
                trainer = GRPOTrainer(
                    model=model, args=GRPOConfig(**config), processing_class=tokenizer,
                    train_dataset=dataset(samples), reward_funcs=exact_reward(stream), peft_config=lora,
                )
                metrics = trainer.train(resume_from_checkpoint=checkpoint).metrics
        else:
            config.update(
                gradient_accumulation_steps=PROMPTS_PER_UPDATE // args.micro_batch,
                completion_only_loss=True, packing=False, max_length=512,
            )
            samples = [{key: row[key] for key in ("prompt", "completion")} for row in records]
            trainer = SFTTrainer(
                model=model, args=SFTConfig(**config), processing_class=tokenizer,
                train_dataset=dataset(samples), peft_config=lora,
            )
            metrics = trainer.train(resume_from_checkpoint=checkpoint).metrics
        trainer.save_model(str(run / "adapter"))
        summary = {"metrics": metrics, "global_step": trainer.state.global_step, "training_prompt_count": count}
        if args.arm == "grpo":
            summary["reward_audit"] = reward_audit(args.data, run)
        atomic_write(run / "training_summary.json", json_bytes(summary))
        print(json_bytes(summary).decode(), end="")
