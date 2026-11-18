from __future__ import annotations

import hashlib
import re
from importlib.metadata import version
from pathlib import Path


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
PACKAGES = ("torch", "transformers", "trl", "datasets", "peft", "accelerate", "pyarrow")


def package_versions() -> dict[str, str]:
    return {name: version(name) for name in PACKAGES}


def fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        file for file in path.iterdir()
        if file.is_file() and file.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".jinja"}
    )
    for file in files:
        digest.update(file.name.encode() + b"\0")
        with file.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def model_reference(name: str, revision: str | None) -> dict:
    path = Path(name)
    if path.is_dir():
        return {"name": str(path.resolve()), "revision": None, "local_sha256": fingerprint(path)}
    revision = revision or (MODEL_REVISION if name == MODEL else None)
    if revision is None or re.fullmatch(r"[a-f0-9]{40}", revision) is None:
        raise ValueError("Use a Hub commit SHA with --revision for this model.")
    return {"name": name, "revision": revision}


def load_model(reference: dict, device: str, offline: bool, adapter: Path | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use a GPU environment or choose --device cpu.")
    kwargs = {"local_files_only": offline, "revision": reference["revision"]}
    tokenizer = AutoTokenizer.from_pretrained(reference["name"], **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(reference["name"], dtype=dtype, **kwargs)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    return model, tokenizer
