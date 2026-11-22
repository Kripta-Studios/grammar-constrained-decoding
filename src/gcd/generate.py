"""Sampling loop with logits masking (Hugging Face transformers model, KV cache).

CPU/GPU overlap: CUDA kernel launches are asynchronous, so after `model(...)` returns
the GPU is still computing the logits for step t. The grammar mask for step t only
depends on tokens < t, so we compute it on the CPU *while* the forward pass runs and
only synchronize when we sample. XGrammar/vLLM do the same, with the mask computation
on a separate thread; here a single thread is enough because torch does not block.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .mask import MaskEngine


@dataclass
class GenResult:
    token_ids: list[int]
    text: str
    finished: bool  # stopped on EOS (vs. hit max_new_tokens)
    step_ms: list[float] = field(default_factory=list)  # wall time per generated token
    mask_ms: list[float] = field(default_factory=list)  # CPU time spent building masks


@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt_ids: list[int],
    eos_ids: list[int],
    engine: MaskEngine | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    seed: int = 0,
) -> GenResult:
    dev = model.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    matcher = engine.pda.matcher() if engine else None
    ids = torch.tensor([prompt_ids], device=dev)
    past = None
    out: list[int] = []
    res = GenResult([], "", False)
    for _ in range(max_new_tokens):
        t0 = time.perf_counter()
        o = model(input_ids=ids, past_key_values=past, use_cache=True)  # async launch
        past = o.past_key_values
        logits = o.logits[0, -1].float()
        if engine is not None:
            tm = time.perf_counter()
            allowed = engine.mask(matcher)  # runs on CPU while the GPU computes `logits`
            res.mask_ms.append((time.perf_counter() - tm) * 1e3)
            allowed_t = torch.from_numpy(allowed[: logits.shape[0]]).to(dev, non_blocking=True)
            logits = logits.masked_fill(~allowed_t, float("-inf"))
        if temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=gen))  # .item() syncs here
        res.step_ms.append((time.perf_counter() - t0) * 1e3)
        if nxt in eos_ids:
            res.finished = True
            break
        out.append(nxt)
        if matcher is not None:
            tok = engine.vocab.tokens[nxt]  # type: ignore[union-attr]
            if tok is None or not matcher.advance(tok):  # impossible if the mask is right
                raise RuntimeError(f"sampled token {nxt} outside the grammar mask")
        ids = torch.tensor([[nxt]], device=dev)
    res.token_ids = out
    res.text = tokenizer.decode(out, skip_special_tokens=True)
    return res


def chat_prompt(tokenizer, user: str) -> list[int]:
    """Qwen3 chat template with thinking disabled (we want the JSON directly)."""
    msgs = [{"role": "user", "content": user}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def summarize_ms(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": float("nan"), "p50": float("nan"), "p99": float("nan")}
    a = np.asarray(xs)
    return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99))}
