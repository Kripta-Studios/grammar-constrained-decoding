"""Token mask computation: implicit token trie + context-independent token cache.

The vocabulary is sorted by token bytes. Neighbouring tokens share a prefix, so a
depth-first walk of the token *trie* is just a linear scan of the sorted list where,
for token i, we reuse the PDA states of the first lcp[i] bytes (longest common prefix
with token i-1) and only feed the new suffix. When a prefix is rejected we skip the
whole subtree (every following token that shares that prefix). This is the layout
XGrammar uses; it needs two int arrays instead of a pointer-heavy trie.

The key idea of XGrammar is the *adaptive token mask cache*. For a stack whose top is
node `n`, run every token starting from the single-frame stack (n,):
  * accepted inside the rule  -> valid whatever the caller is   (context-independent)
  * rejected inside the rule  -> invalid whatever the caller is (context-independent)
  * left the rule before its last byte -> depends on the caller (context-dependent)
The first two classes are computed once per node and stored. At decode time only the
(usually few) context-dependent tokens are re-checked against the real stack.

Context expansion. With only the top node known, tiny rules make almost everything
context-dependent: inside a string the top node is `char`, which finishes after ONE
byte, so every multi-byte token "escapes". XGrammar fixes this with grammar inlining
and "context expansion". We do the simplest version: the cache key is the top
`context_frames` frames of the stack (default 3), so the precomputation already knows
that `char` returns into `chars` which returns into `string`. Only tokens that close
the string *and* continue (e.g. `",`) remain context-dependent. Keys are discovered
lazily at decode time (XGrammar precomputes per node ahead of time).
"""
from __future__ import annotations

import bisect
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .pda import PDA, Matcher, Stacks


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-level BPE alphabet (also used by Qwen): byte -> printable char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


@dataclass
class TokenVocab:
    """Token id -> raw bytes. Tokens with `None` (special/added tokens) are never allowed
    by the grammar; `eos_ids` are allowed exactly when the grammar may stop."""

    tokens: list[bytes | None]
    eos_ids: list[int]
    size: int = 0  # logits width (may exceed len(tokens): padded embedding rows)
    order: np.ndarray = field(init=False)
    lcp: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.size = max(self.size, len(self.tokens))
        ids = [i for i, t in enumerate(self.tokens) if t]
        self.order, self.lcp = sorted_with_lcp(self.tokens, ids)

    @classmethod
    def from_hf(cls, tokenizer, eos_ids: Sequence[int], size: int = 0) -> "TokenVocab":
        dec = {c: b for b, c in bytes_to_unicode().items()}
        added = set(tokenizer.get_added_vocab().values())
        toks: list[bytes | None] = []
        for i, piece in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
            if i in added or piece is None or any(c not in dec for c in piece):
                toks.append(None)
            else:
                toks.append(bytes(dec[c] for c in piece))
        return cls(toks, list(eos_ids), size)


def sorted_with_lcp(tokens: Sequence[bytes | None], ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(ids, key=lambda i: tokens[i])  # type: ignore[arg-type,return-value]
    lcp = np.zeros(len(order), dtype=np.int32)
    for k in range(1, len(order)):
        a, b = tokens[order[k - 1]], tokens[order[k]]
        n, m = 0, min(len(a), len(b))  # type: ignore[arg-type]
        while n < m and a[n] == b[n]:  # type: ignore[index]
            n += 1
        lcp[k] = n
    return np.asarray(order, dtype=np.int32), lcp


@dataclass
class NodeEntry:
    """Adaptive storage (as in XGrammar): keep whichever id list is shorter."""

    ids: np.ndarray  # accepted ids if `store_accepted`, else rejected ids
    store_accepted: bool
    n_accepted: int
    cd_order: np.ndarray  # context-dependent token ids (sorted by bytes)
    cd_lcp: np.ndarray


@dataclass
class MaskStats:
    calls: int = 0
    total_ms: float = 0.0
    cd_checked: int = 0  # context-dependent tokens re-checked at runtime
    precompute_ms: float = 0.0
    nodes_cached: int = 0
    cd_tokens_cached: int = 0  # sum over cache entries of context-dependent tokens


class MaskEngine:
    def __init__(self, pda: PDA, vocab: TokenVocab, use_cache: bool = True, context_frames: int = 3):
        self.pda, self.vocab, self.use_cache, self.k = pda, vocab, use_cache, context_frames
        self.cache: dict[tuple[int, ...], NodeEntry] = {}
        self.stats = MaskStats()

    # ------------------------------------------------------------- the scan
    def scan(self, start: Stacks, order: np.ndarray, lcp: np.ndarray, ci: bool) -> tuple[list[int], list[int]]:
        """Walk tokens `order` (sorted, with `lcp`) from PDA state `start`.

        Returns (accepted ids, context-dependent ids). `ci=False` means `start` is a real
        stack, so "popped the bottom frame" means "grammar complete" instead of "escaped".
        """
        toks, step = self.vocab.tokens, self.pda.step
        # Per depth d of the current trie path (d bytes consumed):
        states: list[Stacks] = [start]  # live stacks
        pops: list[bool] = [False]  # the byte at depth d popped the bottom frame
        esc: list[bool] = [False]  # some pop happened at depth <= d
        acc: list[int] = []
        cd: list[int] = []
        i, n = 0, len(order)
        order_l, lcp_l = order.tolist(), lcp.tolist()
        keys = [toks[t] for t in order_l]
        while i < n:
            tid = order_l[i]
            tok: bytes = toks[tid]  # type: ignore[assignment]
            d = lcp_l[i]
            del states[d + 1 :], pops[d + 1 :], esc[d + 1 :]
            dead_at = -1
            while d < len(tok):
                ns, popped = step(states[d], tok[d])
                d += 1
                states.append(ns)
                pops.append(popped)
                esc.append(esc[-1] or popped)
                if not ns and d < len(tok):
                    dead_at = d
                    break
            if dead_at < 0:
                L = len(tok)
                # Valid if a path is still alive, or the bottom frame finished exactly on the
                # last byte (rule/grammar complete: always fine, whatever comes next).
                if states[L] or pops[L]:
                    acc.append(tid)
                elif ci and esc[L - 1]:  # escaped into the caller with bytes left over
                    cd.append(tid)
                if states[L]:
                    i += 1
                    continue
                # Dead exactly at the end of this token: its extensions are dead too.
                dead_at, i = L, i + 1
                if i >= n or lcp_l[i] < L:
                    continue  # no extension follows (cheap common case)
            # Prefix tok[:dead_at] has no live path: every following token sharing it has
            # the same fate. CD if (in CI mode) some path escaped on the way, else rejected.
            # The subtree ends at the first token >= the prefix with its last byte
            # incremented: a binary search on the sorted bytes instead of an O(|V|) skip.
            succ = _successor(tok[:dead_at])
            j = n if succ is None else bisect.bisect_left(keys, succ, i)
            if ci and esc[dead_at]:
                cd.extend(order_l[i:j])
            i = j
        return acc, cd

    def _entry(self, key: tuple[int, ...]) -> NodeEntry:
        e = self.cache.get(key)
        if e is None:
            t0 = time.perf_counter()
            acc, cd = self.scan(frozenset({key}), self.vocab.order, self.vocab.lcp, ci=True)
            cd_order, cd_lcp = sorted_with_lcp(self.vocab.tokens, cd)
            n = len(self.vocab.order)
            if len(acc) <= n // 2:
                ids, store_acc = np.asarray(acc, dtype=np.int64), True
            else:
                rej = np.ones(self.vocab.size, dtype=np.bool_)
                rej[acc] = False
                ids, store_acc = np.flatnonzero(rej), False  # includes special/padded ids
            e = NodeEntry(ids, store_acc, len(acc), cd_order, cd_lcp)
            self.cache[key] = e
            self.stats.precompute_ms += (time.perf_counter() - t0) * 1e3
            self.stats.nodes_cached += 1
            self.stats.cd_tokens_cached += len(cd)
        return e

    # ------------------------------------------------------------ the mask
    def mask(self, m: Matcher) -> np.ndarray:
        t0 = time.perf_counter()
        out = np.zeros(self.vocab.size, dtype=np.bool_)
        if self.use_cache:
            for s in m.stacks:
                e = self._entry(s[-self.k :])
                if e.store_accepted:
                    out[e.ids] = True
                else:
                    allowed = np.ones(self.vocab.size, dtype=np.bool_)
                    allowed[e.ids] = False
                    out |= allowed
                if len(e.cd_order):
                    self.stats.cd_checked += len(e.cd_order)
                    acc, _ = self.scan(frozenset({s}), e.cd_order, e.cd_lcp, ci=False)
                    out[acc] = True
        elif m.stacks:
            acc, _ = self.scan(m.stacks, self.vocab.order, self.vocab.lcp, ci=False)
            out[acc] = True
        if m.can_end:
            out[self.vocab.eos_ids] = True
        self.stats.calls += 1
        self.stats.total_ms += (time.perf_counter() - t0) * 1e3
        return out


def _successor(prefix: bytes) -> bytes | None:
    """Smallest byte string greater than every string starting with `prefix` (None: none)."""
    p = bytearray(prefix)
    while p and p[-1] == 0xFF:
        p.pop()
    if not p:
        return None
    p[-1] += 1
    return bytes(p)


def brute_force_mask(pda: PDA, vocab: TokenVocab, m: Matcher) -> np.ndarray:
    """Reference implementation: try every token independently. O(|V| * len) per step."""
    out = np.zeros(vocab.size, dtype=np.bool_)
    for tid, tok in enumerate(vocab.tokens):
        if tok:
            stacks, can_end = m.stacks, m.can_end
            for b in tok:
                stacks, can_end = pda.step(stacks, b)
                if not stacks and not can_end:
                    break
            out[tid] = bool(stacks) or can_end
    if m.can_end:
        out[vocab.eos_ids] = True
    return out
