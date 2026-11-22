"""Byte-level nondeterministic pushdown automaton (PDA) for a `Grammar`.

Every grammar position `(rule, alternative, dot)` gets an integer *node id*. A
*stack* is a tuple of node ids: the last entry is where we are now, the earlier ones
are return addresses (the position *after* the rule reference that called us). The
recognizer state is a frozenset of stacks, so ambiguity (e.g. "12" may be a finished
integer or continue with "3") is handled by keeping all live alternatives, like the
persistent stack tree of XGrammar or the item sets of an Earley parser.

Two memoized primitives do all the work:
  close(stack) -> expand rule references / pop finished rules until every stack's top
                  is waiting for a byte (a "frontier" stack).
  step(stacks, byte) -> advance all frontier stacks that accept `byte`, then close.

Both return a flag `popped`: some path popped its *bottom* frame. At runtime (full stack)
that means "the root rule is complete"; during context-independent precomputation
(stack = one node, unknown caller) it means "the path escaped into the unknown caller".
"""
from __future__ import annotations

from typing import Iterable

from .grammar import ByteClass, Grammar

Stack = tuple[int, ...]
Stacks = frozenset[Stack]
_END = -1  # node kind: dot at end of alternative


class PDA:
    def __init__(self, grammar: Grammar, memo_limit: int = 2_000_000):
        grammar.validate()
        names = list(grammar.rules)
        rid = {n: i for i, n in enumerate(names)}
        self.grammar = grammar
        # Flattened node tables: node -> (kind, payload, next_node)
        self.node_class: list[frozenset[int] | None] = []  # byte class if terminal
        self.node_ref: list[int] = []  # referenced rule id, or _END, or -2 for terminal
        self.node_next: list[int] = []
        self.node_label: list[str] = []
        self.rule_starts: list[list[int]] = []  # rule id -> start node of each alternative
        for name in names:
            starts = []
            for a, alt in enumerate(grammar.rules[name]):
                base = len(self.node_class)
                starts.append(base)
                for p in range(len(alt) + 1):
                    e = alt[p] if p < len(alt) else None
                    self.node_class.append(e.allowed if isinstance(e, ByteClass) else None)
                    self.node_ref.append(rid[e] if isinstance(e, str) else (_END if e is None else -2))
                    self.node_next.append(base + p + 1 if p < len(alt) else -1)
                    self.node_label.append(f"{name}#{a}.{p}")
            self.rule_starts.append(starts)
        self.root = rid[grammar.root]
        self._close_memo: dict[Stack, tuple[Stacks, bool]] = {}
        self._step_memo: dict[tuple[Stacks, int], tuple[Stacks, bool]] = {}
        self._memo_limit = memo_limit

    @property
    def num_nodes(self) -> int:
        return len(self.node_class)

    # ------------------------------------------------------------------ core
    def close(self, stack: Stack) -> tuple[Stacks, bool]:
        hit = self._close_memo.get(stack)
        if hit is not None:
            return hit
        out: set[Stack] = set()
        popped = False
        work, seen = [stack], set()
        while work:
            s = work.pop()
            if s in seen:
                continue
            seen.add(s)
            top = s[-1]
            ref = self.node_ref[top]
            if ref == -2:  # waiting for a byte
                out.add(s)
            elif ref == _END:  # rule finished: return to caller
                if len(s) == 1:
                    popped = True
                else:
                    work.append(s[:-1])
            else:  # rule reference: push callee, remember where to return
                nxt = self.node_next[top]
                # Tail-call elimination: if the reference is the last element, returning
                # here would immediately return again, so we don't keep this frame. This
                # makes right-recursive repetition (lists, strings) run in O(1) stack depth.
                base = s[:-1] if self.node_ref[nxt] == _END else s[:-1] + (nxt,)
                for start in self.rule_starts[ref]:
                    work.append(base + (start,))
        res = (frozenset(out), popped)
        if len(self._close_memo) < self._memo_limit:
            self._close_memo[stack] = res
        return res

    def step(self, stacks: Stacks, byte: int) -> tuple[Stacks, bool]:
        key = (stacks, byte)
        hit = self._step_memo.get(key)
        if hit is not None:
            return hit
        out: set[Stack] = set()
        popped = False
        for s in stacks:
            cls = self.node_class[s[-1]]
            if byte in cls:  # type: ignore[operator]
                ns, p = self.close(s[:-1] + (self.node_next[s[-1]],))
                out |= ns
                popped |= p
        res = (frozenset(out), popped)
        if len(self._step_memo) >= self._memo_limit:
            self._step_memo.clear()  # crude bound on memory; memo is only a cache
        self._step_memo[key] = res
        return res

    def initial(self) -> tuple[Stacks, bool]:
        out: set[Stack] = set()
        popped = False
        for start in self.rule_starts[self.root]:
            ns, p = self.close((start,))
            out |= ns
            popped |= p
        return frozenset(out), popped

    # --------------------------------------------------------------- helpers
    def matcher(self) -> "Matcher":
        return Matcher(self)

    def accepts(self, data: bytes) -> bool:
        """Whole-string recognition (used by tests)."""
        m = self.matcher()
        return m.advance(data) and m.can_end


class Matcher:
    """Mutable decoding state: the current set of stacks + whether we may stop."""

    def __init__(self, pda: PDA):
        self.pda = pda
        self.stacks, self.can_end = pda.initial()
        self.history: list[tuple[Stacks, bool]] = []

    def advance(self, data: Iterable[int]) -> bool:
        """Consume bytes; returns False (and leaves state unchanged) if rejected."""
        stacks, can_end = self.stacks, self.can_end
        for b in data:
            stacks, can_end = self.pda.step(stacks, b)
            if not stacks and not can_end:
                return False
        self.history.append((self.stacks, self.can_end))
        self.stacks, self.can_end = stacks, can_end
        return True

    def rollback(self, n: int = 1) -> None:
        """Undo the last `n` successful advances (useful for speculative decoding)."""
        for _ in range(n):
            self.stacks, self.can_end = self.history.pop()

    @property
    def finished(self) -> bool:
        """True if the only thing left to do is to stop (no live stack)."""
        return not self.stacks
