"""A tiny byte-level context-free grammar representation.

Every terminal is a *set of bytes* (a byte class). Literal strings are expanded into
one single-byte class per byte. Working at the byte level (as XGrammar does) is what
lets us handle byte-level BPE vocabularies exactly: a token that ends in the middle of
a multi-byte UTF-8 character is just a byte string that stops in the middle of a rule.

A rule is a list of alternatives; an alternative is a tuple of elements; an element is
either a `ByteClass` (terminal) or a rule name (non-terminal reference).

Restriction (checked by `Grammar.validate`): no left recursion. Repetition is written
with right recursion (`star(X) ::= X star(X) | ""`), which our pushdown automaton runs
in constant stack depth thanks to tail-call elimination (see `pda.py`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Union


@dataclass(frozen=True)
class ByteClass:
    """A terminal matching exactly one byte from `allowed`."""

    allowed: frozenset[int]

    @staticmethod
    def of(chars: Iterable[int]) -> "ByteClass":
        return ByteClass(frozenset(chars))

    @staticmethod
    def rng(lo: int, hi: int) -> "ByteClass":
        return ByteClass(frozenset(range(lo, hi + 1)))

    def __repr__(self) -> str:
        return _class_to_ebnf(self.allowed)


Element = Union[ByteClass, str]
Alternative = tuple[Element, ...]


def lit(s: str | bytes) -> Alternative:
    """Expand a literal into a sequence of single-byte terminals."""
    data = s.encode("utf-8") if isinstance(s, str) else s
    return tuple(ByteClass(frozenset((b,))) for b in data)


@dataclass
class Grammar:
    root: str
    rules: dict[str, list[Alternative]] = field(default_factory=dict)

    def add(self, name: str, *alternatives: Iterable[Element]) -> str:
        if name in self.rules:
            raise ValueError(f"rule {name!r} defined twice")
        self.rules[name] = [tuple(a) for a in alternatives]
        return name

    def star(self, name: str, element: Element) -> str:
        """`name ::= element name | ""` (right recursion: zero or more)."""
        return self.add(name, (element, name), ())

    def validate(self) -> None:
        """Check references and reject left recursion (it would loop the PDA)."""
        for name, alts in self.rules.items():
            for alt in alts:
                for e in alt:
                    if isinstance(e, str) and e not in self.rules:
                        raise ValueError(f"rule {name!r} references unknown rule {e!r}")
        nullable = self._nullable()
        # Edge A -> B if B can appear first in A (after a nullable prefix).
        first: dict[str, set[str]] = {n: set() for n in self.rules}
        for name, alts in self.rules.items():
            for alt in alts:
                for e in alt:
                    if isinstance(e, ByteClass):
                        break
                    first[name].add(e)
                    if e not in nullable:
                        break
        for start in self.rules:  # DFS for a cycle back to `start`
            stack, seen = list(first[start]), set()
            while stack:
                n = stack.pop()
                if n == start:
                    raise ValueError(f"left recursion through rule {start!r}")
                if n not in seen:
                    seen.add(n)
                    stack.extend(first[n])

    def _nullable(self) -> set[str]:
        nullable: set[str] = set()
        changed = True
        while changed:
            changed = False
            for name, alts in self.rules.items():
                if name not in nullable and any(
                    all(isinstance(e, str) and e in nullable for e in alt) for alt in alts
                ):
                    nullable.add(name)
                    changed = True
        return nullable

    def to_ebnf(self) -> str:
        """Human-readable EBNF-ish dump (runs of single bytes are merged into literals)."""
        lines = []
        for name, alts in self.rules.items():
            rendered = [_alt_to_ebnf(a) for a in alts]
            lines.append(f"{name} ::= " + " | ".join(rendered))
        return "\n".join(lines)


def _alt_to_ebnf(alt: Alternative) -> str:
    if not alt:
        return '""'
    parts: list[str] = []
    buf = bytearray()
    for e in alt:
        if isinstance(e, ByteClass) and len(e.allowed) == 1 and 0x20 <= next(iter(e.allowed)) < 0x7F:
            buf.append(next(iter(e.allowed)))
            continue
        if buf:
            parts.append('"' + buf.decode().replace("\\", "\\\\").replace('"', '\\"') + '"')
            buf.clear()
        parts.append(e if isinstance(e, str) else _class_to_ebnf(e.allowed))
    if buf:
        parts.append('"' + buf.decode().replace("\\", "\\\\").replace('"', '\\"') + '"')
    return " ".join(parts)


def _class_to_ebnf(allowed: frozenset[int]) -> str:
    xs = sorted(allowed)
    ranges, i = [], 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        ranges.append((xs[i], xs[j]))
        i = j + 1

    def fmt(b: int) -> str:
        c = chr(b)
        return c if 0x21 <= b < 0x7F and c not in "]\\-^" else f"\\x{b:02X}"

    return "[" + "".join(fmt(a) if a == b else f"{fmt(a)}-{fmt(b)}" for a, b in ranges) + "]"
