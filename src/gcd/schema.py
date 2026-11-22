"""JSON Schema (or a Pydantic model) -> byte-level context-free grammar.

Supported subset (everything else raises `UnsupportedSchema`, never silently ignored):
  type: object (properties, required; additionalProperties is treated as false),
        string, integer, number, boolean, null, array (items, minItems, maxItems),
        a list of types (["string", "null"])
  enum, const, anyOf / oneOf (oneOf is treated like anyOf), $ref into #/$defs
Output is *compact* JSON: object properties in declaration order, at most one space
after ':' and ','. Restricting whitespace matters: with `ws ::= [ \\n\\t]*` a small model
can (and does) loop on whitespace forever, a well-known failure of naive grammars.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .grammar import ByteClass, Grammar, lit


class UnsupportedSchema(ValueError):
    pass


_IGNORED_KEYS = {"title", "description", "default", "examples", "$schema", "additionalProperties"}
_DIGIT = ByteClass.rng(0x30, 0x39)
_CONT = ByteClass.rng(0x80, 0xBF)  # UTF-8 continuation byte


def _base_rules(g: Grammar) -> None:
    """Shared terminal rules, emitted once per grammar."""
    g.add("ws", (), lit(" "))
    # Unescaped string byte: printable ASCII except '"' and '\\'.
    ascii_ok = ByteClass.of(b for b in range(0x20, 0x80) if b not in (0x22, 0x5C))
    hexd = ByteClass.of(b"0123456789abcdefABCDEF")
    # Precise UTF-8 (RFC 3629, table 3-7 of the Unicode standard): no overlongs, no
    # surrogates. A token that splits a character is fine: the PDA simply stops between bytes.
    g.add(
        "char",
        (ascii_ok,),
        (ByteClass.rng(0xC2, 0xDF), _CONT),
        (*lit(b"\xE0"), ByteClass.rng(0xA0, 0xBF), _CONT),
        (ByteClass.of([*range(0xE1, 0xED), 0xEE, 0xEF]), _CONT, _CONT),
        (*lit(b"\xED"), ByteClass.rng(0x80, 0x9F), _CONT),
        (*lit(b"\xF0"), ByteClass.rng(0x90, 0xBF), _CONT, _CONT),
        (ByteClass.rng(0xF1, 0xF3), _CONT, _CONT, _CONT),
        (*lit(b"\xF4"), ByteClass.rng(0x80, 0x8F), _CONT, _CONT),
        (*lit("\\"), ByteClass.of(b'"\\/bfnrt')),
        (*lit("\\u"), hexd, hexd, hexd, hexd),
    )
    g.star("chars", "char")
    g.add("string", (*lit('"'), "chars", *lit('"')))
    g.star("digits", _DIGIT)
    g.add("uint", lit("0"), (ByteClass.rng(0x31, 0x39), "digits"))
    g.add("integer", ("uint",), (*lit("-"), "uint"))
    g.add("frac", (), (*lit("."), _DIGIT, "digits"))
    g.add("exp", (), (ByteClass.of(b"eE"), "sign", _DIGIT, "digits"))
    g.add("sign", (), lit("+"), lit("-"))
    g.add("number", ("integer", "frac", "exp"))
    g.add("boolean", lit("true"), lit("false"))
    g.add("null", lit("null"))


class SchemaCompiler:
    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self.g = Grammar(root="root")
        self._memo: dict[str, str] = {}  # canonical sub-schema JSON -> rule name
        self._n = 0

    def compile(self) -> Grammar:
        _base_rules(self.g)
        top = self.visit(self.schema, "value")
        self.g.add("root", (top,))
        self.g.validate()
        return self.g

    def _fresh(self, hint: str) -> str:
        self._n += 1
        return f"{re.sub(r'[^A-Za-z0-9_]', '_', hint)}_{self._n}"

    def visit(self, s: dict[str, Any], hint: str) -> str:
        """Return the name of a rule matching sub-schema `s` (memoized, so a shared
        definition becomes one shared rule and recursion through $ref terminates)."""
        key = json.dumps(s, sort_keys=True)
        if key in self._memo:
            return self._memo[key]
        unknown = set(s) - _IGNORED_KEYS - {
            "type", "properties", "required", "items", "minItems", "maxItems",
            "enum", "const", "anyOf", "oneOf", "$ref", "$defs", "definitions",
        }
        if unknown:
            raise UnsupportedSchema(f"unsupported keywords {sorted(unknown)}")
        if "$ref" in s:
            ref = s["$ref"]
            m = re.fullmatch(r"#/(\$defs|definitions)/(.+)", ref)
            if not m or m.group(2) not in self.schema.get(m.group(1), {}):
                raise UnsupportedSchema(f"unresolvable $ref {ref!r}")
            name = self._fresh(m.group(2))
            self._memo[key] = name  # register *before* recursing: recursive schemas
            inner = self.visit(self.schema[m.group(1)][m.group(2)], m.group(2))
            self.g.add(name, (inner,))
            return name
        if "const" in s:
            return self._literals([s["const"]], hint, key)
        if "enum" in s:
            return self._literals(s["enum"], hint, key)
        if "anyOf" in s or "oneOf" in s:
            subs = [self.visit(x, hint) for x in s.get("anyOf", s.get("oneOf"))]
            name = self._memo[key] = self.g.add(self._fresh(hint), *[(x,) for x in subs])
            return name
        t = s.get("type")
        if isinstance(t, list):
            subs = [self.visit({**s, "type": x}, hint) for x in t]
            name = self._memo[key] = self.g.add(self._fresh(hint), *[(x,) for x in subs])
            return name
        if t in ("string", "integer", "number", "boolean", "null"):
            self._memo[key] = t
            return t
        if t == "array":
            name = self._memo[key] = self._array(s, hint)
            return name
        if t == "object" or "properties" in s:
            name = self._memo[key] = self._object(s, hint)
            return name
        raise UnsupportedSchema(f"cannot compile sub-schema {s!r}")

    def _literals(self, values: list[Any], hint: str, key: str) -> str:
        alts = [lit(json.dumps(v, separators=(",", ":"), ensure_ascii=False)) for v in values]
        name = self._memo[key] = self.g.add(self._fresh(hint + "_enum"), *alts)
        return name

    def _array(self, s: dict[str, Any], hint: str) -> str:
        item = self.visit(s.get("items", {"type": "string"}), hint + "_item")
        lo, hi = int(s.get("minItems", 0)), s.get("maxItems")
        sep = (*lit(","), "ws")
        name = self._fresh(hint + "_array")
        if hi is not None and int(hi) < lo:
            raise UnsupportedSchema("maxItems < minItems")
        # tail(k) = rule for "zero or more further ', item'" with k = how many more are allowed.
        if hi is None:
            tail = self.g.add(name + "_more", (*sep, item, name + "_more"), ())
        else:
            tail = self.g.add(self._fresh(hint + "_end"), ())
            for _ in range(int(hi) - max(lo, 1)):
                tail = self.g.add(self._fresh(hint + "_opt"), (*sep, item, tail), ())
        # Unroll the required items: item (, item){lo-1} tail
        body: tuple = (item,) + tuple(e for _ in range(max(lo, 1) - 1) for e in (*sep, item)) + (tail,)
        alts = [(*lit("["), *body, *lit("]"))]
        if lo == 0 and (hi is None or int(hi) > 0):
            alts.append(lit("[]"))
        if hi is not None and int(hi) == 0:
            alts = [lit("[]")]
        return self.g.add(name, *alts)

    def _object(self, s: dict[str, Any], hint: str) -> str:
        props: dict[str, Any] = s.get("properties", {})
        required = set(s.get("required", []))
        if not required <= set(props):
            raise UnsupportedSchema("required property missing from properties")
        names = list(props)
        n = len(names)
        base = self._fresh(hint + "_obj")
        # pair_i = '"key":' ws value_i
        pairs = [
            (*lit(json.dumps(k, ensure_ascii=False) + ":"), "ws", self.visit(props[k], k)) for k in names
        ]
        # first_i: properties i.. where none has been emitted yet (no leading comma)
        # rest_i : properties i.. after at least one was emitted (each needs ', ')
        # Optional properties get an extra "skip" alternative. This keeps the grammar
        # linear in the number of properties (no 2^n blow-up for optional subsets).
        first = [f"{base}_f{i}" for i in range(n + 1)]
        rest = [f"{base}_r{i}" for i in range(n + 1)]
        self.g.add(first[n], ())
        self.g.add(rest[n], ())
        for i in reversed(range(n)):
            optional = names[i] not in required
            self.g.add(first[i], (*pairs[i], rest[i + 1]), *([(first[i + 1],)] if optional else []))
            self.g.add(rest[i], (*lit(","), "ws", *pairs[i], rest[i + 1]), *([(rest[i + 1],)] if optional else []))
        return self.g.add(base, (*lit("{"), first[0], *lit("}")))


def schema_to_grammar(schema: dict[str, Any] | type) -> Grammar:
    """Accept a JSON Schema dict or a Pydantic model class."""
    if isinstance(schema, type) and hasattr(schema, "model_json_schema"):
        schema = schema.model_json_schema()
    return SchemaCompiler(schema).compile()
