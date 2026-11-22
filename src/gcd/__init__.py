"""Grammar-constrained decoding from scratch: JSON Schema -> CFG -> byte PDA -> token masks."""
from .grammar import ByteClass, Grammar, lit
from .mask import MaskEngine, TokenVocab, brute_force_mask
from .pda import PDA, Matcher
from .schema import UnsupportedSchema, schema_to_grammar

__all__ = [
    "ByteClass", "Grammar", "lit", "MaskEngine", "TokenVocab", "brute_force_mask",
    "PDA", "Matcher", "UnsupportedSchema", "schema_to_grammar",
]
