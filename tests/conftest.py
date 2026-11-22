import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gcd.mask import TokenVocab  # noqa: E402


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow: set RUN_SLOW=1 (run through the GPU lock)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def _toy_tokens() -> list[bytes]:
    """A small BPE-like vocabulary with the nasty cases a real tokenizer has:
    multi-byte tokens that cross JSON structure boundaries ('",' '":"' '"}'),
    tokens that split a UTF-8 character, and prefixes of keywords ('tr', 'ue')."""
    toks = {bytes([b]) for b in range(0x20, 0x7F)}
    toks |= {b"\n", b"\xc3", b"\xa9", "é".encode(), "日".encode(), "日".encode()[:2], "日".encode()[2:]}
    toks |= {"😀".encode()[:3], "😀".encode()[3:], b"\xc0", b"\xed\xa0"}  # incl. invalid bytes
    toks |= {b'{"', b'":', b'":"', b'",', b'","', b'"}', b'"]', b'}]', b'},', b'},{"', b'["', b'[{"', b': ', b', "'}
    toks |= {b"true", b"tr", b"ue", b"false", b"fal", b"se", b"null", b"nu", b"ll", b"12", b"0.", b"-1", b"e+", b"99"}
    toks |= {b"name", b"age", b"ame", b"email", b"city", b"sunny", b"rain", b"snow", b"cloudy", b" the", b"ing", b"\\n", b"\\u00"}
    toks |= {b'"name":"', b'"age":', b'",\"', b'tags":["', b'"sku', b'quantity":', b"_", b"__"}
    return sorted(toks)


@pytest.fixture(scope="session")
def toy_vocab() -> TokenVocab:
    toks: list[bytes | None] = list(_toy_tokens())
    toks.append(None)  # a special token that must never be allowed
    eos = len(toks)
    toks.append(None)  # EOS
    return TokenVocab(toks, [eos], size=len(toks) + 3)  # padded logits like real models
