import random

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from gcd import PDA, MaskEngine, TokenVocab, brute_force_mask, schema_to_grammar
from gcd.examples import Event, Order, Person, Product, Recipe, Weather
from gcd.mask import bytes_to_unicode

MODELS = [Person, Product, Weather, Order, Event, Recipe]


def random_walk(engine: MaskEngine, rng: random.Random, max_steps: int = 400, check=None):
    """Sample uniformly among allowed tokens (biased toward closing tokens so that walks
    terminate). `check(matcher, mask)` is called at every step."""
    voc = engine.vocab
    m = engine.pda.matcher()
    out = bytearray()
    for _ in range(max_steps):
        mask = engine.mask(m)
        if check:
            check(m, mask)
        allowed = np.flatnonzero(mask).tolist()
        assert allowed, "dead end: the grammar must always allow some continuation"
        if m.can_end and rng.random() < 0.5:
            return bytes(out), True
        closing = [t for t in allowed if voc.tokens[t] and any(c in voc.tokens[t] for c in b'"]},')]
        pick = rng.choice(closing if closing and rng.random() < 0.6 else allowed)
        if pick in voc.eos_ids:
            return bytes(out), True
        assert m.advance(voc.tokens[pick])
        out += voc.tokens[pick]
    return bytes(out), False


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_cached_mask_equals_brute_force(model, toy_vocab):
    pda = PDA(schema_to_grammar(model))
    cached, nocache = MaskEngine(pda, toy_vocab), MaskEngine(pda, toy_vocab, use_cache=False)

    def check(m, mask):
        ref = brute_force_mask(pda, toy_vocab, m)
        np.testing.assert_array_equal(mask, ref)
        np.testing.assert_array_equal(nocache.mask(m), ref)
        assert not mask[len(toy_vocab.tokens):].any()  # padded logits never allowed
        assert not mask[toy_vocab.eos_ids[0] - 1]  # special token never allowed

    rng = random.Random(0)
    for _ in range(3):
        random_walk(cached, rng, max_steps=120, check=check)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: m.__name__)
def test_random_constrained_walks_roundtrip_through_pydantic(model: type[BaseModel], toy_vocab):
    """The guarantee: anything the mask lets through, once finished, validates."""
    engine = MaskEngine(PDA(schema_to_grammar(model)), toy_vocab)
    rng = random.Random(1)
    finished = 0
    for _ in range(25):
        text, done = random_walk(engine, rng)
        if done:
            finished += 1
            try:
                model.model_validate_json(text)
            except ValidationError as e:  # pragma: no cover - would be a real bug
                pytest.fail(f"{text!r}: {e}")
    assert finished >= 5


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_context_frames_trade_cache_keys_for_fewer_runtime_checks(k, toy_vocab):
    """Any amount of context expansion is exact; more context = fewer runtime checks."""
    pda = PDA(schema_to_grammar(Order))
    engine = MaskEngine(pda, toy_vocab, context_frames=k)

    def check(m, mask):
        np.testing.assert_array_equal(mask, brute_force_mask(pda, toy_vocab, m))

    random_walk(engine, random.Random(2), max_steps=150, check=check)
    assert engine.stats.nodes_cached > 0


def test_context_expansion_makes_string_tokens_context_independent(toy_vocab):
    ids = {t: i for i, t in enumerate(toy_vocab.tokens) if t}
    cd_frac = {}
    for k in (1, 3):
        engine = MaskEngine(PDA(schema_to_grammar(Person)), toy_vocab, context_frames=k)
        m = engine.pda.matcher()
        assert m.advance(b'{"name":"')
        engine.mask(m)
        entries = [engine.cache[s[-k:]] for s in m.stacks]
        cd = {int(t) for e in entries for t in e.cd_order}
        assert (ids[b"name"] in cd) == (k == 1)  # 'n' ends `char`: escapes with k=1 only
        # Closing the string and continuing needs the caller. With k=3 the stack that waits
        # for the closing quote is only 3 frames deep, so the key *is* the whole stack.
        assert (ids[b'",'] in cd) == (k == 1)
        cd_frac[k] = len(cd)
    assert cd_frac[3] < cd_frac[1]


def test_partial_utf8_tokens_are_allowed_inside_strings(toy_vocab):
    engine = MaskEngine(PDA(schema_to_grammar({"type": "string"})), toy_vocab)
    m = engine.pda.matcher()
    assert m.advance(b'"')
    mask = engine.mask(m)
    ids = {t: i for i, t in enumerate(toy_vocab.tokens) if t}
    assert mask[ids[b"\xc3"]] and mask[ids["日".encode()[:2]]]
    assert not mask[ids[b"\xa9"]]  # a continuation byte cannot start a character
    assert not mask[ids[b"\xc0"]] and not mask[ids[b"\xed\xa0"]]  # never valid UTF-8
    assert m.advance(b"\xc3")
    mask = engine.mask(m)
    assert mask[ids[b"\xa9"]] and not mask[ids[b'"']]


def test_rollback_restores_state(toy_vocab):
    pda = PDA(schema_to_grammar(Person))
    m = pda.matcher()
    before = (m.stacks, m.can_end)
    assert m.advance(b'{"name":"')
    assert not m.advance(b"")  # control byte rejected: state unchanged
    m.rollback()
    assert (m.stacks, m.can_end) == before


def test_bytes_to_unicode_is_a_bijection():
    table = bytes_to_unicode()
    assert len(table) == 256 and len(set(table.values())) == 256


def test_vocab_from_hf_like_tokenizer():
    class FakeTok:
        def __len__(self):
            return 4

        def get_added_vocab(self):
            return {"<eos>": 3}

        def convert_ids_to_tokens(self, ids):
            return ["Ġhi", "Ã©", "{\"", "<eos>"][: len(ids)]

    v = TokenVocab.from_hf(FakeTok(), eos_ids=[3], size=8)
    assert v.tokens == [b" hi", "é".encode(), b'{"', None] and v.size == 8


@pytest.mark.slow
def test_real_qwen_tokenizer_vocab():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    v = TokenVocab.from_hf(tok, eos_ids=[151645, 151643], size=151936)
    for text in ['{"name": "日本"}', "😀 ok"]:
        ids = tok(text, add_special_tokens=False)["input_ids"]
        assert b"".join(v.tokens[i] for i in ids) == text.encode()
