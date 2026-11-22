import json

import pytest
from pydantic import BaseModel

from gcd import PDA, Grammar, UnsupportedSchema, lit, schema_to_grammar
from gcd.examples import Event, Order, Person, Product, Recipe, Weather


def compact(m: BaseModel) -> bytes:
    return json.dumps(m.model_dump(mode="json"), separators=(",", ":"), ensure_ascii=False).encode()


VALID = [
    Person(name="Ana", age=31, email="ana@x.io", is_student=False),
    Person(name='Q"uote\\ 日本 é 😀', age=-7, email="", is_student=True),
    Product(id=0, name="kb", price=19.99, tags=[], in_stock=True),
    Product(id=12, name="kb", price=-1.5e-3, tags=["a", "b", "c", "d"], in_stock=False),
    Weather(city="Oslo", temperature_c=-3.0, condition="snow", humidity_pct=80),
    Order(order_id="A1", customer={"name": "B", "address": {"street": "s", "city": "c", "zip_code": "1"}},
          items=[{"sku": "x", "quantity": 1}, {"sku": "y", "quantity": 2}], express=True),
    Event(title="t", date="2026-11-22", attendees=["a"], location="Madrid"),
    Recipe(name="r", servings=2, ingredients=[{"name": "egg", "amount": 2, "unit": "piece"}], vegetarian=True),
]


@pytest.mark.parametrize("inst", VALID, ids=lambda m: type(m).__name__)
def test_accepts_valid_instances(inst):
    pda = PDA(schema_to_grammar(type(inst)))
    assert pda.accepts(compact(inst))
    # one space after ':' and ',' is also allowed
    spaced = json.dumps(inst.model_dump(mode="json"), separators=(", ", ": "), ensure_ascii=False).encode()
    assert pda.accepts(spaced)


@pytest.mark.parametrize(
    "model,text",
    [
        (Person, '{"name":"Ana","age":31,"email":"a"}'),  # missing required
        (Person, '{"name":"Ana","age":31.5,"email":"a","is_student":true}'),  # float for int
        (Person, '{"name":"Ana","age":031,"email":"a","is_student":true}'),  # leading zero
        (Person, '{"name":Ana,"age":31,"email":"a","is_student":true}'),  # unquoted
        (Person, '{"name":"Ana","age":31,"email":"a","is_student":true,}'),  # trailing comma
        (Person, '{"name":"Ana","age":31,"email":"a","is_student":true,"x":1}'),  # extra prop
        (Person, '{"name":"A\\q","age":31,"email":"a","is_student":true}'),  # bad escape
        (Person, '{"name":"Ana","age":31,"email":"a","is_student":True}'),  # Python bool
        (Weather, '{"city":"O","temperature_c":1,"condition":"fog","humidity_pct":1}'),  # enum
        (Product, '{"id":1,"name":"k","price":1,"tags":["a","b","c","d","e"],"in_stock":true}'),  # maxItems
        (Order, '{"order_id":"A","customer":{"name":"B","address":{"street":"s","city":"c","zip_code":"1"}},"items":[],"express":true}'),  # minItems
        (Person, '{"name":"Ana","age":31,"email":"a","is_student":true}  '),  # trailing ws
        (Person, '{"name":"Ana",  "age":31,"email":"a","is_student":true}'),  # 2 spaces
    ],
)
def test_rejects_invalid(model, text):
    assert not PDA(schema_to_grammar(model)).accepts(text.encode())


def test_optional_property_may_be_omitted_or_null():
    pda = PDA(schema_to_grammar(Event))
    assert pda.accepts(b'{"title":"t","date":"d","attendees":[]}')
    assert pda.accepts(b'{"title":"t","date":"d","attendees":[],"location":null}')
    assert not pda.accepts(b'{"title":"t","date":"d","location":null}')


def test_all_optional_object_handles_commas():
    schema = {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}, "c": {"type": "integer"}}}
    pda = PDA(schema_to_grammar(schema))
    for ok in ["{}", '{"b":1}', '{"a":1,"c":2}', '{"a":1,"b":2,"c":3}', '{"c":3}']:
        assert pda.accepts(ok.encode()), ok
    for bad in ['{,"b":1}', '{"a":1,}', '{"b":1,"a":2}', '{"a":1"b":2}']:
        assert not pda.accepts(bad.encode()), bad


def test_utf8_is_validated_at_byte_level():
    pda = PDA(schema_to_grammar({"type": "string"}))
    assert pda.accepts('"日本 é 😀"'.encode())
    assert not pda.accepts(b'"\xc0\xaf"')  # overlong encoding of '/'
    assert not pda.accepts(b'"\xed\xa0\x80"')  # UTF-16 surrogate
    assert not pda.accepts(b'"\xe6\x97"')  # truncated 3-byte char
    m = pda.matcher()
    assert m.advance(b'"\xe6') and m.advance(b"\x97") and m.advance(b'\xa5"')  # split across tokens
    assert m.can_end


def test_recursive_schema_via_ref():
    schema = {
        "$defs": {"Node": {"type": "object", "properties": {"v": {"type": "integer"}, "kids": {"type": "array", "items": {"$ref": "#/$defs/Node"}}}, "required": ["v", "kids"]}},
        "$ref": "#/$defs/Node",
    }
    pda = PDA(schema_to_grammar(schema))
    assert pda.accepts(b'{"v":1,"kids":[{"v":2,"kids":[]},{"v":3,"kids":[{"v":4,"kids":[]}]}]}')
    assert not pda.accepts(b'{"v":1,"kids":[{"v":2}]}')


def test_const_enum_anyof_and_type_lists():
    s = {"type": "object", "properties": {
        "k": {"const": "fixed"}, "n": {"enum": [1, "two", None]},
        "u": {"anyOf": [{"type": "integer"}, {"type": "boolean"}]}, "t": {"type": ["string", "null"]}},
        "required": ["k", "n", "u", "t"]}
    pda = PDA(schema_to_grammar(s))
    assert pda.accepts(b'{"k":"fixed","n":"two","u":true,"t":null}')
    assert pda.accepts(b'{"k":"fixed","n":null,"u":-4,"t":"x"}')
    assert not pda.accepts(b'{"k":"other","n":1,"u":1,"t":"x"}')
    assert not pda.accepts(b'{"k":"fixed","n":2,"u":1,"t":"x"}')


def test_unsupported_keywords_raise():
    with pytest.raises(UnsupportedSchema):
        schema_to_grammar({"type": "string", "pattern": "^a+$"})
    with pytest.raises(UnsupportedSchema):
        schema_to_grammar({"$ref": "#/$defs/Missing"})


def test_left_recursion_is_rejected():
    g = Grammar(root="e")
    g.add("e", ("e", *lit("+1")), lit("1"))
    with pytest.raises(ValueError, match="left recursion"):
        g.validate()


def test_ebnf_dump_is_readable():
    text = schema_to_grammar(Person).to_ebnf()
    assert 'string ::= "\\"" chars "\\""' in text
    assert '"\\"age\\":" ws integer' in text


def test_tail_calls_keep_stack_depth_constant():
    pda = PDA(schema_to_grammar({"type": "array", "items": {"type": "integer"}}))
    m = pda.matcher()
    assert m.advance(b"[0")
    depths = []
    for _ in range(200):
        assert m.advance(b",1")
        depths.append(max(len(s) for s in m.stacks))
    assert len(set(depths)) == 1
