"""Judge-response parsing. Pure: no model call, just the parser.

The parser must be robust to the ways a model wraps JSON, and must never raise,
because one malformed judge reply should not crash a whole evaluation run.
"""

from __future__ import annotations

from app.eval.faithfulness import parse_judge_response


def test_parse_plain_supported():
    r = parse_judge_response('{"verdict": "supported", "unsupported_claims": [], "reasoning": "ok"}')
    assert r.verdict == "supported"
    assert r.score == 1.0
    assert r.is_faithful is True
    assert r.unsupported_claims == []


def test_parse_partial_lists_claims():
    text = '{"verdict":"partial","unsupported_claims":["the office has 50 desks"],"reasoning":"one claim unsupported"}'
    r = parse_judge_response(text)
    assert r.verdict == "partial"
    assert r.score == 0.5
    assert r.is_faithful is False
    assert r.unsupported_claims == ["the office has 50 desks"]


def test_parse_unsupported():
    r = parse_judge_response('{"verdict":"unsupported","unsupported_claims":["all of it"]}')
    assert r.verdict == "unsupported"
    assert r.score == 0.0


def test_parse_json_in_code_fence():
    text = "Here is my judgement:\n```json\n{\"verdict\": \"supported\", \"unsupported_claims\": []}\n```"
    r = parse_judge_response(text)
    assert r.verdict == "supported"


def test_parse_json_with_surrounding_prose():
    text = 'The answer looks fine. {"verdict": "supported", "unsupported_claims": []} Done.'
    r = parse_judge_response(text)
    assert r.verdict == "supported"


def test_unparseable_becomes_unknown():
    r = parse_judge_response("I think the answer is mostly fine, hard to say.")
    assert r.verdict == "unknown"
    assert r.score == 0.0
    assert r.is_faithful is False


def test_invalid_verdict_value_becomes_unknown():
    r = parse_judge_response('{"verdict": "looks-good", "unsupported_claims": []}')
    assert r.verdict == "unknown"
    assert r.score == 0.0


def test_missing_keys_do_not_raise():
    r = parse_judge_response('{"verdict": "supported"}')
    assert r.verdict == "supported"
    assert r.unsupported_claims == []
