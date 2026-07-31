"""Directory lookup tests — pure, no I/O beyond reading the shipped JSON."""

from __future__ import annotations

from app.agent.directory import DirectoryRecord, DirectoryService


def _svc() -> DirectoryService:
    return DirectoryService.from_json("data/directory.json")


def test_alias_match_returns_expected_team():
    svc = _svc()
    hits = svc.lookup("phishing")
    assert hits and hits[0].team == "Security"


def test_topic_match_for_on_call():
    svc = _svc()
    hits = svc.lookup("on-call")
    assert any(r.team == "Engineering On-Call" for r in hits)


def test_no_match_returns_empty():
    svc = _svc()
    assert svc.lookup("quantum teleportation budget") == []


def test_empty_query_returns_empty():
    assert _svc().lookup("   ") == []


def test_exact_team_alias_beats_substring():
    # "it" is an exact alias of the IT Service Desk; ensure it ranks it first
    # rather than being swamped by substring hits elsewhere.
    svc = _svc()
    hits = svc.lookup("it")
    assert hits[0].team == "IT Service Desk"


def test_record_render_contains_key_fields():
    rec = DirectoryRecord(
        id="x", team="X", aliases=["x"], owner="O", contact_channel="#x",
        email="x@e", escalation="E", on_call="24/7", location="L", notes="N",
    )
    rendered = rec.render()
    assert "Contact channel: #x" in rendered
    assert "Escalation: E" in rendered
