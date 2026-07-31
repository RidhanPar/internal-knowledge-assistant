"""Agent graph behaviour, driven by a scripted fake LLM.

No Bedrock and no database: we patch `converse_with_tools` with a scripted
sequence of responses and use a fake retriever. This lets us assert the parts
that make the agent trustworthy — tool routing, multi-tool source accumulation,
citation mapping, and the grounding guard that forces "I don't know".
"""

from __future__ import annotations

from app.agent import graph as graph_module
from app.agent.directory import DirectoryService
from app.agent.graph import KnowledgeAgent
from app.agent.service import AgentService
from app.config import Settings
from app.db.repository import RetrievedChunk
from app.rag import prompts
from app.rag.llm import ConverseResult, ToolUse
from app.rag.retriever import RetrievalOutcome


# --- fakes -------------------------------------------------------------------
class FakeRetriever:
    def __init__(self, evidence: list[RetrievedChunk]) -> None:
        self._evidence = evidence

    async def retrieve(self, question: str, top_k=None) -> RetrievalOutcome:
        return RetrievalOutcome(retrieved=self._evidence, evidence=self._evidence)


def _tool_use(*calls: tuple[str, dict]) -> ConverseResult:
    tus = [ToolUse(id=f"t{i}", name=n, input=inp) for i, (n, inp) in enumerate(calls)]
    content = [{"toolUse": {"toolUseId": t.id, "name": t.name, "input": t.input}} for t in tus]
    return ConverseResult(
        text="", tool_uses=tus, stop_reason="tool_use",
        raw_message={"role": "assistant", "content": content},
        input_tokens=10, output_tokens=5,
    )


def _answer(text: str) -> ConverseResult:
    return ConverseResult(
        text=text, tool_uses=[], stop_reason="end_turn",
        raw_message={"role": "assistant", "content": [{"text": text}]},
        input_tokens=10, output_tokens=8,
    )


def _script(monkeypatch, responses: list[ConverseResult]) -> None:
    seq = iter(responses)

    async def _fake(system, messages, tool_specs):
        return next(seq)

    monkeypatch.setattr(graph_module, "converse_with_tools", _fake)


def _chunk(doc_id, title, heading, content, sim) -> RetrievedChunk:
    return RetrievedChunk(doc_id, title, f"corpus/{doc_id}.md", heading, content, sim)


def _service(evidence) -> AgentService:
    directory = DirectoryService.from_json("data/directory.json")
    agent = KnowledgeAgent(FakeRetriever(evidence), directory)
    return AgentService(agent, Settings())


# --- tests -------------------------------------------------------------------
async def test_single_tool_answer_with_citation(monkeypatch):
    evidence = [
        _chunk("security-policy", "Security Policy", "Auth > MFA", "MFA is mandatory.", 0.7),
        _chunk("employee-handbook", "Handbook", "Annual Leave", "28 days of leave.", 0.5),
    ]
    _script(monkeypatch, [
        _tool_use(("search_documents", {"query": "is MFA required"})),
        _answer("Yes, MFA is mandatory for all accounts [1]."),
    ])
    resp = await _service(evidence).answer("Is MFA required?")

    assert resp.no_answer is False
    assert [c.marker for c in resp.citations] == [1]
    assert resp.citations[0].source_type == "document"
    assert resp.retrieved_count == 2
    assert any("search_documents" in s for s in resp.steps)


async def test_no_relevant_docs_forces_no_answer(monkeypatch):
    _script(monkeypatch, [
        _tool_use(("search_documents", {"query": "capital of France"})),
        _answer(prompts.NO_ANSWER_SENTINEL),
    ])
    resp = await _service(evidence=[]).answer("What is the capital of France?")

    assert resp.no_answer is True
    assert resp.answer == prompts.NO_ANSWER_SENTINEL
    assert resp.citations == []


async def test_multi_tool_accumulates_and_maps_markers(monkeypatch):
    evidence = [
        _chunk("incident-response-runbook", "Runbook", "Security Incidents",
               "A breach is at least SEV2.", 0.66),
        _chunk("security-policy", "Security Policy", "Reporting",
               "Report to #security-incidents.", 0.6),
    ]
    _script(monkeypatch, [
        _tool_use(("search_documents", {"query": "security incident severity"})),
        _tool_use(("lookup_directory", {"topic": "security"})),
        _answer("A breach is at least SEV2 [1]; contact the Security team [3]."),
    ])
    resp = await _service(evidence).answer(
        "How severe is a breach and who do I contact?"
    )

    # 2 document sources (markers 1,2) + 1 directory source (marker 3).
    assert resp.retrieved_count == 3
    assert [c.marker for c in resp.citations] == [1, 3]
    types = {c.marker: c.source_type for c in resp.citations}
    assert types[1] == "document" and types[3] == "directory"
    assert sum("search_documents" in s for s in resp.steps) == 1
    assert sum("lookup_directory" in s for s in resp.steps) == 1


async def test_answer_without_any_tool_is_guarded(monkeypatch):
    # Model tries to answer directly with no tool call -> no sources gathered ->
    # the finalize guard must refuse rather than trust an ungrounded answer.
    _script(monkeypatch, [_answer("The capital of France is Paris.")])
    resp = await _service(evidence=[]).answer("Capital of France?")

    assert resp.no_answer is True
    assert resp.answer == prompts.NO_ANSWER_SENTINEL
    assert resp.citations == []


async def test_uncited_answer_exposes_all_sources(monkeypatch):
    evidence = [_chunk("handbook", "Handbook", "Leave", "28 days.", 0.7)]
    _script(monkeypatch, [
        _tool_use(("search_documents", {"query": "leave"})),
        _answer("Employees get 28 days of leave."),  # no [n] marker
    ])
    resp = await _service(evidence).answer("How much leave?")
    assert resp.no_answer is False
    assert [c.marker for c in resp.citations] == [1]


async def test_step_budget_bounds_the_loop(monkeypatch):
    # Model keeps asking for tools forever; the budget must stop it and the
    # guard must refuse (no final answer was produced).
    evidence = [_chunk("handbook", "Handbook", "Leave", "28 days.", 0.7)]
    _script(monkeypatch, [_tool_use(("search_documents", {"query": "x"}))] * 20)
    settings = Settings(max_agent_steps=3)
    directory = DirectoryService.from_json("data/directory.json")
    agent = KnowledgeAgent(FakeRetriever(evidence), directory)
    resp = await AgentService(agent, settings).answer("loop?")

    assert resp.no_answer is True
    assert resp.answer == prompts.NO_ANSWER_SENTINEL
