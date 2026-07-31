"""The knowledge agent as a LangGraph state machine.

Why a graph (and why these nodes)
---------------------------------
A single LLM call can't *decide* to retrieve, retrieve twice, or combine two
different sources — it answers in one shot. Making the assistant agentic means
giving the model a loop where it chooses tools, sees results, and chooses again.
LangGraph models that loop as an explicit, inspectable state machine:

    START ─▶ agent ──(tool_use)──▶ tools ──▶ agent ──(end_turn)──▶ finalize ─▶ END
                 └────────────────(answer / budget hit)──────────────┘

- **agent** — the reasoner/router. It calls Claude with the tool specs. Claude
  either asks to call a tool (stopReason "tool_use") or writes a final answer.
  This is the node that *decides whether and what to retrieve*; putting it in a
  loop is what lets it handle multi-step questions.
- **tools** — the executor. It runs whichever tool(s) the agent requested
  (`search_documents`, `lookup_directory`), appends the results to the
  conversation as toolResult blocks, and accumulates the cited sources. Separating
  execution from reasoning keeps tools deterministic and testable, and keeps the
  side effects (DB / directory access) out of the model-facing node.
- **finalize** — the grounding guard. It decides `no_answer` and resolves which
  sources were actually cited. It is where "say you don't know rather than
  hallucinate" is *enforced structurally*: if no tool produced usable sources, or
  the model emitted the refusal sentinel, the answer is forced to no-answer — the
  model's output alone is never trusted to be grounded.

The conditional edge out of **agent** is the routing decision: loop to **tools**
while the model wants tools and the step budget remains, otherwise go to
**finalize**. The step budget (`max_iterations`) bounds the loop so a confused
agent can't retrieve forever.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent import tools as agent_tools
from app.agent.directory import DirectoryService
from app.agent.prompts import AGENT_SYSTEM_PROMPT
from app.agent.state import AgentState
from app.rag import prompts as rag_prompts
from app.rag.citations import cited_markers
from app.rag.llm import converse_with_tools
from app.rag.retriever import Retriever


class KnowledgeAgent:
    def __init__(self, retriever: Retriever, directory: DirectoryService) -> None:
        self._retriever = retriever
        self._directory = directory
        self._graph = self._build()

    # --- Nodes ------------------------------------------------------------
    async def agent_node(self, state: AgentState) -> dict:
        """Ask the model what to do next: call a tool, or answer."""
        result = await converse_with_tools(
            AGENT_SYSTEM_PROMPT, state["messages"], agent_tools.TOOL_SPECS
        )
        messages = state["messages"] + [result.raw_message]
        update: dict = {
            "messages": messages,
            "stop_reason": result.stop_reason,
            "input_tokens": state.get("input_tokens", 0) + result.input_tokens,
            "output_tokens": state.get("output_tokens", 0) + result.output_tokens,
        }
        if result.stop_reason == "tool_use" and result.tool_uses:
            update["pending_tool_uses"] = result.tool_uses
            update["final_text"] = None
        else:
            update["pending_tool_uses"] = []
            update["final_text"] = result.text
        return update

    async def tools_node(self, state: AgentState) -> dict:
        """Execute the requested tool(s) and feed results back to the model."""
        sources = list(state.get("sources", []))
        steps = list(state.get("steps", []))
        result_blocks: list[dict] = []

        for tu in state["pending_tool_uses"]:
            marker_start = len(sources) + 1
            if tu.name == "search_documents":
                outcome = await agent_tools.run_search_documents(
                    self._retriever, tu.input.get("query", ""), marker_start
                )
            elif tu.name == "lookup_directory":
                outcome = await agent_tools.run_lookup_directory(
                    self._directory, tu.input.get("topic", ""), marker_start
                )
            else:
                outcome = agent_tools.ToolOutcome(
                    result_text=f"Unknown tool: {tu.name}", sources=[], step=f"unknown_tool({tu.name})"
                )
            sources.extend(outcome.sources)
            steps.append(outcome.step)
            result_blocks.append(
                {
                    "toolResult": {
                        "toolUseId": tu.id,
                        "content": [{"text": outcome.result_text}],
                        "status": "success",
                    }
                }
            )

        messages = state["messages"] + [{"role": "user", "content": result_blocks}]
        return {
            "messages": messages,
            "sources": sources,
            "steps": steps,
            "iterations": state.get("iterations", 0) + 1,
            "pending_tool_uses": [],
        }

    def finalize_node(self, state: AgentState) -> dict:
        """Grounding guard: enforce no-answer, resolve cited markers."""
        sources = state.get("sources", [])
        final_text = state.get("final_text")

        # No sources gathered, or the loop was cut off mid-tool-call: nothing to
        # ground on, so we must not present an answer.
        if not sources or final_text is None:
            return {
                "final_text": rag_prompts.NO_ANSWER_SENTINEL,
                "no_answer": True,
                "used_markers": [],
            }

        if final_text.strip() == rag_prompts.NO_ANSWER_SENTINEL:
            return {"no_answer": True, "used_markers": []}

        markers = cited_markers(final_text, len(sources))
        if not markers:
            # Answered without explicit citations (against instructions): expose
            # all gathered sources so the answer is never uncitable.
            markers = [s.marker for s in sources]
        return {"no_answer": False, "used_markers": markers}

    # --- Routing ----------------------------------------------------------
    def route_after_agent(self, state: AgentState) -> str:
        wants_tools = bool(state.get("pending_tool_uses"))
        within_budget = state.get("iterations", 0) < state.get("max_iterations", 5)
        return "tools" if wants_tools and within_budget else "finalize"

    # --- Assembly ---------------------------------------------------------
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("agent", self.agent_node)
        g.add_node("tools", self.tools_node)
        g.add_node("finalize", self.finalize_node)
        g.add_edge(START, "agent")
        g.add_conditional_edges(
            "agent", self.route_after_agent, {"tools": "tools", "finalize": "finalize"}
        )
        g.add_edge("tools", "agent")
        g.add_edge("finalize", END)
        return g.compile()

    @property
    def graph(self):
        return self._graph
