"""Print the compiled agent graph as Mermaid (for docs / sanity-checking).

    py -3.11 -m scripts.print_graph

Builds the graph with lightweight stand-ins for the retriever and directory —
no database or Bedrock needed, since we only inspect the graph's shape.
"""

from __future__ import annotations

from app.agent.directory import DirectoryService
from app.agent.graph import KnowledgeAgent


class _NullRetriever:
    async def retrieve(self, question: str, top_k=None):  # pragma: no cover - shape only
        raise NotImplementedError


def main() -> None:
    directory = DirectoryService(records=[])
    agent = KnowledgeAgent(_NullRetriever(), directory)  # type: ignore[arg-type]
    print(agent.graph.get_graph().draw_mermaid())


if __name__ == "__main__":
    main()
