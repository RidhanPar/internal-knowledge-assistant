"""Run the evaluation suite against the live system and print a scored report.

    py -3.11 -m scripts.evaluate           # run the whole dataset
    py -3.11 -m scripts.evaluate --limit 5 # first 5 cases
    py -3.11 -m scripts.evaluate --json     # machine-readable output

COST WARNING: this calls AWS Bedrock. Each policy case makes roughly one
embedding call, one agent generation (often with a tool round-trip, so two model
calls), and one judge generation. For the full 19-case dataset that is on the
order of 50 to 70 Bedrock model calls. On Claude 3.5 Sonnet and Titan v2 that is
a few US cents at current prices, not dollars, but it is not zero. Use --limit
while iterating. The database must be running and the corpus ingested first.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json

from app.agent.directory import DirectoryService
from app.agent.graph import KnowledgeAgent
from app.agent.service import AgentService
from app.config import get_settings
from app.core.logging import configure_logging
from app.db.pool import create_pool
from app.db.repository import Repository
from app.eval.dataset import load_dataset
from app.eval.runner import EvalReport, Evaluator
from app.rag.retriever import Retriever


def _print_report(report: EvalReport) -> None:
    b = report.behaviour
    print("\n" + "=" * 68)
    print("EVALUATION REPORT")
    print("=" * 68)

    print("\nPer case:")
    print(f"  {'id':22} {'category':13} {'behaviour':18} {'faith':7} flag")
    for r in report.cases:
        faith = f"{r.faithfulness.verdict}" if r.faithfulness else "-"
        flag = "FLAG" if r.flagged else ""
        print(f"  {r.id:22} {r.category:13} {r.behaviour_bucket:18} {faith:7} {flag}")

    print("\nRetrieval (scored over "
          f"{report.retrieval_scored_count} document cases):")
    print(f"  hit@k        {report.hit_at_k:.3f}   (right doc retrieved at all)")
    print(f"  recall@k     {report.recall_at_k:.3f}   (fraction of relevant docs found)")
    print(f"  precision@k  {report.precision_at_k:.3f}   (fraction of retrieved that were relevant)")
    print(f"  MRR          {report.mrr:.3f}   (rank of first relevant doc)")

    print("\nBehaviour (answer vs refuse):")
    print(f"  accuracy           {b.behaviour_accuracy:.3f}")
    print(f"  answered correctly {b.answered_correctly}/{b.should_answer}")
    print(f"  refused correctly  {b.refused_correctly}/{b.should_refuse}")
    print(f"  false answers      {b.false_answer}  (answered an unanswerable question)")
    print(f"  false refusals     {b.false_refusal}  (refused an answerable question)")
    print(f"  hallucination rate {b.false_answer_rate:.3f}  (false answers / should-refuse)")

    print("\nFaithfulness (judged over "
          f"{report.faithfulness_judged_count} answered cases):")
    print(f"  mean score     {report.faithfulness_mean:.3f}   (1.0 supported, 0.5 partial, 0.0 unsupported)")
    print(f"  flagged        {report.flagged_count}  (false answer or not fully grounded)")

    print(f"\nAnswer content match rate: {report.answer_match_rate:.3f}")
    print("=" * 68 + "\n")


async def _run(limit: int | None, as_json: bool) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    pool = await create_pool(settings)
    try:
        repo = Repository(pool)
        docs, chunks = await repo.counts()
        if chunks == 0:
            raise SystemExit(
                "No chunks in the database. Run `py -3.11 -m scripts.ingest` first."
            )
        retriever = Retriever(repo, settings)
        directory = DirectoryService.from_json(settings.directory_path)
        agent = AgentService(KnowledgeAgent(retriever, directory), settings)

        cases = load_dataset()
        if limit:
            cases = cases[:limit]

        report = await Evaluator(retriever, agent, settings.retrieval_top_k).run(cases)

        if as_json:
            payload = dataclasses.asdict(report)
            print(json.dumps(payload, default=lambda o: o.__dict__, indent=2))
        else:
            _print_report(report)
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation suite (uses Bedrock).")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()
    asyncio.run(_run(args.limit, args.json))


if __name__ == "__main__":
    main()
