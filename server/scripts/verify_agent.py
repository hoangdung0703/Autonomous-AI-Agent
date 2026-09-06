"""Temporary manual verification script for the full ReAct agent loop.

NOT part of the final architecture — just a sanity check for this step.
Runs 3 real end-to-end questions against the actual seeded knowledge base
(9 documents, 63 chunks), with no conversation history, and prints the full
Thought/Action/Observation trace plus the final answer and sources for
each — bypassing routes/conversation_service entirely (run_agent only needs
a question and a history list, which we pass as []).

Run from server/ after filling in real credentials in .env and running
scripts/seed.py at least once:

    python scripts/verify_agent.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.agent_loop import run_agent  # noqa: E402

QUESTIONS = [
    "What is VinTech Corp's probationary period duration?",
    "An employee who worked 8 months — how many annual leave days are they entitled to?",
    "What is VinTech Corp's remote work policy?",
]


def _print_step(i: int, step) -> None:
    if step.type == "thought":
        print(f"  [{i}] THOUGHT: {step.content}")
    elif step.type == "action":
        print(f"  [{i}] ACTION: {step.tool}({step.params})")
    elif step.type == "observation":
        print(f"  [{i}] OBSERVATION: {step.content}")


async def main() -> None:
    for question in QUESTIONS:
        print("=" * 100)
        print(f"QUESTION: {question}")
        print("=" * 100)

        result = await run_agent(question, conversation_history=[])

        print()
        for i, step in enumerate(result.steps, start=1):
            _print_step(i, step)

        print()
        print(f"FINAL ANSWER: {result.answer}")
        print()
        print("SOURCES:")
        for source in result.sources:
            print(f"  - {source.document_name} / {source.section_title}: {source.excerpt[:120]}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
