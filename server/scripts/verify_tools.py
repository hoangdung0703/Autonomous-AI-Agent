"""Temporary manual verification script for the 5 agent tools.

NOT part of the final architecture — just a sanity check for this step,
bypassing the agent loop (which doesn't exist yet) and calling
tool_registry.execute() directly, the same entry point the future
agent_loop.py will use.

The knowledge base is still empty at this point (ingestion pipeline is a
later step), so this only smoke-tests the tools that behave sensibly
against an empty collection: search_knowledge_base, list_documents, and
calculate_or_verify (both the success and ValueError paths).

get_document and compare_sections require real ingested data to exercise
meaningfully (an empty collection just exercises their "not found" paths,
which is already implicitly covered by search_knowledge_base/list_documents
here) — they'll be properly tested after Prompt 3 (ingestion pipeline).

Run from server/ after filling in real credentials in .env:

    python scripts/verify_tools.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import tool_registry  # noqa: E402
from app.services import vector_db_service  # noqa: E402


async def main() -> None:
    await vector_db_service.ensure_collection()

    print("=== search_knowledge_base (expect: 'No relevant chunks found.') ===")
    observation = await tool_registry.execute(
        "search_knowledge_base", {"query": "does not matter, kb is empty"}
    )
    print(observation)
    assert observation == "No relevant chunks found.", "expected empty-KB observation"

    print("\n=== list_documents (expect: graceful empty message) ===")
    observation = await tool_registry.execute("list_documents", {})
    print(observation)

    print("\n=== calculate_or_verify — valid expression ===")
    observation = await tool_registry.execute(
        "calculate_or_verify",
        {"expression": "(8/12) * 12", "context": "annual leave for 8 months"},
    )
    print(observation)
    assert observation.startswith("Result:"), "expected a Result: observation"

    print("\n=== calculate_or_verify — invalid expression (expect an error observation, not a crash) ===")
    observation = await tool_registry.execute(
        "calculate_or_verify",
        {"expression": "this is not a valid ((( expression", "context": "should fail"},
    )
    print(observation)
    assert observation.startswith("Error executing calculate_or_verify:"), "expected an error observation"

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
