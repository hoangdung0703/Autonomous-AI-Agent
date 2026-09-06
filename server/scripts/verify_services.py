"""Temporary manual verification script for the services layer.

NOT part of the final architecture — just a sanity check for this step.
Run from server/ after filling in real credentials in .env:

    python scripts/verify_services.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services import llm_service, supabase_service, vector_db_service  # noqa: E402


async def main() -> None:
    results: dict[str, bool] = {}

    results["Gemini (llm_service.health_check)"] = await llm_service.health_check()

    try:
        await vector_db_service.ensure_collection()
        results["Qdrant (ensure_collection + health_check)"] = await vector_db_service.health_check()
    except Exception as exc:
        print(f"Qdrant setup raised: {exc}")
        results["Qdrant (ensure_collection + health_check)"] = False

    results["Supabase (supabase_service.health_check)"] = await supabase_service.health_check()

    print()
    print("=== Service verification results ===")
    for name, ok in results.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")


if __name__ == "__main__":
    asyncio.run(main())
