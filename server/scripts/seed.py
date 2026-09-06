"""Run once to ingest knowledge-base/ into Qdrant (Section 6.5 of
requirements.md).

Scans knowledge-base/ recursively, detects category from the immediate
parent folder name (hr/legal/ops), and runs the ingestion pipeline per
file. Incremental by default — a file whose document_name is already
indexed in Qdrant is skipped; pass --force to re-ingest everything.

Usage (from server/):
    python scripts/seed.py
    python scripts/seed.py --force
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion.document_ingester import ingest_document  # noqa: E402
from app.services import vector_db_service  # noqa: E402

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent.parent / "knowledge-base"
SUPPORTED_SUFFIXES = {".pdf", ".docx", ".xlsx"}


def _scan_files() -> list[tuple[Path, str]]:
    """Returns (file_path, category) pairs, category from the parent folder
    name, sorted for deterministic run order."""
    files = []
    for path in sorted(KNOWLEDGE_BASE_DIR.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            category = path.parent.name
            files.append((path, category))
    return files


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest server/knowledge-base/ into Qdrant.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest every file, even ones already indexed in Qdrant.",
    )
    args = parser.parse_args()

    await vector_db_service.ensure_collection()

    already_indexed: set[str] = set()
    if not args.force:
        existing = await vector_db_service.list_documents()
        already_indexed = {doc["document_name"] for doc in existing}

    files = _scan_files()
    successes = 0
    skipped = 0
    total_chunks = 0

    for file_path, category in files:
        document_name = file_path.name

        if not args.force and document_name in already_indexed:
            print(f"[SKIPPED] {document_name} (already indexed)")
            skipped += 1
            continue

        result = await ingest_document(str(file_path), category)
        if result["status"] == "success":
            print(f"[SUCCESS] {document_name} -> {result['chunks_created']} chunks")
            successes += 1
            total_chunks += result["chunks_created"]
        else:
            reason = result["status"].removeprefix("failed: ")
            print(f"[FAIL] {document_name}: {reason}")

    attempted = len(files) - skipped
    print()
    print(f"{successes}/{attempted} documents indexed, {total_chunks} total chunks ({skipped} skipped)")


if __name__ == "__main__":
    asyncio.run(main())
