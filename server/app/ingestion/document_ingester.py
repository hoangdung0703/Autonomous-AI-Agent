"""Orchestrates the full ingestion pipeline for a single document —
text extraction (falling back to OCR for scanned files) -> semantic
chunking -> embedding -> Qdrant upsert (Section 6 of requirements.md).

Resilient by design: any failure for one document is caught, logged, and
reported in the returned status rather than raised, so a single bad file
never aborts a whole seed run.
"""

import uuid
from pathlib import Path

from qdrant_client.http import models as qdrant_models

from app.ingestion import embedder, ocr_extractor, semantic_chunker, text_extractor
from app.ingestion.text_extractor import ScannedDocumentError
from app.services import vector_db_service
from app.utils.logger import logger

# Rough constant used only for the cosmetic, explicitly-approximate
# page_estimate payload field (Section 6.2) — not derived from any real
# page count, since text_extractor/ocr_extractor return plain text only.
AVG_CHARS_PER_PAGE = 3000

_ID_NAMESPACE = uuid.NAMESPACE_DNS


async def ingest_document(file_path: str, category: str) -> dict:
    document_name = Path(file_path).name
    try:
        try:
            text = await text_extractor.extract_text(file_path)
        except ScannedDocumentError:
            logger.info("%s has no extractable text layer — falling back to OCR", document_name)
            text = await ocr_extractor.extract_via_ocr(file_path)

        sections = semantic_chunker.chunk(text)
        if not sections:
            return {"document_name": document_name, "chunks_created": 0, "status": "failed: no chunks produced"}

        chunk_texts = [section["text"] for section in sections]
        embeddings = await embedder.embed_chunks(chunk_texts)

        points = []
        cumulative_chars = 0
        for chunk_index, (section, embedding) in enumerate(zip(sections, embeddings)):
            char_count = len(section["text"])
            if embedding is None:
                logger.warning(
                    "Skipping chunk %d of %s — embedding failed after retries", chunk_index, document_name
                )
                cumulative_chars += char_count
                continue

            page_estimate = max(1, round(cumulative_chars / AVG_CHARS_PER_PAGE) + 1)
            point_id = str(uuid.uuid5(_ID_NAMESPACE, f"{document_name}_chunk_{chunk_index}"))
            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "document_name": document_name,
                        "category": category,
                        "section_title": section["section_title"],
                        "chunk_index": chunk_index,
                        "page_estimate": page_estimate,
                        "char_count": char_count,
                        "text": section["text"],
                    },
                )
            )
            cumulative_chars += char_count

        if not points:
            return {"document_name": document_name, "chunks_created": 0, "status": "failed: all chunks failed to embed"}

        await vector_db_service.upsert_points(points)
        return {"document_name": document_name, "chunks_created": len(points), "status": "success"}

    except Exception as exc:
        logger.error("Ingestion failed for %s: %s", document_name, exc)
        return {"document_name": document_name, "chunks_created": 0, "status": f"failed: {exc}"}
