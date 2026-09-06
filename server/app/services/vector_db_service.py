"""Qdrant client wrapper — collection lifecycle, upsert, search, and
aggregation helpers, matching Section 6.4 of requirements.md.
"""

from typing import Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from app.config import settings
from app.utils.logger import logger

_client = AsyncQdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


async def ensure_collection() -> None:
    """Create the collection with the configured vector size/distance, only
    if it doesn't already exist. Also ensures payload indexes exist for the
    two fields we filter on (category, document_name) — Qdrant Cloud
    rejects a filtered search/scroll on a keyword field with no index once
    the collection is non-empty.
    """
    if not await _client.collection_exists(settings.QDRANT_COLLECTION):
        await _client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=settings.EMBEDDING_DIMENSIONS,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection '%s'", settings.QDRANT_COLLECTION)

    for field_name in ("category", "document_name"):
        await _client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )


async def upsert_points(points: list[models.PointStruct]) -> None:
    """Batch upsert points into the collection."""
    await _client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)


async def search(
    query_vector: list[float],
    limit: int = 5,
    category: Optional[str] = None,
    document_name: Optional[str] = None,
    min_score: Optional[float] = None,
) -> list[dict]:
    """Vector search, optionally filtered by payload.category and/or
    payload.document_name, and optionally dropping results below
    `min_score` (cosine similarity). Default None preserves prior
    behavior exactly — no score filtering, always top-`limit` results.

    Returns a list of payload dicts, each augmented with a
    "similarity_score" key.
    """
    conditions = []
    if category:
        conditions.append(models.FieldCondition(key="category", match=models.MatchValue(value=category)))
    if document_name:
        conditions.append(
            models.FieldCondition(key="document_name", match=models.MatchValue(value=document_name))
        )
    query_filter = models.Filter(must=conditions) if conditions else None
    response = await _client.query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        score_threshold=min_score,
        with_payload=True,
    )
    return [{**point.payload, "similarity_score": point.score} for point in response.points]


async def scroll_by_document(document_name: str) -> list[dict]:
    """All chunk payloads for a document, ordered by chunk_index ascending."""
    scroll_filter = models.Filter(
        must=[models.FieldCondition(key="document_name", match=models.MatchValue(value=document_name))]
    )
    records: list[models.Record] = []
    offset = None
    while True:
        batch, offset = await _client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=100,
            offset=offset,
            with_payload=True,
        )
        records.extend(batch)
        if offset is None:
            break
    payloads = [record.payload for record in records]
    payloads.sort(key=lambda p: p.get("chunk_index", 0))
    return payloads


async def list_documents(category: Optional[str] = None) -> list[dict]:
    """Aggregate {document_name, category, chunk_count} across all points,
    optionally filtered by category, by scrolling and grouping in Python.
    """
    scroll_filter = None
    if category:
        scroll_filter = models.Filter(
            must=[models.FieldCondition(key="category", match=models.MatchValue(value=category))]
        )
    aggregation: dict[str, dict] = {}
    offset = None
    while True:
        batch, offset = await _client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=scroll_filter,
            limit=200,
            offset=offset,
            with_payload=["document_name", "category"],
        )
        for record in batch:
            name = record.payload.get("document_name")
            if name is None:
                continue
            entry = aggregation.setdefault(
                name,
                {"document_name": name, "category": record.payload.get("category"), "chunk_count": 0},
            )
            entry["chunk_count"] += 1
        if offset is None:
            break
    return list(aggregation.values())


async def count_points() -> int:
    """Distinct document count — used for `documents_indexed` in the health
    check, since that better reflects "documents indexed" than raw chunk
    count.
    """
    documents = await list_documents()
    return len(documents)


async def health_check() -> bool:
    """Verify the collection exists and Qdrant is reachable."""
    try:
        return await _client.collection_exists(settings.QDRANT_COLLECTION)
    except Exception as exc:
        logger.error("Qdrant health check failed: %s", exc)
        return False
