"""Embedding via gemini-embedding-001 (Section 6.3 of requirements.md).

Batched (batch size 2) with asyncio.gather(return_exceptions=True), adaptive
delay between batches (500ms -> 2000ms once more than 3 consecutive
failures have been seen, reset to 500ms on any success), and up to 3 retry
attempts per chunk before giving up on that chunk alone.
"""

import asyncio

from app.services import llm_service
from app.utils.logger import logger

BATCH_SIZE = 2
BASE_DELAY_SECONDS = 0.5
MAX_DELAY_SECONDS = 2.0
MAX_RETRIES = 3
FAILURE_THRESHOLD = 3


async def embed_chunks(chunks: list[str]) -> list[list[float] | None]:
    """Returns one embedding per input chunk, in the same order. A chunk
    that still fails after MAX_RETRIES attempts is represented as None
    (skipped) rather than failing the whole document — the caller
    (document_ingester) filters these out when building Qdrant points.
    """
    results: list[list[float] | None] = [None] * len(chunks)
    consecutive_failures = 0
    delay = BASE_DELAY_SECONDS

    for batch_start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[batch_start : batch_start + BATCH_SIZE]
        batch_results = await asyncio.gather(
            *(_embed_one(text) for text in batch), return_exceptions=True
        )
        for offset, result in enumerate(batch_results):
            idx = batch_start + offset
            if isinstance(result, Exception) or result is None:
                consecutive_failures += 1
                results[idx] = None
            else:
                consecutive_failures = 0
                results[idx] = result

        delay = MAX_DELAY_SECONDS if consecutive_failures > FAILURE_THRESHOLD else BASE_DELAY_SECONDS
        if batch_start + BATCH_SIZE < len(chunks):
            await asyncio.sleep(delay)

    return results


async def _embed_one(text: str) -> list[float] | None:
    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return await llm_service.embed_text(text, task_type="RETRIEVAL_DOCUMENT")
        except Exception as exc:
            last_exc = exc
            logger.warning("Embedding attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
    logger.warning("Giving up on chunk after %d attempts — skipping it. Last error: %s", MAX_RETRIES, last_exc)
    return None
