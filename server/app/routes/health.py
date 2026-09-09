"""GET /api/health — reports Qdrant/Supabase/Gemini connectivity and the
number of documents indexed.
"""

from fastapi import APIRouter

from app.models import HealthResponse
from app.services import llm_service, supabase_service, vector_db_service
from app.utils.logger import logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    qdrant_ok = await _check(vector_db_service.health_check, "Qdrant")
    supabase_ok = await _check(supabase_service.health_check, "Supabase")
    gemini_ok = await _check(llm_service.health_check, "Gemini")

    documents_indexed = 0
    if qdrant_ok:
        try:
            documents_indexed = await vector_db_service.count_points()
        except Exception as exc:
            logger.error("Failed to count indexed documents: %s", exc)
            qdrant_ok = False

    status = "ok" if (qdrant_ok and supabase_ok and gemini_ok) else "degraded"

    return HealthResponse(
        status=status,
        qdrant="connected" if qdrant_ok else "unreachable",
        supabase="connected" if supabase_ok else "unreachable",
        gemini="reachable" if gemini_ok else "unreachable",
        documents_indexed=documents_indexed,
    )


async def _check(check_fn, label: str) -> bool:
    try:
        return bool(await check_fn())
    except Exception as exc:
        logger.error("%s health check raised: %s", label, exc)
        return False
