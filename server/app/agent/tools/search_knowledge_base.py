"""search_knowledge_base(query, category=None) tool.

Embeds the query via gemini-embedding-001, queries Qdrant with an optional
category payload filter (hr | legal | ops), and returns up to the top 5
chunks (document_name, section_title, excerpt, similarity_score) scoring at
or above MIN_SCORE, formatted as a plain-text observation for the model to
read. If nothing clears the threshold, returns exactly
"No relevant chunks found." — the trigger for the agent's self-correction
behavior (Section 4.4/13.3 of requirements.md).
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.services import llm_service, vector_db_service

EXCERPT_CHARS = 400
TOP_K = 5

# Tuned against real observed scores (server/scripts/verify_agent.py):
# genuinely relevant top hits scored ~0.73/0.73/0.72, while an off-topic
# query's best hit scored ~0.67 — this sits strictly between the two so a
# weak/off-topic query can actually produce "No relevant chunks found."
# (Section 4.4/13.3's self-correction trigger), without rejecting real hits.
MIN_SCORE = 0.70


class SearchKnowledgeBaseParams(BaseModel):
    query: str = Field(
        ...,
        description="The natural-language question or topic to search for in the knowledge base.",
    )
    category: Optional[Literal["hr", "legal", "ops"]] = Field(
        default=None,
        description="Restrict the search to one document category. Omit to search all categories.",
    )


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No relevant chunks found."

    lines = []
    for i, chunk in enumerate(results, start=1):
        excerpt = (chunk.get("text") or "")[:EXCERPT_CHARS]
        lines.append(
            f"{i}. document: {chunk.get('document_name')} | section: {chunk.get('section_title')} "
            f"| score: {chunk.get('similarity_score'):.4f}\n"
            f"   excerpt: {excerpt}"
        )
    return "\n".join(lines)


async def search_knowledge_base(
    query: str,
    category: Optional[str] = None,
    document_name: Optional[str] = None,
    limit: int = TOP_K,
    min_score: Optional[float] = MIN_SCORE,
) -> str:
    """Implementation. `document_name`, `limit`, and `min_score` are
    internal-only extra params used by compare_sections to scope a search
    to a single document, cap it to 3 results, and disable score filtering
    (side-by-side comparison stays useful even at moderate relevance) —
    none of these are part of the model-facing schema.
    """
    query_vector = await llm_service.embed_text(query, task_type="RETRIEVAL_QUERY")
    results = await vector_db_service.search(
        query_vector=query_vector,
        limit=limit,
        category=category,
        document_name=document_name,
        min_score=min_score,
    )
    return _format_results(results)
