"""compare_sections(doc1, doc2, topic) tool.

Calls the search_knowledge_base implementation twice — once scoped to doc1,
once to doc2, via the document_name filter added to vector_db_service.search
— and returns both result sets side by side for comparison.
"""

from pydantic import BaseModel, Field

from app.agent.tools.search_knowledge_base import search_knowledge_base

TOP_K_PER_DOC = 3


class CompareSectionsParams(BaseModel):
    doc1: str = Field(..., description="The exact file name of the first document to compare, e.g. 'leave-policy.pdf'.")
    doc2: str = Field(..., description="The exact file name of the second document to compare, e.g. 'employment-contract.pdf'.")
    topic: str = Field(..., description="The topic or question to compare between the two documents, e.g. 'sick leave allowance'.")


async def compare_sections(doc1: str, doc2: str, topic: str) -> str:
    # No score threshold here (min_score=None) — side-by-side comparison is
    # still useful even at moderate relevance, unlike a plain search.
    results_1 = await search_knowledge_base(topic, document_name=doc1, limit=TOP_K_PER_DOC, min_score=None)
    results_2 = await search_knowledge_base(topic, document_name=doc2, limit=TOP_K_PER_DOC, min_score=None)
    return (
        f"=== Results from {doc1} ===\n"
        f"{results_1}\n\n"
        f"=== Results from {doc2} ===\n"
        f"{results_2}"
    )
