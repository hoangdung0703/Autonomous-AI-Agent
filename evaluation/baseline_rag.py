"""Simple non-agentic RAG baseline (Section 13.4 of requirements.md).

Embeds the question, retrieves the top 5 chunks from Qdrant with NO category
filter and NO min_score threshold (unlike search_knowledge_base's tool
implementation, which applies MIN_SCORE=0.70 — the baseline intentionally
does not get that relevance-filtering benefit), builds a one-shot prompt,
and calls llm_service.generate_simple() once. Always exactly 1 step, and
never calls any tool — in particular it cannot call calculate_or_verify, so
computation-category queries are expected to fail here (score 0%), which is
the whole point of comparing it against the ReAct agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

from app.services import llm_service, vector_db_service  # noqa: E402

TOP_K = 5


def _build_simple_prompt(question: str, chunks: list[dict]) -> str:
    if chunks:
        context = "\n\n".join(
            f"[{chunk.get('document_name')} / {chunk.get('section_title')}]\n{chunk.get('text', '')}"
            for chunk in chunks
        )
    else:
        context = "(no relevant chunks retrieved)"

    return (
        "You are an assistant answering questions about VinTech Corp using only the "
        "context below. If the context does not contain the answer, say so — do not "
        "fabricate information. Cite the source document and section when possible.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


async def query(question: str) -> dict:
    """Single-shot retrieve-then-generate baseline. Returns
    {"answer": str, "steps": 1, "tools_used": [], "retrieved_chunks": list[dict]}
    — retrieved_chunks is included so run_eval.py can score baseline
    faithfulness against the same chunks the baseline actually used.
    """
    query_vector = await llm_service.embed_text(question, task_type="RETRIEVAL_QUERY")
    chunks = await vector_db_service.search(query_vector=query_vector, limit=TOP_K)
    prompt = _build_simple_prompt(question, chunks)
    answer = await llm_service.generate_simple(prompt)
    return {"answer": answer, "steps": 1, "tools_used": [], "retrieved_chunks": chunks}
