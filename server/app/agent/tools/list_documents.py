"""list_documents(category=None) tool.

Returns unique document names + category + chunk count from Qdrant payload
aggregation, optionally filtered by category. Used for agent orientation
before searching.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.services import vector_db_service


class ListDocumentsParams(BaseModel):
    category: Optional[Literal["hr", "legal", "ops"]] = Field(
        default=None,
        description="Restrict the listing to one document category. Omit to list documents from all categories.",
    )


async def list_documents(category: Optional[str] = None) -> str:
    documents = await vector_db_service.list_documents(category=category)
    if not documents:
        return "No documents are indexed in the knowledge base yet."

    lines = [
        f"- {doc['document_name']} (category: {doc['category']}, chunks: {doc['chunk_count']})"
        for doc in documents
    ]
    return "\n".join(lines)
