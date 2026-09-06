"""get_document(document_name) tool.

Retrieves all chunks for a document ordered by chunk_index (Qdrant
scroll/filter by payload) and returns the concatenated full content, for
deep reading of a specific file.
"""

from pydantic import BaseModel, Field

from app.services import vector_db_service


class GetDocumentParams(BaseModel):
    document_name: str = Field(
        ...,
        description="The exact file name of the document to read in full, e.g. 'leave-policy.pdf'. Use list_documents first if you don't know the exact name.",
    )


async def get_document(document_name: str) -> str:
    chunks = await vector_db_service.scroll_by_document(document_name)
    if not chunks:
        return f"Document not found: {document_name}. Use list_documents to see available files."
    return "\n\n".join(chunk.get("text", "") for chunk in chunks)
