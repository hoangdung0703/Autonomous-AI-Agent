"""Pydantic request/response models for the API and internal agent state.

Matches Section 8 of requirements.md. FastAPI uses these to auto-generate
OpenAPI docs at /docs.
"""

from typing import Literal, Optional

from pydantic import BaseModel


class Step(BaseModel):
    type: Literal["thought", "action", "observation", "retry"]
    content: Optional[str] = None
    tool: Optional[str] = None
    params: Optional[dict] = None


class Source(BaseModel):
    document_name: str
    section_title: str
    excerpt: str


class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    conversation_id: str
    steps: list[Step]
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    supabase: str
    gemini: str
    documents_indexed: int


class AgentResult(BaseModel):
    """Result of a full agent run — the future agent_loop's return type."""

    answer: str
    steps: list[Step]
    sources: list[Source]
