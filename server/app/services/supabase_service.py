"""Supabase client wrapper — Postgres only (Auth/Storage unused), matching
Section 7 of requirements.md.

supabase-py's client is synchronous; every call here is wrapped with
asyncio.to_thread so these methods stay usable from async route handlers
without blocking the event loop.
"""

import asyncio
from typing import Optional

from supabase import Client, create_client

from app.config import settings
from app.utils.logger import logger

_client: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)


async def create_conversation() -> str:
    """Insert a new row into `conversations`, return its id."""

    def _insert() -> str:
        response = _client.table("conversations").insert({}).execute()
        return response.data[0]["id"]

    return await asyncio.to_thread(_insert)


async def add_message(
    conversation_id: str,
    role: str,
    content: str,
    steps: Optional[list] = None,
    sources: Optional[list] = None,
) -> None:
    """Insert a row into `messages`."""

    def _insert() -> None:
        _client.table("messages").insert(
            {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "steps": steps,
                "sources": sources,
            }
        ).execute()

    await asyncio.to_thread(_insert)


async def get_conversation_history(conversation_id: str) -> list[dict]:
    """All messages for a conversation, ordered by created_at ascending."""

    def _select() -> list[dict]:
        response = (
            _client.table("messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
            .execute()
        )
        return response.data

    return await asyncio.to_thread(_select)


async def health_check() -> bool:
    """Minimal select query wrapped in try/except."""

    def _select() -> None:
        _client.table("conversations").select("id").limit(1).execute()

    try:
        await asyncio.to_thread(_select)
        return True
    except Exception as exc:
        logger.error("Supabase health check failed: %s", exc)
        return False
