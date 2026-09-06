"""Higher-level conversation orchestration built on top of
supabase_service.
"""

from typing import Optional

from app.services import supabase_service


async def get_or_create_conversation(conversation_id: Optional[str]) -> str:
    """Return the given conversation id unchanged, or create a new one."""
    if conversation_id:
        return conversation_id
    return await supabase_service.create_conversation()


async def load_history_for_prompt(conversation_id: str) -> list[dict]:
    """Normalized [{role, content}, ...] history, oldest first.

    This is plain data, not Gemini `contents` objects — building the actual
    `contents` list for the model belongs to agent_loop.py in a later step.
    """
    messages = await supabase_service.get_conversation_history(conversation_id)
    return [{"role": message["role"], "content": message["content"]} for message in messages]


async def save_turn(
    conversation_id: str,
    question: str,
    answer: str,
    steps: list,
    sources: list,
) -> None:
    """Persist both the user message and the agent's reply for this turn."""
    await supabase_service.add_message(conversation_id, role="user", content=question)
    await supabase_service.add_message(
        conversation_id, role="agent", content=answer, steps=steps, sources=sources
    )
