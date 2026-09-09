"""POST /api/chat — runs the ReAct agent loop for a question and returns
the answer, full Thought/Action/Observation trace, and sources.
"""

from fastapi import APIRouter, HTTPException

from app.agent.agent_loop import run_agent
from app.models import ChatRequest, ChatResponse
from app.services import conversation_service
from app.utils.logger import logger

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        conversation_id = await conversation_service.get_or_create_conversation(request.conversation_id)
        history = await conversation_service.load_history_for_prompt(conversation_id)
        result = await run_agent(request.question, history)
        await conversation_service.save_turn(
            conversation_id,
            request.question,
            result.answer,
            [step.model_dump(mode="json") for step in result.steps],
            [source.model_dump(mode="json") for source in result.sources],
        )
        return ChatResponse(
            conversation_id=conversation_id,
            steps=result.steps,
            answer=result.answer,
            sources=result.sources,
        )
    except Exception as exc:
        logger.exception("Agent run failed for question: %s", request.question)
        raise HTTPException(status_code=500, detail=f"Agent run failed: {exc}") from exc
