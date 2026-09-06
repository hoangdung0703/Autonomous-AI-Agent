"""Gemini wrapper — native function calling for the ReAct loop.

Wraps the `google-genai` SDK:
  - generate_with_tools drives the agent loop's Thought/Action step using
    Gemini's native function calling (Section 13.1) — it takes tool
    declarations as a parameter and never hardcodes a specific tool.
  - generate_simple is a plain single-turn call with no tools, used later by
    the baseline RAG and LLM-as-judge evaluation.
  - embed_text wraps gemini-embedding-001 for both ingestion
    (RETRIEVAL_DOCUMENT) and query (RETRIEVAL_QUERY) embeddings.
"""

from typing import Literal

from google import genai
from google.genai import types

from app.config import settings
from app.utils.logger import logger

_client = genai.Client(api_key=settings.GOOGLE_API_KEY)

# Hard-capped to control cost — thinking tokens bill as output tokens at
# $12/1M for gemini-3.1-pro-preview. Do not raise this without discussing
# budget impact first. Applied to every Gemini Pro call in this project —
# not an env var, not a caller-overridable parameter.
_THINKING_CONFIG = types.ThinkingConfig(thinking_level="low")


async def generate_with_tools(
    contents: list,
    tools: list[types.FunctionDeclaration],
    system_instruction: str,
) -> types.GenerateContentResponse:
    """Call Gemini with native function calling enabled.

    Returns the raw response — the caller inspects
    response.candidates[0].content.parts for text and/or function_call, per
    Section 13.1. `tools` are supplied by the caller (the future
    tool_registry), so this wrapper stays tool-agnostic.
    """
    return await _client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=[types.Tool(function_declarations=tools)],
            # Our own ReAct loop dispatches each function call manually
            # (Section 4.3) — disable the SDK's automatic function-calling
            # so it never executes a tool on our behalf.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=_THINKING_CONFIG,
        ),
    )


async def generate_simple(prompt: str) -> str:
    """Plain single-turn generation with no tools (baseline RAG, LLM-as-judge)."""
    response = await _client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=_THINKING_CONFIG,
        ),
    )
    return response.text or ""


async def generate_from_image(image_bytes: bytes, prompt: str) -> str:
    """Gemini Vision call — used by ocr_extractor for scanned documents that
    have no extractable text layer (Section 6.1). Same hard-capped thinking
    config as every other Gemini Pro call; no separate un-capped code path.
    """
    response = await _client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            thinking_config=_THINKING_CONFIG,
        ),
    )
    return response.text or ""


async def embed_text(
    text: str,
    task_type: Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"] = "RETRIEVAL_DOCUMENT",
) -> list[float]:
    """Embed a single text via gemini-embedding-001.

    task_type differentiates document embeddings (ingestion) from query
    embeddings (search) — this asymmetry improves retrieval quality.
    """
    response = await _client.aio.models.embed_content(
        model=settings.GEMINI_EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=settings.EMBEDDING_DIMENSIONS,
        ),
    )
    return response.embeddings[0].values


async def health_check() -> bool:
    """Minimal reachability check for GET /api/health."""
    try:
        await generate_simple("ping")
        return True
    except Exception as exc:
        logger.error("Gemini health check failed: %s", exc)
        return False
