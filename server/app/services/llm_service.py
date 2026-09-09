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

import asyncio
from typing import Awaitable, Callable, Literal, TypeVar

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import settings
from app.utils.logger import logger

_T = TypeVar("_T")

# Request timeout (HttpOptions.timeout is in milliseconds) applied to every
# call made through this client — generate_with_tools, generate_simple,
# embed_text, and generate_from_image all share it. Without this, a hung
# network request blocks the caller indefinitely instead of raising; 60s is
# generous for a single Gemini call (including evaluation/run_eval.py's
# LLM-as-judge calls) while still failing fast enough to retry or surface
# the error instead of stalling a batch run for tens of minutes.
_client = genai.Client(
    api_key=settings.GOOGLE_API_KEY,
    http_options=types.HttpOptions(timeout=60_000),
)

# Hard-capped to control cost — thinking tokens bill as output tokens at
# $12/1M for gemini-3.1-pro-preview. Do not raise this without discussing
# budget impact first. Applied to every Gemini Pro call in this project —
# not an env var, not a caller-overridable parameter.
_THINKING_CONFIG = types.ThinkingConfig(thinking_level="low")

# Retry policy for transient upstream failures on the two live user-facing
# call sites (generate_with_tools drives the agent loop, generate_simple
# drives GET /api/health and generate_from_image/embed_text are left alone).
# 3 attempts, 1s initial delay, doubling — each attempt is still bounded by
# the client's own 60s HttpOptions timeout above, so this wraps around that
# timeout rather than replacing it: worst case is 3 timed-out attempts plus
# ~3s of backoff between them, not one longer call.
_RETRY_ATTEMPTS = 3
_RETRY_INITIAL_DELAY_SECONDS = 1.0
_RETRY_BACKOFF_MULTIPLIER = 2

# Status codes worth retrying — momentary overload/timeout on Google's side,
# not a problem with our request. 500/502/503/504 are the standard transient
# server codes; 499 (client closed request / cancelled) is included because
# in practice it's Gemini's own upstream cancelling a slow request — the same
# transient failure mode as 503/504, just wrapped in a ClientError by the SDK
# since 499 falls in the 4xx range. A genuine client error (400 invalid
# argument, 401/403 auth, 404 not found, ...) is NOT in this set, so it fails
# immediately on the first attempt instead of retrying 3 times for nothing.
_RETRYABLE_STATUS_CODES = {499, 500, 502, 503, 504}


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, genai_errors.ServerError):
        return True
    if isinstance(exc, genai_errors.ClientError):
        return exc.code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))


async def _call_with_retry(fn: Callable[[], Awaitable[_T]]) -> _T:
    """Runs `fn`, retrying on transient upstream errors only (see
    _is_transient) with exponential backoff. A non-transient error (a real
    client mistake) or the final attempt's error is re-raised immediately —
    shared by generate_with_tools and generate_simple so neither duplicates
    this loop.
    """
    delay = _RETRY_INITIAL_DELAY_SECONDS
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            return await fn()
        except (genai_errors.APIError, httpx.TimeoutException, httpx.ConnectError) as exc:
            if not _is_transient(exc) or attempt == _RETRY_ATTEMPTS:
                raise
            logger.warning(
                "Transient Gemini error on attempt %d/%d (%s) — retrying in %.0fs",
                attempt,
                _RETRY_ATTEMPTS,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            delay *= _RETRY_BACKOFF_MULTIPLIER


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

    async def _call() -> types.GenerateContentResponse:
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

    return await _call_with_retry(_call)


async def generate_simple(prompt: str) -> str:
    """Plain single-turn generation with no tools (baseline RAG, LLM-as-judge)."""

    async def _call() -> types.GenerateContentResponse:
        return await _client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=_THINKING_CONFIG,
            ),
        )

    response = await _call_with_retry(_call)
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
