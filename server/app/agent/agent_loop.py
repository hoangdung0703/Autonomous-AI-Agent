"""Self-implemented ReAct (Reasoning + Acting) loop (Section 4 of
requirements.md) — no LangChain AgentExecutor, no agent framework.

Drives the Thought/Action/Observation cycle using Gemini's native function
calling (Section 13.1): each turn's response is inspected directly for a
function_call part (loop continues) or its absence (Final Answer) —
execution branches entirely on the model's own decision, never a hardcoded
tool sequence. Self-correction (Section 4.4) is not separate code: it
emerges from faithfully feeding every observation — including "No relevant
chunks found." — back into `contents`, letting the model read it on the
next turn and decide what to do.
"""

import re
import unicodedata

from google.genai import types

from app.agent import tool_registry
from app.config import settings
from app.models import AgentResult, Source, Step
from app.services import llm_service
from app.utils.parse_agent_response import ParsedAgentResponse, parse_agent_response

SYSTEM_INSTRUCTION = """You are an AI Agent for enterprise knowledge retrieval at VinTech Corp.
Before each tool call, briefly state your reasoning as plain text (this is your "Thought").
Then call the appropriate function.
When you have enough information, respond with plain text only (no function call) —
this is your Final Answer. Always cite the source document and section name in the
Final Answer.

Every document_name and section_title in your Final Answer must be copied exactly from
text that appeared in a previous Observation in this conversation. Never state a
document name, section title, or numeric formula that did not literally appear in an
Observation you received — not from general knowledge, not by guessing a
plausible-sounding filename, and not by inventing a formula that "sounds right." If
you're not certain a fact is grounded in a real Observation, say you don't have enough
information rather than guessing. Fabricating a source or a number is worse than
admitting you could not find an answer.

For numeric questions, retrieve the formula first, then use calculate_or_verify.
When your Final Answer needs to combine multiple facts from different sources (e.g. a
comparison across documents), explicitly verify you have included every fact you
retrieved that is relevant to the question before responding — do not drop a retrieved
fact when synthesizing the final answer."""

_EMPTY_ANSWER_NUDGE = (
    "Your previous response had no content. Please provide either a tool call to "
    "gather more information, or a complete final answer with your findings so far."
)

_MALFORMED_ANSWER_NUDGE = (
    "Your previous response was not valid natural-language text -- it appeared to be "
    "garbled output or a raw echo of tool-call/tool-response formatting rather than a "
    "genuine answer. Please provide either a proper tool call, or a clear, "
    "natural-language final answer with your findings so far."
)

_UNVERIFIED_CITATION_NUDGE = (
    "Your previous answer referenced a document that was not actually retrieved in "
    "this conversation. Only cite documents and facts that appeared in your own "
    "Observations, or say you don't have enough information."
)

# Neither an empty response nor a malformed/hallucinated-looking one is real
# reasoning progress, so retrying either must not eat into MAX_AGENT_STEPS
# (the budget for genuine tool-call/reasoning steps) -- that was the
# original bug: a query needing several real tool calls could silently lose
# its remaining budget to invisible retries and hit the step-limit fallback
# despite having already gathered everything it needed. This is a small,
# separate budget, and every retry is recorded as a visible "retry" Step so
# it can never be silently absorbed again.
MAX_INVALID_RETRIES = 2

# Markers that should never co-occur in a genuine natural-language Final
# Answer -- this is exactly the internal formatting search_knowledge_base's
# _format_results emits ("N. document: ... | section: ... | score: ...").
# The system prompt never asks the model to surface a similarity score to
# the user, so a real answer citing sources says "document"/"section" at
# most, never "score:" as well -- requiring all three keeps this conservative.
_TOOL_ECHO_MARKERS = ("document:", "section:", "score:")

# Substrings that only appear in a raw tool-call/tool-response echo, never
# in genuine prose (e.g. Gemini's own function-calling wire format).
_TOOL_ECHO_SUBSTRINGS = ("response:", "default_api:")

# Unicode script blocks that never legitimately appear in this project's
# English-language answers (the knowledge base and system prompt are
# English, with occasional Vietnamese Latin-script proper nouns) -- CJK
# ideographs, Hangul, Hiragana/Katakana, and CJK punctuation. A character in
# one of these is a strong signal of sampling-noise garbage, as observed in
# practice (a stray Hangul syllable leading a garbled response).
_SUSPICIOUS_UNICODE_RANGES = (
    (0x3000, 0x30FF),  # CJK punctuation, Hiragana, Katakana
    (0x3400, 0x9FFF),  # CJK Unified Ideographs (+ Extension A)
    (0xAC00, 0xD7A3),  # Hangul syllables
)


def is_malformed_response(text: str) -> bool:
    """Conservative heuristic guard against a rare Gemini failure mode: a
    non-empty response that is garbled sampling noise or a literal echo of
    tool-call/tool-response formatting, rather than genuine natural-language
    text. Deliberately conservative -- it should catch obvious garbage, not
    flag legitimate answers that happen to mention technical terms.
    """
    if not text:
        return False

    for ch in text:
        if ch in "\n\r\t":
            continue
        if unicodedata.category(ch).startswith("C"):  # control/format/surrogate/etc.
            return True
        code_point = ord(ch)
        if any(low <= code_point <= high for low, high in _SUSPICIOUS_UNICODE_RANGES):
            return True

    lowered = text.lower()
    if any(marker in lowered for marker in _TOOL_ECHO_SUBSTRINGS):
        return True
    if all(marker in lowered for marker in _TOOL_ECHO_MARKERS):
        return True

    return False

# Tools whose string observation lists retrieved chunks — parsed back into
# structured Source entries for ChatResponse.sources. The format is exactly
# what search_knowledge_base._format_results emits (also used, twice, inside
# compare_sections' output), so one pattern covers both tools.
_SOURCE_TOOLS = {"search_knowledge_base", "compare_sections"}

_RESULT_PATTERN = re.compile(
    r"^\d+\.\s*document:\s*(?P<document>.+?)\s*\|\s*section:\s*(?P<section>.+?)\s*\|\s*score:.*?\n"
    r"\s*excerpt:\s*(?P<excerpt>.*?)(?=\n\d+\.\s*document:|\Z)",
    re.MULTILINE | re.DOTALL,
)

# A Final Answer's prose doesn't follow _RESULT_PATTERN's structured
# "document: X | section: Y" format (it cites documents inline, e.g.
# "according to employee-handbook.pdf" or "**compensation-and-benefits.pdf**")
# so citation verification matches on the one thing every real document_name
# has in common instead: a supported knowledge-base file extension (kept in
# sync with seed.py's SUPPORTED_SUFFIXES). Markdown wrapping (backticks,
# asterisks) is naturally excluded since it isn't a word/hyphen/dot character.
_DOCUMENT_CITATION_PATTERN = re.compile(r"\b[\w][\w\-.]*\.(?:pdf|docx|xlsx)\b", re.IGNORECASE)


def _extract_cited_document_names(text: str) -> set[str]:
    """Filename-shaped citations found in a Final Answer's prose, for
    verifying each one actually appeared in a real Observation (Section 4.4
    self-correction / anti-fabrication guard)."""
    return {match.group(0) for match in _DOCUMENT_CITATION_PATTERN.finditer(text)}


async def run_agent(question: str, conversation_history: list[dict]) -> AgentResult:
    contents = _build_initial_contents(question, conversation_history)
    steps: list[Step] = []
    sources: list[Source] = []
    seen_sources: set[tuple[str, str]] = set()

    # conversation_history only carries {role, content} (see
    # conversation_service.load_history_for_prompt) -- no structured sources
    # -- but a prior agent turn's Final Answer already passed this same
    # citation guard when it was produced, so any document_name cited in it
    # is a real, previously-grounded fact, not a fresh guess. Without this,
    # correctly recalling an earlier turn's source purely from memory (e.g.
    # "what document was that formula from again?" two turns later, with no
    # re-search) would be misflagged as fabrication just because it wasn't
    # re-observed in *this* run_agent() call -- that was the scoping bug.
    history_documents: set[str] = set()
    for message in conversation_history:
        if message["role"] == "agent":
            history_documents |= _extract_cited_document_names(message["content"])

    for _ in range(settings.MAX_AGENT_STEPS):
        known_documents = history_documents | {source.document_name for source in sources}
        response, parsed = await _generate_with_retry(contents, steps, known_documents)
        if response is None:
            # Invalid-retry budget exhausted without ever getting a real
            # function_call or a genuine non-empty, non-malformed answer --
            # distinct from the step-limit fallback below, since this means
            # the model itself never produced usable content, not that it
            # ran out of room to keep reasoning.
            return AgentResult(
                answer="Agent produced no valid response after retries.",
                steps=steps,
                sources=sources,
            )

        if parsed.thought_text:
            steps.append(Step(type="thought", content=parsed.thought_text))

        if parsed.function_call is None:
            # _generate_with_retry only returns a response with no
            # function_call when the text is non-empty and passes the
            # malformed-response check (Section 13.2) -> the model's own
            # stopping decision, not a max-steps cutoff.
            return AgentResult(answer=parsed.thought_text, steps=steps, sources=sources)

        tool_name = parsed.function_call.name
        params = dict(parsed.function_call.args or {})
        steps.append(Step(type="action", tool=tool_name, params=params))

        observation = await tool_registry.execute(tool_name, params)
        steps.append(Step(type="observation", content=observation))

        if tool_name in _SOURCE_TOOLS:
            for source in _extract_sources(observation):
                key = (source.document_name, source.section_title)
                if key not in seen_sources:
                    seen_sources.add(key)
                    sources.append(source)

        # Feed this turn back into contents so the next iteration has full
        # context: the model's own turn (thought + function_call, exactly
        # as returned) followed by the tool's response.
        contents.append(response.candidates[0].content)
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_function_response(name=tool_name, response={"result": observation})],
            )
        )

    return AgentResult(
        answer="Could not complete reasoning within step limit.",
        steps=steps,
        sources=sources,
    )


async def _generate_with_retry(
    contents: list[types.Content], steps: list[Step], known_documents: set[str]
) -> tuple[types.GenerateContentResponse | None, ParsedAgentResponse | None]:
    """Call the model, retrying on an empty/whitespace response, a
    malformed/hallucinated-looking one (see is_malformed_response), or a
    Final Answer that cites a document_name never seen in a real Observation
    in this conversation (anti-fabrication guard, e.g. the q17 regression
    where the agent cited a nonexistent "compensation-and-benefits.pdf") --
    all three share MAX_INVALID_RETRIES, entirely separate from the caller's
    MAX_AGENT_STEPS loop, since none of them represents a real reasoning
    step. Every retry is appended to `steps` as a "retry" Step, with a
    reason that identifies which check failed, so it is always visible in
    the trace and never silently absorbed.

    A function_call is always accepted immediately regardless of any
    accompanying thought text -- the malformed/citation checks only gate
    whether text-with-no-function_call is treated as a genuine Final Answer.

    `known_documents` is the set of document_names that have actually
    appeared in this run's Observations so far (via _SOURCE_TOOLS results) --
    a citation to anything outside that set is, by definition, not grounded
    in something the model actually retrieved.

    Returns (response, parsed) once a function_call or a genuine non-empty,
    non-malformed, fully-grounded answer is obtained, or (None, None) if the
    retry budget is exhausted first.
    """
    retry_count = 0
    while True:
        response = await llm_service.generate_with_tools(
            contents,
            tools=tool_registry.TOOL_DECLARATIONS,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        parsed = parse_agent_response(response)

        if parsed.function_call is not None:
            return response, parsed

        malformed = bool(parsed.thought_text) and is_malformed_response(parsed.thought_text)
        unverified_citations: set[str] = set()
        if parsed.thought_text and not malformed:
            unverified_citations = _extract_cited_document_names(parsed.thought_text) - known_documents
            if not unverified_citations:
                return response, parsed

        if malformed:
            reason = "Malformed output detected"
            nudge = _MALFORMED_ANSWER_NUDGE
        elif unverified_citations:
            reason = f"Citation to unverified document(s) detected: {', '.join(sorted(unverified_citations))}"
            nudge = _UNVERIFIED_CITATION_NUDGE
        else:
            reason = "Empty response received"
            nudge = _EMPTY_ANSWER_NUDGE

        if retry_count >= MAX_INVALID_RETRIES:
            steps.append(
                Step(
                    type="retry",
                    content=f"{reason}; retry budget ({MAX_INVALID_RETRIES}) exhausted.",
                )
            )
            return None, None

        retry_count += 1
        steps.append(
            Step(
                type="retry",
                content=f"{reason} -- retrying ({retry_count}/{MAX_INVALID_RETRIES}).",
            )
        )
        contents.append(response.candidates[0].content)
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=nudge)]))


def _build_initial_contents(question: str, conversation_history: list[dict]) -> list[types.Content]:
    contents = [
        types.Content(
            role="model" if message["role"] == "agent" else "user",
            parts=[types.Part.from_text(text=message["content"])],
        )
        for message in conversation_history
    ]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    return contents


def _extract_sources(observation: str) -> list[Source]:
    return [
        Source(
            document_name=match.group("document").strip(),
            section_title=match.group("section").strip(),
            excerpt=re.sub(r"\s+", " ", match.group("excerpt")).strip(),
        )
        for match in _RESULT_PATTERN.finditer(observation)
    ]
