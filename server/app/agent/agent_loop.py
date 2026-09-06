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

from google.genai import types

from app.agent import tool_registry
from app.config import settings
from app.models import AgentResult, Source, Step
from app.services import llm_service
from app.utils.parse_agent_response import parse_agent_response

SYSTEM_INSTRUCTION = """You are an AI Agent for enterprise knowledge retrieval at VinTech Corp.
Before each tool call, briefly state your reasoning as plain text (this is your "Thought").
Then call the appropriate function.
When you have enough information, respond with plain text only (no function call) —
this is your Final Answer. Always cite the source document and section name in the
Final Answer. If context is insufficient, say so — do not fabricate information.
For numeric questions, retrieve the formula first, then use calculate_or_verify."""

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


async def run_agent(question: str, conversation_history: list[dict]) -> AgentResult:
    contents = _build_initial_contents(question, conversation_history)
    steps: list[Step] = []
    sources: list[Source] = []
    seen_sources: set[tuple[str, str]] = set()

    for _ in range(settings.MAX_AGENT_STEPS):
        response = await llm_service.generate_with_tools(
            contents,
            tools=tool_registry.TOOL_DECLARATIONS,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        parsed = parse_agent_response(response)

        if parsed.thought_text:
            steps.append(Step(type="thought", content=parsed.thought_text))

        if parsed.function_call is None:
            # No function_call anywhere in the response -> the model chose
            # to answer directly. This is the model's own stopping
            # decision (Section 13.2), not a max-steps cutoff.
            return AgentResult(answer=parsed.thought_text or "", steps=steps, sources=sources)

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
