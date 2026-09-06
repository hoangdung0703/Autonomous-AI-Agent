"""Parse Thought/Action from a Gemini function-calling response.

Reads response.candidates[0].content.parts directly (Section 13.1 of
requirements.md) — never free-text regex parsing of the model's reply, only
a structural read of the SDK's own response object:
  - any part with `.text` is the model's "Thought" (or, when no part in the
    response carries a function_call, this same text is the Final Answer —
    that decision belongs to the caller, agent_loop.py).
  - a part with `.function_call` is the tool the model decided to call.
"""

from dataclasses import dataclass
from typing import Optional

from google.genai import types


@dataclass
class ParsedAgentResponse:
    thought_text: Optional[str]
    function_call: Optional[types.FunctionCall]


def parse_agent_response(response: types.GenerateContentResponse) -> ParsedAgentResponse:
    parts = response.candidates[0].content.parts or []

    text_parts = [part.text for part in parts if part.text]
    function_call = next((part.function_call for part in parts if part.function_call), None)

    thought_text = "\n".join(text_parts).strip() if text_parts else None
    return ParsedAgentResponse(thought_text=thought_text, function_call=function_call)
