"""Unit tests for app/utils/parse_agent_response.py using hand-built fake
response objects that mirror the shape google-genai's
GenerateContentResponse actually has (response.candidates[0].content.parts,
each part with .text / .function_call) -- no real Gemini call, no SDK types
constructed, just plain objects with the same attribute shape the function
reads.
"""

from types import SimpleNamespace

from app.utils.parse_agent_response import parse_agent_response


def _make_part(text=None, function_call=None):
    return SimpleNamespace(text=text, function_call=function_call)


def _make_response(parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])


def test_function_call_part_parsed_as_action():
    fake_function_call = SimpleNamespace(name="search_knowledge_base", args={"query": "overtime"})
    response = _make_response([_make_part(text=None, function_call=fake_function_call)])

    parsed = parse_agent_response(response)

    assert parsed.function_call is fake_function_call
    assert parsed.thought_text is None


def test_function_call_part_with_accompanying_thought_text():
    # A real Gemini turn commonly carries both a Thought and a function_call
    # in the same response -- both must come through.
    fake_function_call = SimpleNamespace(name="calculate_or_verify", args={"expression": "1+1"})
    response = _make_response(
        [
            _make_part(text="I need to compute this.", function_call=None),
            _make_part(text=None, function_call=fake_function_call),
        ]
    )

    parsed = parse_agent_response(response)

    assert parsed.function_call is fake_function_call
    assert parsed.thought_text == "I need to compute this."


def test_text_only_part_parsed_as_final_answer():
    response = _make_response([_make_part(text="The overtime formula is base salary / 176.", function_call=None)])

    parsed = parse_agent_response(response)

    assert parsed.function_call is None
    assert parsed.thought_text == "The overtime formula is base salary / 176."


def test_whitespace_only_text_and_no_function_call_is_treated_as_empty():
    # Matches agent_loop.py's empty-answer handling: bool("") is False, so
    # this is treated the same as no answer at all, triggering the
    # _EMPTY_ANSWER_NUDGE retry path rather than being accepted as a genuine
    # Final Answer.
    response = _make_response([_make_part(text="   ", function_call=None)])

    parsed = parse_agent_response(response)

    assert parsed.function_call is None
    assert not parsed.thought_text  # "" after strip(), not None -- still falsy either way


def test_no_parts_at_all_is_treated_as_empty():
    response = _make_response([])

    parsed = parse_agent_response(response)

    assert parsed.function_call is None
    assert parsed.thought_text is None
