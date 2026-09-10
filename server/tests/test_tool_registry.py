"""Unit tests for app/agent/tool_registry.py's execute() dispatch and error
handling -- no real tool implementation is invoked (search_knowledge_base
etc. would need live Qdrant/Gemini), only its dispatch/error-catching
contract: execute() must always return a string observation, never raise.
"""

from app.agent import tool_registry


async def test_execute_unknown_tool_returns_error_string_without_raising():
    result = await tool_registry.execute("this_tool_does_not_exist", {})

    assert result == "Error executing this_tool_does_not_exist: unknown tool."


async def test_execute_catches_exception_from_tool_and_returns_observation(monkeypatch):
    async def _boom(**kwargs):
        raise RuntimeError("simulated downstream failure")

    monkeypatch.setitem(tool_registry.TOOL_DISPATCH, "boom_tool", _boom)

    result = await tool_registry.execute("boom_tool", {"anything": "ignored"})

    assert isinstance(result, str)
    assert result == "Error executing boom_tool: simulated downstream failure"


async def test_execute_passes_params_through_to_the_real_implementation(monkeypatch):
    captured = {}

    async def _fake_tool(expression, context):
        captured["expression"] = expression
        captured["context"] = context
        return f"ok:{expression}:{context}"

    monkeypatch.setitem(tool_registry.TOOL_DISPATCH, "fake_tool", _fake_tool)

    result = await tool_registry.execute("fake_tool", {"expression": "1+1", "context": "test"})

    assert result == "ok:1+1:test"
    assert captured == {"expression": "1+1", "context": "test"}


async def test_execute_catches_bad_params_as_a_typeerror_observation(monkeypatch):
    # A model-issued call with a missing/misnamed argument is a
    # TypeError at the Python call boundary -- execute() must catch this
    # exactly like any other tool-raised exception, not let it propagate.
    async def _needs_arg(required_param):
        return "unreachable"

    monkeypatch.setitem(tool_registry.TOOL_DISPATCH, "strict_tool", _needs_arg)

    result = await tool_registry.execute("strict_tool", {"wrong_param": "value"})

    assert result.startswith("Error executing strict_tool:")


def test_all_five_real_tools_are_registered():
    expected = {
        "search_knowledge_base",
        "get_document",
        "list_documents",
        "compare_sections",
        "calculate_or_verify",
    }
    assert expected.issubset(tool_registry.TOOL_DISPATCH.keys())
