"""Tool registration and dispatch.

Converts the Pydantic parameter models declared in app/agent/tools/ into
Gemini FunctionDeclaration objects (TOOL_DECLARATIONS) and executes the
corresponding async tool implementation by name (TOOL_DISPATCH / execute())
when the model issues a function call. This is the single place that knows
about all 5 tools — agent_loop.py only imports TOOL_DECLARATIONS and
execute() from here.
"""

from typing import Awaitable, Callable

from google.genai import types
from pydantic import BaseModel

from app.agent.tools.calculate_or_verify import CalculateOrVerifyParams, calculate_or_verify
from app.agent.tools.compare_sections import CompareSectionsParams, compare_sections
from app.agent.tools.get_document import GetDocumentParams, get_document
from app.agent.tools.list_documents import ListDocumentsParams, list_documents
from app.agent.tools.search_knowledge_base import SearchKnowledgeBaseParams, search_knowledge_base
from app.utils.logger import logger


def _pydantic_to_function_declaration(
    name: str, description: str, params_model: type[BaseModel]
) -> types.FunctionDeclaration:
    """Shared conversion logic: a Pydantic model's JSON schema is exactly
    the "parameters" object shape Gemini's FunctionDeclaration expects, so
    every tool reuses this instead of hand-writing its own schema.
    """
    schema = params_model.model_json_schema()
    schema.pop("title", None)
    for prop_schema in schema.get("properties", {}).values():
        prop_schema.pop("title", None)
    return types.FunctionDeclaration(
        name=name,
        description=description,
        parameters_json_schema=schema,
    )


# name -> (description, params model, async implementation)
# Descriptions are written for the model, not for humans reading this file —
# tool selection quality depends entirely on how clearly these describe
# *when* to use each tool.
_TOOL_SPECS: list[tuple[str, str, type[BaseModel], Callable[..., Awaitable[str]]]] = [
    (
        "search_knowledge_base",
        "Search the enterprise knowledge base for chunks of text relevant to a query. "
        "This is the primary tool for answering factual questions — use it first for any "
        "question about company policy, procedures, or documented rules. Optionally restrict "
        "to one category ('hr', 'legal', or 'ops') if you already know which area the answer "
        "lives in. If the result is 'No relevant chunks found.', try rephrasing the query, "
        "removing the category filter, or call list_documents to see what is available before "
        "giving up.",
        SearchKnowledgeBaseParams,
        search_knowledge_base,
    ),
    (
        "get_document",
        "Retrieve the full text of one specific document by its exact file name, ordered "
        "section by section. Use this when you need to read an entire document in depth — "
        "for example, after search_knowledge_base points you to a document and you need more "
        "surrounding context than a single chunk provides. Do not guess a file name; get it "
        "from a prior search_knowledge_base or list_documents result.",
        GetDocumentParams,
        get_document,
    ),
    (
        "list_documents",
        "List every document currently indexed in the knowledge base, with its category and "
        "chunk count. Use this to orient yourself when you don't know what documents exist, "
        "when a search_knowledge_base call returns no results and you want to check what is "
        "actually available, or before calling get_document or compare_sections if you are "
        "unsure of the exact file name.",
        ListDocumentsParams,
        list_documents,
    ),
    (
        "compare_sections",
        "Search two specific documents for the same topic and return the results side by "
        "side. Use this for multi-hop questions that ask you to compare, reconcile, or find "
        "differences between two named documents (e.g. 'do the leave policy and the "
        "employment contract agree on sick days?'). Requires the exact file names of both "
        "documents — use list_documents first if you don't already know them.",
        CompareSectionsParams,
        compare_sections,
    ),
    (
        "calculate_or_verify",
        "Safely evaluate a numeric math expression, e.g. to pro-rate a benefit or compute "
        "pay. Use this whenever a question requires arithmetic — never compute or estimate "
        "numbers yourself in text. Always retrieve the relevant formula or figures from the "
        "knowledge base with search_knowledge_base first, then pass the concrete expression "
        "here; do not invent numbers that were not found in a document.",
        CalculateOrVerifyParams,
        calculate_or_verify,
    ),
]

TOOL_DECLARATIONS: list[types.FunctionDeclaration] = [
    _pydantic_to_function_declaration(name, description, params_model)
    for name, description, params_model, _ in _TOOL_SPECS
]

TOOL_DISPATCH: dict[str, Callable[..., Awaitable[str]]] = {
    name: implementation for name, _, _, implementation in _TOOL_SPECS
}


async def execute(tool_name: str, params: dict) -> str:
    """Dispatch a model-issued function call to its implementation.

    Always returns a string observation — a failed tool call (unknown tool,
    bad params, downstream error) is information for the agent to reason
    about, not a crash, so every exception is caught and turned into an
    error observation instead of propagating.
    """
    implementation = TOOL_DISPATCH.get(tool_name)
    if implementation is None:
        return f"Error executing {tool_name}: unknown tool."
    try:
        return await implementation(**params)
    except Exception as exc:
        logger.warning("Tool '%s' raised during execution: %s", tool_name, exc)
        return f"Error executing {tool_name}: {exc}"
