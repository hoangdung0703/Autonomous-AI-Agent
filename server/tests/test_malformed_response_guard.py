"""Unit tests for the malformed-response and citation-verification
heuristics in app/agent/agent_loop.py, exercised directly with crafted
strings -- no LLM call, no tool execution.

Note on the garbled-text fixture below: the exact one-off Gemini output that
originally surfaced this failure mode (see commit 7645102's message: "a
non-empty but garbled/hallucinated response... structurally echoing internal
tool-call/response formatting") was never persisted verbatim anywhere in
this repo or its history -- it was observed once during a live eval re-run
and not saved. The fixture here is a faithful reconstruction built from the
project's own documented description of it (a leading Hangul syllable
followed by a "response:"/"default_api:" echo of the internal function-call
wire format), which exercises the exact same two guard mechanisms
(_SUSPICIOUS_UNICODE_RANGES and _TOOL_ECHO_SUBSTRINGS) the real bug tripped
-- not a byte-for-byte replay of the original text.
"""

from app.agent.agent_loop import _extract_cited_document_names, is_malformed_response


def test_garbled_tool_echo_text_is_flagged_malformed():
    garbled = "쓸response:default_api:search_knowledge_base(query='overtime pay formula')"
    assert is_malformed_response(garbled) is True


def test_legitimate_final_answer_is_not_flagged_malformed():
    genuine = (
        "The standard hourly rate is calculated as monthly base salary divided by "
        "176 working hours, per benefits-guide.pdf, Section 3. Overtime Compensation."
    )
    assert is_malformed_response(genuine) is False


def test_empty_text_is_not_flagged_malformed():
    # is_malformed_response only judges non-empty text -- emptiness is a
    # separate failure mode (_EMPTY_ANSWER_NUDGE) handled elsewhere in
    # agent_loop.py, not by this heuristic.
    assert is_malformed_response("") is False


def test_tool_echo_markers_alone_without_suspicious_unicode_still_flagged():
    # All three of "document:"/"section:"/"score:" co-occurring is exactly
    # search_knowledge_base's own internal formatting leaking into a
    # "Final Answer" -- conservative, but a real answer never says "score:".
    echoed = "1. document: benefits-guide.pdf | section: 3. Overtime Compensation | score: 0.75"
    assert is_malformed_response(echoed) is True


def test_answer_mentioning_document_and_section_without_score_is_not_flagged():
    # Guards against being too aggressive: a real answer commonly says
    # "document" and "section" together in prose without leaking "score:".
    genuine = "See document benefits-guide.pdf, section 3 for the overtime formula."
    assert is_malformed_response(genuine) is False


def test_citation_to_known_document_is_verified():
    known_documents = {"benefits-guide.pdf", "leave-policy.pdf"}
    answer = "According to benefits-guide.pdf, overtime is 1.5x the standard hourly rate."

    cited = _extract_cited_document_names(answer)
    unverified = cited - known_documents

    assert cited == {"benefits-guide.pdf"}
    assert not unverified


def test_citation_to_unknown_document_is_flagged_unverified():
    # Reproduces the exact real regression this guard was added to catch
    # (q17: the agent cited a nonexistent "compensation-and-benefits.pdf").
    known_documents = {"benefits-guide.pdf", "leave-policy.pdf"}
    answer = (
        "According to the **compensation-and-benefits.pdf** (Section: *4. Overtime Pay "
        "Calculation*), the standard hourly rate is monthly base salary / 160 hours."
    )

    cited = _extract_cited_document_names(answer)
    unverified = cited - known_documents

    assert "compensation-and-benefits.pdf" in unverified
    assert "compensation-and-benefits.pdf" not in known_documents
