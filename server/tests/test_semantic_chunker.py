"""Unit tests for app/ingestion/semantic_chunker.py's chunk() -- pure string
logic, no PDF/OCR/embedding involved. Body lengths are deliberately chosen
relative to MIN_CHUNK_CHARS (150) / MAX_CHUNK_CHARS (1200) so each test
exercises exactly the boundary it names, without an unrelated merge/split
kicking in and confusing the assertion.
"""

from app.ingestion.semantic_chunker import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, chunk

_LONG_ENOUGH = (
    "This paragraph is deliberately padded with filler words so that its "
    "length comfortably clears the minimum chunk size threshold used by the "
    "chunker for merge decisions, well past one hundred fifty characters."
)
assert len(_LONG_ENOUGH) >= MIN_CHUNK_CHARS  # sanity-check the fixture itself


def test_markdown_headings_split_into_sections():
    text = (
        f"\n## Introduction\n{_LONG_ENOUGH} intro specific marker.\n"
        f"\n## Overtime Policy\n{_LONG_ENOUGH} overtime specific marker.\n"
    )
    result = chunk(text)

    assert len(result) == 2
    assert result[0]["section_title"] == "## Introduction"
    assert "intro specific marker" in result[0]["text"]
    assert result[1]["section_title"] == "## Overtime Policy"
    assert "overtime specific marker" in result[1]["text"]


def test_numbered_headings_split_into_sections():
    text = (
        f"\n1. Purpose\n{_LONG_ENOUGH} purpose specific marker.\n"
        f"\n2. Scope\n{_LONG_ENOUGH} scope specific marker.\n"
    )
    result = chunk(text)

    assert len(result) == 2
    assert result[0]["section_title"] == "1. Purpose"
    assert "purpose specific marker" in result[0]["text"]
    assert result[1]["section_title"] == "2. Scope"
    assert "scope specific marker" in result[1]["text"]


def test_no_headings_falls_back_to_paragraph_splitting():
    # No markdown/ALL-CAPS/numbered heading anywhere -- pattern set A only
    # finds a split at its final blank-line stage, which carries no heading
    # text, so _titled() falls back to using each paragraph's first line.
    para_one = f"First paragraph marker sentence. {_LONG_ENOUGH}"
    para_two = f"Second paragraph marker sentence. {_LONG_ENOUGH}"
    text = f"{para_one}\n\n{para_two}"

    result = chunk(text)

    assert len(result) == 2
    assert "First paragraph marker sentence" in result[0]["text"]
    assert "Second paragraph marker sentence" in result[1]["text"]
    # Fallback title is derived from each section's own first line.
    assert result[0]["section_title"].startswith("First paragraph marker sentence")
    assert result[1]["section_title"].startswith("Second paragraph marker sentence")


def test_oversized_section_is_split_at_sentence_boundaries():
    # One sentence repeated enough times to comfortably exceed
    # MAX_CHUNK_CHARS, with no headings/blank lines anywhere -- forces the
    # single-section fallback in _detect_sections, then _expand_oversized
    # must split it further.
    sentence = "This is a sentence about the overtime policy for testing purposes. "
    text = sentence * 40
    assert len(text) > MAX_CHUNK_CHARS * 2  # fixture is big enough to force >= 2 splits

    result = chunk(text)

    assert len(result) >= 2
    for section in result:
        assert len(section["text"]) <= MAX_CHUNK_CHARS
        # No chunk should have collapsed below the merge threshold either --
        # if it had, _merge_undersized would have folded it into a neighbor.
        assert len(section["text"]) >= MIN_CHUNK_CHARS
    # No sentence content should have been silently dropped in the split.
    rejoined = " ".join(section["text"] for section in result)
    assert rejoined.count("overtime policy for testing purposes") == 40


def test_undersized_section_is_merged_with_next():
    short_body = "Short intro."
    assert len(short_body) < MIN_CHUNK_CHARS
    text = f"\n1. Purpose\n{short_body}\n\n2. Scope\n{_LONG_ENOUGH} scope specific marker.\n"

    result = chunk(text)

    # The undersized "1. Purpose" section has no choice but to merge forward
    # into "2. Scope" -- there is nothing before it to merge into instead.
    assert len(result) == 1
    assert result[0]["section_title"] == "2. Scope"
    assert "Short intro." in result[0]["text"]
    assert "scope specific marker" in result[0]["text"]


def test_empty_string_does_not_crash():
    result = chunk("")

    assert isinstance(result, list)
    for section in result:
        assert "text" in section and "section_title" in section
    # No real content should appear out of nowhere.
    assert "".join(section["text"] for section in result) == ""
