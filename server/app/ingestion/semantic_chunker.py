"""Semantic chunking strategy (Section 6.2 of requirements.md).

Splits at natural section boundaries, not fixed character count:
  1. Try pattern set A (generic headings) first, one pattern at a time, in
     order: markdown #, ALL-CAPS lines, numbered headings ("2. Annual Leave
     Entitlement"), then blank-line paragraph breaks — using whichever
     pattern is the first to find >= 2 sections. Tried individually rather
     than combined, since real extracted PDF text (single-\n line wraps,
     numbered inline headings) otherwise gets over-fragmented by unrelated
     boundary types matching alongside the real one.
  2. If none of pattern set A's patterns detect >= 2 sections, retry with
     pattern set B (VN legal-style headings: "Điều", "Khoản", "Mục",
     combined — these commonly co-occur within the same document).
  3. If both fail, the whole text is treated as a single section.
  4. Any resulting section longer than MAX_CHUNK_CHARS is split further at
     sentence/line boundaries.
  5. Any resulting chunk shorter than MIN_CHUNK_CHARS is merged with its
     neighbor.

Returns a flat list of {"text": ..., "section_title": ...} dicts — chunk
metadata that depends on the surrounding document (chunk_index,
page_estimate, char_count, document_name, category) is attached later by
document_ingester.py, not here.
"""

import re

MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 150

# Pattern set A — generic headings (Section 6.2, pattern set A), tried one
# at a time in this order; each entry is a single-pattern group passed to
# _split_by_patterns.
_PATTERNS_A_SEQUENCE = [
    [re.compile(r"\n(#{1,3}\s+.+)")],                       # markdown headers, e.g. "## Annual Leave"
    [re.compile(r"\n([A-Z][A-Z ]{5,})\n")],                 # ALL-CAPS heading lines
    [re.compile(r"\n(\d{1,2}\.\s+[A-Z][^\n]{3,80})\n")],    # numbered headings, e.g. "2. Annual Leave Entitlement"
    [re.compile(r"\n\n+")],                                  # blank-line paragraph breaks
]

# Pattern set B — VN legal-style headings (Section 6.2, pattern set B),
# combined since these markers commonly co-occur within the same document.
_PATTERNS_B = [
    re.compile(r"\n(Điều\s+\d+[^\n]*)"),
    re.compile(r"\n(Khoản\s+\d+[^\n]*)"),
    re.compile(r"\n(Mục\s+\d+[^\n]*)"),
]

_SENTENCE_BOUNDARY = re.compile(r"(?<=\. )|\n+")


def chunk(text: str) -> list[dict]:
    sections = _detect_sections(text)
    expanded = _expand_oversized(sections)
    merged = _merge_undersized(expanded)
    return [{"text": body, "section_title": title} for title, body in merged]


def _detect_sections(text: str) -> list[tuple[str, str]]:
    for patterns in _PATTERNS_A_SEQUENCE:
        pieces = _split_by_patterns(text, patterns)
        if len(pieces) >= 2:
            return [_titled(title, body) for title, body in pieces]

    pieces = _split_by_patterns(text, _PATTERNS_B)
    if len(pieces) >= 2:
        return [_titled(title, body) for title, body in pieces]

    # Pattern set A (including its final blank-line stage) and pattern set B
    # both failed to find at least 2 sections — treat the whole text as one.
    return [_titled(None, text.strip())]


def _split_by_patterns(text: str, patterns: list[re.Pattern]) -> list[tuple[str | None, str]]:
    """Find every heading match for any pattern in `patterns`, then split
    `text` into (title, body) pieces at those boundaries. `title` is None
    for a boundary that carries no heading text (a plain blank-line break).
    """
    raw_matches = []
    for pattern in patterns:
        for m in pattern.finditer(text):
            title = m.group(1).strip() if m.groups() else None
            raw_matches.append((m.start(), m.end(), title))
    if not raw_matches:
        return [(None, text)]

    raw_matches.sort(key=lambda t: t[0])
    matches = []
    last_end = -1
    for start, end, title in raw_matches:
        if start < last_end:
            continue  # overlapping match from another pattern — skip
        matches.append((start, end, title))
        last_end = end

    pieces: list[tuple[str | None, str]] = []
    prev_end = 0
    prev_title: str | None = None
    for start, end, title in matches:
        body = text[prev_end:start].strip()
        if body:
            pieces.append((prev_title, body))
        prev_end = end
        prev_title = title
    tail = text[prev_end:].strip()
    if tail:
        pieces.append((prev_title, tail))
    return pieces


def _titled(title: str | None, body: str) -> tuple[str, str]:
    if title:
        return title, body
    first_line = body.split("\n", 1)[0].strip()
    return (first_line[:80] if first_line else "Untitled Section"), body


def _expand_oversized(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    expanded: list[tuple[str, str]] = []
    for title, body in sections:
        for part in _split_oversized(body):
            expanded.append((title, part))
    return expanded


def _split_oversized(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    # Splitting at "\n" boundaries discards the newline itself, and most of
    # our newlines are just mid-sentence line wraps rather than real
    # paragraph breaks — so units are rejoined with a single space to avoid
    # gluing words together (e.g. "Version 1.0" + "1. Purpose").
    units = [u.strip() for u in _SENTENCE_BOUNDARY.split(text) if u and u.strip()]
    chunks: list[str] = []
    current_units: list[str] = []
    current_len = 0
    for unit in units:
        added_len = len(unit) + (1 if current_units else 0)
        if current_units and current_len + added_len > MAX_CHUNK_CHARS:
            chunks.append(" ".join(current_units))
            current_units = [unit]
            current_len = len(unit)
        else:
            current_units.append(unit)
            current_len += added_len
    if current_units:
        chunks.append(" ".join(current_units))
    return chunks or [text]


def _merge_undersized(pieces: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not pieces:
        return pieces

    working = list(pieces)
    merged: list[tuple[str, str]] = []
    i = 0
    n = len(working)
    while i < n:
        title, text = working[i]
        if len(text) < MIN_CHUNK_CHARS and i + 1 < n:
            next_title, next_text = working[i + 1]
            working[i + 1] = (next_title, f"{text}\n{next_text}")
            i += 1
            continue
        merged.append((title, text))
        i += 1

    # The very last piece has no "next" to merge into — if it's still too
    # short, fold it into the previous emitted chunk instead.
    if len(merged) >= 2 and len(merged[-1][1]) < MIN_CHUNK_CHARS:
        prev_title, prev_text = merged[-2]
        _, last_text = merged[-1]
        merged[-2] = (prev_title, f"{prev_text}\n{last_text}")
        merged.pop()

    return merged
