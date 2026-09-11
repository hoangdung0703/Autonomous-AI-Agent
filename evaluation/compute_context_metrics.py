"""RAGAS-style Context Precision / Context Recall, computed retroactively
from already-stored evaluation traces -- read-only analysis, no new agent or
baseline runs.

Answer Faithfulness (run_eval.py) checks whether the *answer* contradicts
what was retrieved; Task Completion checks whether the *answer* matches
ground truth. Neither isolates the *retriever* on its own -- a faithful,
correct-as-far-as-it-goes answer (e.g. q15) can still come from retrieval
that never surfaced a needed document. Context Precision/Recall are scored
purely from the retrieved chunks and the ground truth, without looking at
the generated answer at all, specifically to isolate that failure mode.

Chunk source: the agent's stored `agent.sources` list in each results file
(document_name/section_title/excerpt) is already exactly "every distinct
chunk excerpt that appeared across all search_knowledge_base/compare_sections
Observation steps in this run" -- it's produced by agent_loop.py's own
_extract_sources()/seen_sources dedup during the original run, so this
script reuses that stored field directly rather than re-parsing raw step
text with a second, possibly-divergent regex.

Baseline: results*.json's baseline.retrieved_chunks entries only ever stored
document_name/section_title/similarity_score (see run_eval.py's
run_one_query) -- never the chunk excerpt/text -- so there is no stored text
to show a judge. Context metrics are therefore computed for the AGENT only;
this is reported explicitly below rather than silently omitted or faked.

Run from server/, same convention as the other evaluation scripts:

    cd server
    ..\\venv\\Scripts\\python.exe ..\\evaluation\\compute_context_metrics.py
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

from app.services import llm_service  # noqa: E402
from app.utils.logger import logger  # noqa: E402

import run_eval  # noqa: E402 -- reuses _parse_judge_json unmodified

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = EVAL_DIR / "context-metrics.json"

# (test-set path, results path, a label identifying which pipeline this is)
SOURCES = [
    ("test-set.json", "results.json", "main"),
    ("test-set-distractor.json", "results-distractor.json", "distractor"),
    ("test-set-heldout.json", "results-heldout.json", "heldout"),
]

CONTEXT_PRECISION_PROMPT_TMPL = """You are evaluating the retrieval quality of a RAG system's retriever, \
independent of how any final answer was worded.

Question: {question}

Retrieved chunks (each shown once, numbered):
{numbered_chunks}

For each numbered chunk, decide whether it is relevant to answering the question above -- i.e. \
whether it contains information that would help construct a correct, complete answer. A chunk on a \
related topic that does not actually help answer this specific question is NOT relevant.

Respond with JSON: {{ "relevant_indices": [list of the integer numbers of the relevant chunks, e.g. \
[1, 3]], "reason": "brief explanation" }}"""

CONTEXT_RECALL_PROMPT_TMPL = """You are evaluating the retrieval quality of a RAG system's retriever, \
independent of how any final answer was worded.

Question: {question}
Ground truth answer: {ground_truth}

Combined retrieved context (every distinct chunk the system retrieved, concatenated):
{combined_context}

Ground truth answers -- especially for multi-hop/comparison questions -- often contain several \
distinct key facts. Identify the distinct key facts in the ground truth answer, then estimate what \
PROPORTION of them are actually supported by the retrieved context above (stated or directly \
derivable from it, not from general knowledge) -- not just whether the general topic was retrieved.

Respond with JSON: {{ "key_facts_total": <int>, "key_facts_supported": <int>, "recall": <float \
between 0.0 and 1.0, key_facts_supported / key_facts_total>, "reason": "brief explanation" }}"""


def _format_numbered_chunks(chunks: list[str]) -> str:
    return "\n".join(f"{i}. {chunk}" for i, chunk in enumerate(chunks, start=1))


async def score_context_precision(question: str, chunks: list[str]) -> dict:
    """precision = (# chunks the judge marked relevant) / (# chunks retrieved).
    Undefined (None) if nothing was retrieved at all -- not scored as 0,
    since that would conflate "retrieved nothing useful" with "retrieved
    nothing.\""""
    if not chunks:
        return {"precision": None, "relevant_indices": [], "num_total": 0, "reason": "No chunks retrieved."}

    prompt = CONTEXT_PRECISION_PROMPT_TMPL.format(
        question=question, numbered_chunks=_format_numbered_chunks(chunks)
    )
    raw = await llm_service.generate_simple(prompt)
    parsed = run_eval._parse_judge_json(raw)
    if parsed is None or "relevant_indices" not in parsed:
        return {
            "precision": None,
            "relevant_indices": [],
            "num_total": len(chunks),
            "reason": f"JUDGE_PARSE_ERROR: {raw[:300]!r}",
        }

    valid_indices = {
        i for i in parsed["relevant_indices"] if isinstance(i, int) and 1 <= i <= len(chunks)
    }
    return {
        "precision": len(valid_indices) / len(chunks),
        "relevant_indices": sorted(valid_indices),
        "num_total": len(chunks),
        "reason": parsed.get("reason", ""),
    }


async def score_context_recall(question: str, ground_truth: str, chunks: list[str]) -> dict:
    """recall = judge's graded estimate of what proportion of the ground
    truth's key facts are covered by the combined retrieved context."""
    if not chunks:
        return {"recall": 0.0, "reason": "No chunks retrieved."}

    combined_context = "\n\n".join(chunks)
    prompt = CONTEXT_RECALL_PROMPT_TMPL.format(
        question=question, ground_truth=ground_truth, combined_context=combined_context
    )
    raw = await llm_service.generate_simple(prompt)
    parsed = run_eval._parse_judge_json(raw)
    if parsed is None:
        return {"recall": None, "reason": f"JUDGE_PARSE_ERROR: {raw[:300]!r}"}

    recall = parsed.get("recall")
    if not isinstance(recall, (int, float)):
        total = parsed.get("key_facts_total")
        supported = parsed.get("key_facts_supported")
        if isinstance(total, (int, float)) and total > 0 and isinstance(supported, (int, float)):
            recall = supported / total
        else:
            return {"recall": None, "reason": f"JUDGE_PARSE_ERROR (no usable recall field): {raw[:300]!r}"}

    recall = max(0.0, min(1.0, float(recall)))
    return {
        "recall": recall,
        "key_facts_total": parsed.get("key_facts_total"),
        "key_facts_supported": parsed.get("key_facts_supported"),
        "reason": parsed.get("reason", ""),
    }


def _load_query_records() -> list[dict]:
    """One record per query across all three (test-set, results) pairs --
    ground_truth from the test set, retrieved-chunk excerpts from the
    stored agent.sources in the matching results file."""
    records = []
    for test_set_name, results_name, source_label in SOURCES:
        test_set = json.loads((EVAL_DIR / test_set_name).read_text(encoding="utf-8"))
        results = json.loads((EVAL_DIR / results_name).read_text(encoding="utf-8"))
        results_by_id = {r["id"]: r for r in results if isinstance(r, dict) and "id" in r}

        for item in test_set:
            qid = item["id"]
            result = results_by_id.get(qid)
            if result is None or "agent" not in result or "sources" not in result["agent"]:
                logger.warning("Skipping %s (%s) -- no stored agent trace found in %s.",
                                qid, source_label, results_name)
                continue
            chunks = [s["excerpt"] for s in result["agent"]["sources"]]
            records.append({
                "id": qid,
                "type": item["type"],
                "source_file": source_label,
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "chunks": chunks,
            })
    return records


async def _score_one(record: dict) -> dict:
    precision_result = await score_context_precision(record["question"], record["chunks"])
    recall_result = await score_context_recall(record["question"], record["ground_truth"], record["chunks"])
    return {
        "id": record["id"],
        "type": record["type"],
        "source_file": record["source_file"],
        "num_chunks_retrieved": len(record["chunks"]),
        "context_precision": precision_result["precision"],
        "context_precision_detail": precision_result,
        "context_recall": recall_result["recall"],
        "context_recall_detail": recall_result,
    }


def _avg(values: list[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _print_aggregate(results: list[dict]) -> None:
    def fmt(x):
        return f"{x:.3f}" if x is not None else "n/a"

    print()
    print("=" * 100)
    print("CONTEXT PRECISION / CONTEXT RECALL -- AGGREGATE (agent retrieval only)")
    print("=" * 100)
    overall_p = _avg([r["context_precision"] for r in results])
    overall_r = _avg([r["context_recall"] for r in results])
    print(f"{'Overall (N=' + str(len(results)) + ')':<28} | precision={fmt(overall_p)} | recall={fmt(overall_r)}")
    print("-" * 100)

    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)
    for qtype in sorted(by_type):
        rs = by_type[qtype]
        p = _avg([r["context_precision"] for r in rs])
        rec = _avg([r["context_recall"] for r in rs])
        print(f"{qtype + ' (N=' + str(len(rs)) + ')':<28} | precision={fmt(p)} | recall={fmt(rec)}")
    print("=" * 100)


def _is_complete(entry: Optional[dict]) -> bool:
    """A prior result is reusable only if both judge calls actually
    succeeded (no top-level error, no JUDGE_PARSE_ERROR on either metric) --
    same resumability discipline as run_eval.py's _is_complete, since this
    script hits the same shared per-day Gemini quota and a run can be
    interrupted mid-way through no fault of the query itself."""
    if not entry or "error" in entry:
        return False
    if entry.get("context_precision") is None and entry.get("num_chunks_retrieved", 0) > 0:
        return False
    if entry.get("context_recall") is None:
        return False
    return True


async def main() -> None:
    records = _load_query_records()
    total = len(records)

    existing_by_id: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        try:
            for entry in json.loads(OUTPUT_PATH.read_text(encoding="utf-8")):
                if isinstance(entry, dict) and "id" in entry:
                    existing_by_id[entry["id"]] = entry
        except json.JSONDecodeError:
            logger.warning("Could not parse existing %s -- starting fresh.", OUTPUT_PATH)

    reusable = sum(1 for r in records if _is_complete(existing_by_id.get(r["id"])))
    if reusable:
        logger.info("Resuming: %d/%d queries already complete in %s and will be skipped.",
                     reusable, total, OUTPUT_PATH)

    logger.info("Scoring context precision/recall for %d queries (agent retrieval only)...", total)

    results: list[dict] = []
    for i, record in enumerate(records, start=1):
        prior = existing_by_id.get(record["id"])
        if _is_complete(prior):
            logger.info("[%d/%d] %s (%s/%s): SKIPPED -- reusing complete result.",
                        i, total, record["id"], record["source_file"], record["type"])
            result = prior
        else:
            logger.info("[%d/%d] %s (%s/%s): %s", i, total, record["id"], record["source_file"],
                        record["type"], record["question"])
            try:
                result = await _score_one(record)
                logger.info("    precision=%s recall=%s (n_chunks=%d)",
                            result["context_precision"], result["context_recall"],
                            result["num_chunks_retrieved"])
            except Exception as exc:  # noqa: BLE001
                logger.error("    ERROR on %s: %s", record["id"], exc)
                result = {
                    "id": record["id"],
                    "type": record["type"],
                    "source_file": record["source_file"],
                    "num_chunks_retrieved": len(record["chunks"]),
                    "context_precision": None,
                    "context_recall": None,
                    "error": str(exc),
                }
        results.append(result)
        # Incremental write, same discipline as run_eval.py, so a partial
        # run (e.g. hitting the daily quota) doesn't lose earlier progress.
        OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Wrote %d results to %s", len(results), OUTPUT_PATH)
    _print_aggregate(results)

    q15 = next((r for r in results if r["id"] == "q15"), None)
    if q15:
        overall_p = _avg([r["context_precision"] for r in results])
        overall_r = _avg([r["context_recall"] for r in results])
        print()
        print("=" * 100)
        print("q15 -- INDIVIDUAL SCORES")
        print("=" * 100)
        print(f"context_precision = {q15['context_precision']} (overall avg = {overall_p:.3f})")
        print(f"context_recall    = {q15['context_recall']} (overall avg = {overall_r:.3f})")
        print(f"num_chunks_retrieved = {q15['num_chunks_retrieved']}")
        print(f"recall reason: {q15.get('context_recall_detail', {}).get('reason', '')}")
        print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
