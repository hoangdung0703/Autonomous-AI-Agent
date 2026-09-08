"""Automated evaluation script (Section 15.2 of requirements.md).

For each query in test-set.json, runs both the ReAct agent (agent_loop.run_agent)
and the non-agentic baseline (baseline_rag.query), scores correctness and
faithfulness via LLM-as-judge (Gemini, through llm_service.generate_simple —
never a direct API call, so the hard-capped thinking_level="low" config in
llm_service.py applies to every judge call automatically), scores tool
selection accuracy for the agent, writes full per-query results to
results.json, and prints the Section 15.3 aggregate results table.

Run from server/ so app.config's relative ".env" resolves correctly:

    cd server
    ..\venv\Scripts\python.exe ..\evaluation\run_eval.py      (or activate venv first)
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))

from app.agent.agent_loop import run_agent  # noqa: E402
from app.services import llm_service  # noqa: E402
from app.utils.logger import logger  # noqa: E402

import baseline_rag  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TEST_SET_PATH = EVAL_DIR / "test-set.json"
RESULTS_PATH = EVAL_DIR / "results.json"

CORRECTNESS_PROMPT_TMPL = """You are evaluating an AI system's answer.

Question: {question}
Ground truth answer: {ground_truth}
System answer: {system_answer}

Does the system answer correctly address the question and contain the key facts from the ground truth?
Respond with JSON: {{ "correct": true/false, "reason": "brief explanation" }}"""

FAITHFULNESS_PROMPT_TMPL = """You are evaluating whether an AI answer is grounded in its source documents.

Question: {question}
Answer: {answer}
Source chunks used: {retrieved_chunks}

Is every factual claim in the answer supported by the source chunks?
Facts, numbers, or premises that are stated directly in the Question itself (e.g. figures,
names, or conditions the user supplied as part of asking the question) are NOT unsupported
claims, even if they don't appear in the source chunks -- they are given inputs, not
retrieved or fabricated facts. Only flag a claim as unsupported if it is NOT traceable to
either (a) a premise stated in the Question, or (b) the source chunks.
Respond with JSON: {{ "faithful": true/false, "unsupported_claims": [...] }}"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)
_BRACES_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(raw: str) -> Optional[dict]:
    """Defensively parse a judge response as JSON, stripping markdown code
    fences first (Gemini frequently wraps JSON in ```json ... ``` even when
    asked for raw JSON)."""
    text = (raw or "").strip()
    fence_match = _FENCE_RE.match(text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    braces_match = _BRACES_RE.search(text)
    if braces_match:
        try:
            return json.loads(braces_match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _format_sources_for_prompt(sources: list) -> str:
    if not sources:
        return "(no sources retrieved)"
    return "\n\n".join(
        f"[{s.document_name} / {s.section_title}]\n{s.excerpt}" for s in sources
    )


def _format_chunks_for_prompt(chunks: list[dict]) -> str:
    if not chunks:
        return "(no chunks retrieved)"
    return "\n\n".join(
        f"[{c.get('document_name')} / {c.get('section_title')}]\n{(c.get('text') or '')[:400]}"
        for c in chunks
    )


async def score_correctness(question: str, ground_truth: str, system_answer: str) -> dict:
    prompt = CORRECTNESS_PROMPT_TMPL.format(
        question=question, ground_truth=ground_truth, system_answer=system_answer
    )
    raw = await llm_service.generate_simple(prompt)
    parsed = _parse_judge_json(raw)
    if parsed is None or "correct" not in parsed:
        return {"correct": False, "reason": f"JUDGE_PARSE_ERROR: {raw[:300]!r}"}
    return {"correct": bool(parsed["correct"]), "reason": parsed.get("reason", "")}


async def score_faithfulness(question: str, answer: str, source_chunks_text: str) -> dict:
    prompt = FAITHFULNESS_PROMPT_TMPL.format(
        question=question, answer=answer, retrieved_chunks=source_chunks_text
    )
    raw = await llm_service.generate_simple(prompt)
    parsed = _parse_judge_json(raw)
    if parsed is None or "faithful" not in parsed:
        return {
            "faithful": False,
            "unsupported_claims": [],
            "reason": f"JUDGE_PARSE_ERROR: {raw[:300]!r}",
        }
    return {
        "faithful": bool(parsed["faithful"]),
        "unsupported_claims": parsed.get("unsupported_claims", []),
    }


def score_tool_selection(actual_tools: list[str], expected_tools: list[str]) -> bool:
    """Match = every expected tool was used at least once (order-independent
    subset check), per Section 13.5's "used the expected tools" wording --
    not strict set equality. This tolerates legitimate extra orientation
    calls (e.g. list_documents before compare_sections to confirm exact file
    names, which tool_registry's own compare_sections description
    recommends) without penalizing them as wrong tool selection."""
    return set(expected_tools).issubset(set(actual_tools))


async def run_one_query(index: int, total: int, item: dict) -> dict:
    qid, question, qtype = item["id"], item["question"], item["type"]
    ground_truth = item["ground_truth"]
    expected_tools = item["expected_tools"]

    logger.info("[%d/%d] %s (%s): %s", index, total, qid, qtype, question)

    logger.info("    running agent...")
    t0 = time.monotonic()
    agent_result = await run_agent(question, conversation_history=[])
    agent_elapsed = time.monotonic() - t0

    agent_tools_used = [s.tool for s in agent_result.steps if s.type == "action" and s.tool]
    agent_num_reasoning_steps = sum(1 for s in agent_result.steps if s.type == "action")
    agent_steps_full = [s.model_dump() for s in agent_result.steps]
    agent_sources = [s.model_dump() for s in agent_result.sources]
    logger.info(
        "    agent done in %.1fs -- %d step(s), tools: %s",
        agent_elapsed, agent_num_reasoning_steps, agent_tools_used,
    )

    logger.info("    running baseline...")
    baseline_result = await baseline_rag.query(question)

    logger.info("    judging correctness (agent, baseline)...")
    agent_correctness = await score_correctness(question, ground_truth, agent_result.answer)
    baseline_correctness = await score_correctness(question, ground_truth, baseline_result["answer"])

    logger.info("    judging faithfulness (agent, baseline)...")
    agent_faithfulness = await score_faithfulness(
        question, agent_result.answer, _format_sources_for_prompt(agent_result.sources)
    )
    baseline_faithfulness = await score_faithfulness(
        question, baseline_result["answer"], _format_chunks_for_prompt(baseline_result["retrieved_chunks"])
    )

    tool_selection_match = score_tool_selection(agent_tools_used, expected_tools)

    return {
        "id": qid,
        "type": qtype,
        "question": question,
        "ground_truth": ground_truth,
        "expected_source_documents": item["expected_source_documents"],
        "expected_tools": expected_tools,
        "agent": {
            "answer": agent_result.answer,
            "steps": agent_steps_full,
            "num_reasoning_steps": agent_num_reasoning_steps,
            "tools_used": agent_tools_used,
            "sources": agent_sources,
            "source_documents_used": sorted({s["document_name"] for s in agent_sources}),
            "correctness": agent_correctness,
            "faithfulness": agent_faithfulness,
            "tool_selection_match": tool_selection_match,
        },
        "baseline": {
            "answer": baseline_result["answer"],
            "steps": baseline_result["steps"],
            "tools_used": baseline_result["tools_used"],
            "retrieved_chunks": [
                {
                    "document_name": c.get("document_name"),
                    "section_title": c.get("section_title"),
                    "similarity_score": c.get("similarity_score"),
                }
                for c in baseline_result["retrieved_chunks"]
            ],
            "correctness": baseline_correctness,
            "faithfulness": baseline_faithfulness,
        },
    }


def _pct(n: int, d: int) -> float:
    return (n / d * 100.0) if d else 0.0


def _print_aggregate_table(results: list[dict]) -> None:
    total = len(results)
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    def completion_rate(rs: list[dict], system: str) -> float:
        return _pct(sum(1 for r in rs if r[system]["correctness"]["correct"]), len(rs))

    def faithfulness_rate(rs: list[dict], system: str) -> float:
        return _pct(sum(1 for r in rs if r[system]["faithfulness"]["faithful"]), len(rs))

    baseline_overall = completion_rate(results, "baseline")
    agent_overall = completion_rate(results, "agent")

    baseline_faith = faithfulness_rate(results, "baseline")
    agent_faith = faithfulness_rate(results, "agent")

    avg_agent_steps = (
        sum(r["agent"]["num_reasoning_steps"] for r in results) / total if total else 0.0
    )

    print()
    print("=" * 100)
    print("SECTION 15.3 -- AGGREGATE RESULTS TABLE")
    print("=" * 100)
    print(f"{'Metric':<32} | {'Baseline RAG':>14} | {'ReAct Agent':>14} | {'Improvement':>12}")
    print("-" * 100)
    print(
        f"{'Task Completion Rate (all)':<32} | {baseline_overall:>13.1f}% | {agent_overall:>13.1f}% | "
        f"{agent_overall - baseline_overall:>+11.1f}%"
    )
    for qtype, label in (("simple", "Simple"), ("multi_hop", "Multi-hop"), ("computation", "Computation")):
        rs = by_type.get(qtype, [])
        b = completion_rate(rs, "baseline")
        a = completion_rate(rs, "agent")
        print(
            f"{'Task Completion -- ' + label:<32} | {b:>13.1f}% | {a:>13.1f}% | {a - b:>+11.1f}%"
        )
    print(
        f"{'Answer Faithfulness':<32} | {baseline_faith:>13.1f}% | {agent_faith:>13.1f}% | "
        f"{agent_faith - baseline_faith:>+11.1f}%"
    )
    print(f"{'Avg Reasoning Steps':<32} | {1.0:>14.1f} | {avg_agent_steps:>14.2f} | {'--':>12}")
    print("-" * 100)

    tool_matches = sum(1 for r in results if r["agent"]["tool_selection_match"])
    print(f"Tool Selection Accuracy (agent): {tool_matches}/{total} ({_pct(tool_matches, total):.1f}%)")
    print(f"N = {total} queries "
          f"(simple={len(by_type.get('simple', []))}, "
          f"multi_hop={len(by_type.get('multi_hop', []))}, "
          f"computation={len(by_type.get('computation', []))})")
    print("=" * 100)
    sys.stdout.flush()


def _is_complete(entry: Optional[dict]) -> bool:
    """A prior result is reusable only if it finished cleanly -- no top-level
    error, and both systems' correctness/faithfulness verdicts are present.
    A stub error-entry from a crashed run (see the except branch below) must
    NOT be treated as complete, so it gets retried on the next run."""
    if not entry or "error" in entry:
        return False
    agent = entry.get("agent") or {}
    baseline = entry.get("baseline") or {}
    return (
        "correctness" in agent
        and "faithfulness" in agent
        and "correctness" in baseline
        and "faithfulness" in baseline
    )


async def main() -> None:
    test_set = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    total = len(test_set)

    existing_by_id: dict[str, dict] = {}
    if RESULTS_PATH.exists():
        try:
            for entry in json.loads(RESULTS_PATH.read_text(encoding="utf-8")):
                if isinstance(entry, dict) and "id" in entry:
                    existing_by_id[entry["id"]] = entry
        except json.JSONDecodeError:
            logger.warning("Could not parse existing %s -- starting fresh.", RESULTS_PATH)

    reusable = sum(1 for item in test_set if _is_complete(existing_by_id.get(item["id"])))
    if reusable:
        logger.info(
            "Resuming: %d/%d queries already complete in %s and will be skipped.",
            reusable, total, RESULTS_PATH,
        )

    results: list[dict] = []
    for i, item in enumerate(test_set, start=1):
        qid = item["id"]
        prior = existing_by_id.get(qid)
        if _is_complete(prior):
            logger.info("[%d/%d] %s (%s): SKIPPED -- reusing complete result from a prior run.",
                        i, total, qid, item["type"])
            result = prior
        else:
            try:
                result = await run_one_query(i, total, item)
            except Exception as exc:  # noqa: BLE001
                logger.error("    ERROR on %s: %s", qid, exc)
                result = {
                    "id": qid,
                    "type": item["type"],
                    "question": item["question"],
                    "error": str(exc),
                    "agent": {"correctness": {"correct": False}, "faithfulness": {"faithful": False},
                              "tool_selection_match": False, "num_reasoning_steps": 0},
                    "baseline": {"correctness": {"correct": False}, "faithfulness": {"faithful": False}},
                }
        results.append(result)
        # Write incrementally so a late failure doesn't lose earlier progress,
        # and so a subsequent run can resume from here.
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Wrote %d results to %s", len(results), RESULTS_PATH)
    _print_aggregate_table(results)


if __name__ == "__main__":
    asyncio.run(main())
