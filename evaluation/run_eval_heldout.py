"""Held-out final validation run (10 queries never seen during any prior
debugging/fix cycle) -- Section 15.2 pipeline, plus wall-clock latency.

Deliberately does NOT import run_eval.run_one_query or modify run_eval.py at
all: this is a fully standalone script that reuses run_eval.py's scoring
building blocks (score_correctness, score_faithfulness, score_tool_selection,
the prompt-formatting helpers, _print_aggregate_table) completely
unmodified, so there is zero risk of this held-out run affecting the
existing main/distractor pipelines' scoring logic. The only addition beyond
what run_one_query already does is a wall-clock time.monotonic() measurement
around each system's call, stored as "latency_seconds" in the result --
purely additive, not a new scoring dimension.

Run from server/, same as run_eval.py:

    cd server
    ..\\venv\\Scripts\\python.exe ..\\evaluation\\run_eval_heldout.py
"""

import asyncio
import json
import time
from pathlib import Path

import run_eval  # noqa: E402
from app.agent.agent_loop import run_agent  # noqa: E402
from app.utils.logger import logger  # noqa: E402

import baseline_rag  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TEST_SET_PATH = EVAL_DIR / "test-set-heldout.json"
RESULTS_PATH = EVAL_DIR / "results-heldout.json"


async def run_one_heldout_query(index: int, total: int, item: dict) -> dict:
    qid, question, qtype = item["id"], item["question"], item["type"]
    ground_truth = item["ground_truth"]
    expected_tools = item["expected_tools"]

    logger.info("[%d/%d] %s (%s): %s", index, total, qid, qtype, question)

    logger.info("    running agent...")
    t0 = time.monotonic()
    agent_result = await run_agent(question, conversation_history=[])
    agent_latency = time.monotonic() - t0

    agent_tools_used = [s.tool for s in agent_result.steps if s.type == "action" and s.tool]
    agent_num_reasoning_steps = sum(1 for s in agent_result.steps if s.type == "action")
    agent_steps_full = [s.model_dump() for s in agent_result.steps]
    agent_sources = [s.model_dump() for s in agent_result.sources]
    logger.info(
        "    agent done in %.2fs -- %d step(s), tools: %s",
        agent_latency, agent_num_reasoning_steps, agent_tools_used,
    )

    logger.info("    running baseline...")
    t1 = time.monotonic()
    baseline_result = await baseline_rag.query(question)
    baseline_latency = time.monotonic() - t1
    logger.info("    baseline done in %.2fs", baseline_latency)

    logger.info("    judging correctness (agent, baseline)...")
    agent_correctness = await run_eval.score_correctness(question, ground_truth, agent_result.answer)
    baseline_correctness = await run_eval.score_correctness(question, ground_truth, baseline_result["answer"])

    logger.info("    judging faithfulness (agent, baseline)...")
    agent_faithfulness = await run_eval.score_faithfulness(
        question, agent_result.answer, run_eval._format_sources_for_prompt(agent_result.sources)
    )
    baseline_faithfulness = await run_eval.score_faithfulness(
        question, baseline_result["answer"], run_eval._format_chunks_for_prompt(baseline_result["retrieved_chunks"])
    )

    tool_selection_match = run_eval.score_tool_selection(agent_tools_used, expected_tools)

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
            "latency_seconds": round(agent_latency, 3),
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
            "latency_seconds": round(baseline_latency, 3),
        },
    }


async def main() -> None:
    test_set = json.loads(TEST_SET_PATH.read_text(encoding="utf-8"))
    total = len(test_set)

    results: list[dict] = []
    for i, item in enumerate(test_set, start=1):
        try:
            result = await run_one_heldout_query(i, total, item)
        except Exception as exc:  # noqa: BLE001
            logger.error("    ERROR on %s: %s", item["id"], exc)
            result = {
                "id": item["id"],
                "type": item["type"],
                "question": item["question"],
                "error": str(exc),
                "agent": {"correctness": {"correct": False}, "faithfulness": {"faithful": False},
                          "tool_selection_match": False, "num_reasoning_steps": 0},
                "baseline": {"correctness": {"correct": False}, "faithfulness": {"faithful": False}},
            }
        results.append(result)
        # Incremental write so a mid-run crash doesn't lose earlier progress --
        # this is a one-shot "run exactly once" script, not a resumable one,
        # so a re-run always starts fresh rather than skipping prior entries.
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Wrote %d results to %s", len(results), RESULTS_PATH)
    run_eval._print_aggregate_table(results)

    complete = [r for r in results if "error" not in r]
    agent_latencies = sorted(r["agent"]["latency_seconds"] for r in complete)
    baseline_latencies = sorted(r["baseline"]["latency_seconds"] for r in complete)

    def _median(values: list[float]) -> float:
        n = len(values)
        if n == 0:
            return 0.0
        mid = n // 2
        return values[mid] if n % 2 else (values[mid - 1] + values[mid]) / 2

    if complete:
        print()
        print("=" * 100)
        print("LATENCY (wall-clock seconds, N=%d complete queries)" % len(complete))
        print("=" * 100)
        print(f"{'Metric':<20} | {'Baseline RAG':>14} | {'ReAct Agent':>14}")
        print("-" * 100)
        print(f"{'Average':<20} | {sum(baseline_latencies)/len(baseline_latencies):>13.2f}s | "
              f"{sum(agent_latencies)/len(agent_latencies):>13.2f}s")
        print(f"{'Median':<20} | {_median(baseline_latencies):>13.2f}s | {_median(agent_latencies):>13.2f}s")
        print(f"{'Min':<20} | {min(baseline_latencies):>13.2f}s | {min(agent_latencies):>13.2f}s")
        print(f"{'Max':<20} | {max(baseline_latencies):>13.2f}s | {max(agent_latencies):>13.2f}s")
        print("=" * 100)


if __name__ == "__main__":
    asyncio.run(main())
