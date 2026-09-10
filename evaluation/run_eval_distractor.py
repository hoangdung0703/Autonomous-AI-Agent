"""Runs the same evaluation pipeline as run_eval.py (agent + baseline +
correctness/faithfulness judges, Section 15.2) but against
test-set-distractor.json, writing to results-distractor.json instead of the
main test-set.json / results.json pair.

Reuses run_one_query, _is_complete, and _print_aggregate_table from
run_eval.py unmodified -- this file only swaps the input/output paths and
mirrors run_eval.py's own resume behavior (skip any entry that already
completed cleanly; retry anything missing or holding an "error" stub, e.g.
from a prior quota exhaustion) so the original 20-query test set logic in
run_eval.py stays untouched.

Run from server/, same as run_eval.py:

    cd server
    ..\\venv\\Scripts\\python.exe ..\\evaluation\\run_eval_distractor.py
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import run_eval
from app.utils.logger import logger  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TEST_SET_PATH = EVAL_DIR / "test-set-distractor.json"
RESULTS_PATH = EVAL_DIR / "results-distractor.json"


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

    reusable = sum(1 for item in test_set if run_eval._is_complete(existing_by_id.get(item["id"])))
    if reusable:
        logger.info(
            "Resuming: %d/%d queries already complete in %s and will be skipped.",
            reusable, total, RESULTS_PATH,
        )

    results: list[dict] = []
    for i, item in enumerate(test_set, start=1):
        qid = item["id"]
        prior: Optional[dict] = existing_by_id.get(qid)
        if run_eval._is_complete(prior):
            logger.info("[%d/%d] %s (%s): SKIPPED -- reusing complete result from a prior run.",
                        i, total, qid, item["type"])
            result = prior
        else:
            try:
                result = await run_eval.run_one_query(i, total, item)
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
        RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Wrote %d results to %s", len(results), RESULTS_PATH)
    run_eval._print_aggregate_table(results)


if __name__ == "__main__":
    asyncio.run(main())
