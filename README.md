# Autonomous AI Agent for Enterprise Knowledge Retrieval

An AI agent for enterprise knowledge retrieval that reasons over a corporate document set using a **self-implemented ReAct (Reasoning + Acting) loop** — not a framework (no LangChain `AgentExecutor` or equivalent), and not a RAG wrapper with a chat UI glued on top. The distinguishing property is that the agent's Thought/Action/Observation trace, shown in full in the UI, is not decoration generated after the fact for display: it is a direct read of Gemini's native function-calling response object, and that same read is what drives execution. A tool call happens if and only if the model's response contains a `function_call` part; the loop stops if and only if it doesn't. The chain of thought the user sees is the actual control flow, not a summary of it.

## Architecture

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI |
| Frontend | React 19, Vite, TypeScript |
| Vector store | Qdrant Cloud (cosine similarity, 768-dim) |
| Conversation history | Supabase (Postgres only — Auth/Storage unused) |
| Reasoning / function calling | Gemini 3.1 Pro (`gemini-3.1-pro-preview`) |
| OCR (scanned documents) | Gemini Vision |
| Embeddings | `gemini-embedding-001` |
| Safe expression evaluation | `asteval` (explicitly not Python `eval()`) |

The system is a straightforward three-layer application: a React single-page chat client talks to a FastAPI backend over one endpoint (`POST /api/chat`); the backend runs the ReAct loop, dispatching to Qdrant for retrieval and Supabase for conversation persistence; Gemini is the reasoning engine for both the agent loop and, separately, ingestion-time OCR and embedding. There is no queue, no worker process, and no containerization — both cloud services (Qdrant, Supabase) are managed free tiers, so the only local processes are `uvicorn` and `npm run dev`.

### The ReAct loop

```
                        ┌──────────────────────────────────┐
                        │  Gemini generate_content(         │
                        │    contents, tools=TOOL_DECLS)     │
                        └──────────────────┬─────────────────┘
                                           │
                          response.candidates[0].content.parts
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     │                                            │
            has a function_call part                     text only, no function_call
                     │                                            │
                     ▼                                            ▼
       record Step(type="action", tool, params)          record Step(type="thought"/answer)
                     │                                            │
       observation = tool_registry.execute(tool, params)          ▼
                     │                                    Final Answer -- loop stops
       record Step(type="observation", content)
                     │
       append (model turn, tool response) to contents
                     │
                     └──────────► loop, up to MAX_AGENT_STEPS (6)
```

The stopping condition is model-driven: the 6-step cap (`MAX_AGENT_STEPS`) is a safety limit, not the normal way the loop ends. Across the 20-query evaluation set, the agent averages **1.65 reasoning steps per query** — most queries resolve in 1-2 steps, and the harder multi-hop/computation queries pull that average up, which is itself evidence that step count is adaptive rather than fixed. Self-correction (retrying a failed search with different wording, or calling `list_documents` when a search comes up empty) is not hardcoded fallback logic — it emerges entirely from feeding every observation, including `"No relevant chunks found."`, back into the conversation for the model to read and react to on its next turn.

## The five tools

Each tool is declared once as a Pydantic model, which is converted directly into a Gemini `FunctionDeclaration` — the same schema drives both FastAPI's validation and the model's function-calling contract.

- **`search_knowledge_base(query, category=None)`** — embeds the query and searches Qdrant, optionally filtered to `hr`/`legal`/`ops`. The primary retrieval tool; returns up to 5 chunks with document name, section title, excerpt, and similarity score.
- **`get_document(document_name)`** — retrieves every chunk of one named document in order, for when a single retrieved excerpt isn't enough surrounding context.
- **`list_documents(category=None)`** — lists everything currently indexed, for orientation when the model doesn't know what exists or a search comes back empty.
- **`compare_sections(doc1, doc2, topic)`** — searches two named documents for the same topic and returns both result sets side by side, for multi-hop reconciliation questions.
- **`calculate_or_verify(expression, context)`** — safely evaluates a numeric expression via `asteval`, never Python's `eval()`. This is the key differentiator from plain RAG: the agent retrieves a formula from a document, then actually computes with it, rather than retrieving a passage that merely states a formula and leaving the arithmetic to the reader (see the baseline's failure mode on computation queries below).

## Knowledge base

**51 documents, 306 chunks**, spanning `hr/`, `legal/`, and `ops/` categories. The knowledge base began at 9 documents (63 chunks) and was deliberately expanded to 51 (306 chunks) partway through the project specifically to stress-test retrieval precision at a more realistic corpus scale. The 42 added documents include a set of **deliberate decoys**: an archived 2023 handbook with a different (wrong, if used) probationary period than the current one, a superseded vendor-contract version with an outdated confidentiality term, and branch-specific leave-policy addenda (Ho Chi Minh City, Da Nang) with entitlements that differ from the company-wide default and from each other. These exist to test whether the agent (and the baseline) retrieve the *correct* source among close, similarly-worded near-duplicates rather than the most "familiar" or highest-frequency one — see the distractor evaluation set below.

Two documents (`vendor-contract-terms.pdf`, `incident-response-playbook.pdf`) have no extractable text layer and are ingested via the Gemini Vision OCR fallback rather than `pdfplumber`, exercising the full text-extraction pipeline described in the ingestion strategy, not just the common path.

## Getting started

No Docker, no deployment target — this runs locally against two managed cloud free tiers.

```bash
# 1. Create a Qdrant Cloud cluster and a Supabase project (web dashboards),
#    and note their URLs/keys.

# 2. Install backend dependencies
cd server
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Install frontend dependencies
cd ../client
npm install

# 4. Configure environment
cp server/.env.example server/.env
cp client/.env.example client/.env
# Fill in: GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, SUPABASE_URL, SUPABASE_KEY

# 5. In the Supabase SQL Editor, run the `conversations` / `messages` table
#    definitions once (see server/app/services/supabase_service.py for the
#    exact columns each table needs).

# 6. Seed Qdrant from server/knowledge-base/ (idempotent -- skips anything
#    already indexed; pass --force to re-ingest everything)
cd ../server
python scripts/seed.py

# 7. Start the backend
uvicorn app.main:app --reload --port 5001

# 8. Start the frontend (separate terminal)
cd ../client && npm run dev   # http://localhost:5173
```

## Evaluation results

Three layers of evidence, run in sequence as the project matured: a main 26-query set (the original 20-query test set plus a 6-query distractor set, both re-verified at two knowledge-base scales), and finally a 10-query held-out set constructed and run only after all debugging was frozen, specifically to check for overfitting to the development process.

### Layer 1: main 20-query set, 306-chunk corpus (final, locked)

| Metric | Baseline RAG | ReAct Agent | Improvement |
|---|---|---|---|
| Task Completion Rate (all) | 90.0% | 95.0% | +5.0% |
| Task Completion — Simple | 100.0% | 100.0% | +0.0% |
| Task Completion — Multi-hop | 71.4% | 85.7% | +14.3% |
| Task Completion — Computation | 100.0% | 100.0% | +0.0% |
| Answer Faithfulness | 100.0% | 100.0% | +0.0% |
| Avg Reasoning Steps | 1.0 | 1.65 | — |

Tool Selection Accuracy (agent): 20/20 (100.0%). N = 20 (simple = 8, multi-hop = 7, computation = 5).

**A note on methodology:** the initial evaluation, run against the original 9-document (63-chunk) knowledge base, showed the agent and baseline essentially **tied on overall task completion (95% each)**, with the agent ahead only on faithfulness (90% vs 85%) and tool selection. At that scale, retrieval is close to trivial — nearly every query's relevant chunk is one of only a handful of candidates, so a fixed top-5 similarity search performs about as well as an agent that can reason about what to search for. Rather than treating that as the headline result, the knowledge base was deliberately expanded 5x, including documents specifically designed to be confusable with each other, before re-running the same 20 queries. That is where the agent's actual advantage shows up: **+14.3 points on multi-hop reasoning**. This is a more defensible finding than assuming the gap upfront — it isolates the agent's advantage to what should genuinely require reasoning (cross-document synthesis, disambiguating near-duplicate sources) rather than crediting it for a retrieval task simple enough that any competent similarity search would also succeed at.

### Layer 2: distractor set — 6 queries targeting the decoy/superseded documents

| Metric | Baseline RAG | ReAct Agent | Improvement |
|---|---|---|---|
| Task Completion Rate | 83.3% (5/6) | 100.0% (6/6) | +16.7% |
| Answer Faithfulness | 100.0% | 100.0% | +0.0% |

The baseline's one failure (distinguishing the current, non-archived probationary period from an archived document with a different figure) is exactly the failure mode this set was designed to surface: its fixed top-5, no-reasoning retrieval sometimes doesn't even return the current document in its top results, let alone reconcile it against the decoy. The agent's extra retrieval/reasoning step consistently disambiguated current-vs-archived and correct-vs-superseded documents across all 6 queries.

### Layer 3: held-out validation set — 10 queries, constructed after debugging was frozen

Once every bug fix from the section below had been made and reverified, 10 entirely new queries were written against previously-untested facts from the 42 added documents (on-call compensation, learning budget, procurement/expense approval thresholds, disaster-recovery RPO vs backup frequency, sales commission, travel per-diem, and three further decoy/distractor pairs) and run exactly once, under a rule of accepting whatever the results showed rather than iterating further:

| Metric | Baseline RAG | ReAct Agent | Improvement |
|---|---|---|---|
| Task Completion Rate (all) | 80.0% | 100.0% | +20.0% |
| Task Completion — Simple | 100.0% | 100.0% | +0.0% |
| Task Completion — Multi-hop | 66.7% | 100.0% | +33.3% |
| Task Completion — Computation | 100.0% | 100.0% | +0.0% |
| Answer Faithfulness | 100.0% | 100.0% | +0.0% |

Tool Selection Accuracy (agent): 9/10 (90.0%) — the one mismatch is a test-annotation artifact, not an agent defect: one query's expected tool was annotated as `search_knowledge_base`, but the agent instead used `list_documents` followed by `compare_sections`, which still retrieved both relevant documents and produced a correct, fully faithful answer. All 10 agent answers were correct and faithful; the baseline's two failures reproduced the same pattern seen in Layers 1-2 — declaring a fact "not in the provided context" when it was simply outside its fixed top-5 retrieval, not actually absent from the corpus.

This is the most important result in the whole evaluation for the overfitting question: performance on genuinely unseen queries, against genuinely untested facts, was **equal to or stronger than the main set**, not weaker (100% vs 95% overall completion, +33.3 vs +14.3 points on multi-hop). None of these documents or facts were touched during any of the five bug-fix cycles below. That is meaningful evidence that the agent's advantage generalizes, rather than being an artifact of five rounds of fixing behavior specifically on the 26 queries used during development.

### Latency

Wall-clock time was measured per query on the held-out run:

| | Baseline RAG | ReAct Agent |
|---|---|---|
| Average | 7.61s | 12.59s |
| Median | 8.19s | 11.66s |
| Min | 4.99s | 7.69s |
| Max | 10.06s | 22.45s |

The agent costs roughly 1.6-1.7x the baseline's latency on average — an honest and expected accuracy-vs-speed tradeoff: every additional Thought/Action/Observation cycle is another full round trip to Gemini. The multi-hop queries that most benefit from the agent's reasoning (Layer 3's +33.3-point gain) are also the ones that take the longest, since they require 2+ tool calls rather than 1.

### Known limitation: retrieval breadth on 3-document comparisons

One test query (internally `q15`, asking the agent to compare confidentiality obligations across three separate documents) is not answered completely by either system: the agent's search queries surface two of the three relevant documents but not the third, so the missing fact is never retrieved to synthesize into the final answer. This is a retrieval-breadth/query-diversity gap, not a reasoning or synthesis failure — the agent correctly combines whatever it retrieves, it simply doesn't retrieve enough. It is left unfixed and documented rather than patched with a special case; see Future Work below.

## Engineering robustness — bugs found and fixed during evaluation

Five real bugs surfaced during evaluation and live testing over the course of the project, each found through actual failing behavior rather than code review, each diagnosed to a root cause, fixed, and reverified with a concrete before/after trace rather than assumed fixed. This iterative find-fix-reverify cycle is treated here as a feature of the engineering process, not an embarrassment to minimize:

1. **Empty Final Answer.** The agent could return a completely empty answer when Gemini produced a response with neither a `function_call` nor any text (found on a multi-hop query). Fixed by treating empty/whitespace text as an invalid response requiring a retry, not a valid (if vacuous) stopping condition.
2. **Retry budget silently draining the step budget.** The initial fix for (1) retried using the same `MAX_AGENT_STEPS` budget used for genuine tool calls, so a query needing several real steps could exhaust its entire step budget on invisible empty-response retries and hit the step-limit fallback despite having already gathered everything it needed. Fixed by giving invalid-response retries their own independent `MAX_INVALID_RETRIES` budget, with every retry recorded as a visible `Step(type="retry")` in the trace so it can never be silently absorbed again.
3. **Malformed/hallucinated tool-echo output.** A rare Gemini sampling failure produced a non-empty but garbled response — fabricated facts structurally echoing the model's own internal function-call/response wire format, with a stray non-Latin character leading the text. Fixed with `is_malformed_response()`, a conservative heuristic catching control characters, unexpected non-Latin Unicode script blocks, and internal-formatting echo patterns (`"document:"+"section:"+"score:"` co-occurring, `"default_api:"`); reproduced the failure and confirmed the guard recovers a valid, correct answer on retry.
4. **Citation/source fabrication at scale.** After the knowledge base expanded to 306 chunks, one computation query's agent answer cited a document (`compensation-and-benefits.pdf`) that does not exist anywhere in the 51-file corpus, along with a fabricated formula, under genuine retrieval ambiguity (a real, similarly-named document existed among the new near-duplicates). Fixed with a strengthened system instruction plus a citation-verification check reusing the same retry infrastructure from (2)/(3): a Final Answer citing any document never seen in that run's real tool observations is rejected and retried. Reverified against the exact fabricated text (still correctly caught) and spot-checked against known-correct answers (zero false positives).
5. **Conversation-history scoping gap in the citation guard.** Live multi-turn testing surfaced a false positive in fix (4): the guard only tracked documents observed within the *current* `run_agent()` call, so when the agent correctly recalled and re-cited a document from an earlier turn's real observation — without re-searching, purely from conversation memory — the guard misflagged it as fabrication and forced an unnecessary retry. Fixed by seeding the guard's known-document set from citations already present in the loaded conversation history's prior agent turns (reusing the same extraction logic, not a duplicate mechanism). Reverified live: the same 3-turn conversation that previously showed one unneeded retry on turn 2 now completes turn 2 with no citation retry and turn 3 with zero tool calls and zero retries, while the fabrication case from (4) is still caught with no regression.

## Testing

Three distinct testing layers, each serving a different purpose:

- **`server/tests/`** — a pytest suite of 29 unit tests covering pure logic with no live API calls (fast, safe to run offline or in CI): `test_semantic_chunker.py` (heading detection, oversized-section splitting, undersized-section merging, empty input), `test_calculate_or_verify.py` (valid/invalid expressions, a static AST check that the builtin `eval()` is never called), `test_parse_agent_response.py` (mocked Gemini response shapes — function-call, text-only, empty), `test_malformed_response_guard.py` (the malformed-response and citation-verification heuristics from `agent_loop.py`), and `test_tool_registry.py` (dispatch and error-handling contract). Run with `pytest server/tests/ -v`.
- **`server/scripts/verify_*.py`** — manual smoke-test scripts (`verify_services.py`, `verify_tools.py`, `verify_agent.py`) that make real calls to Gemini, Qdrant, and Supabase, run by hand against a configured `.env` rather than as an automated suite.
- **`evaluation/run_eval.py`** — the full evaluation harness described above: runs both the agent and the baseline against every test-set query, scores correctness and faithfulness via LLM-as-judge, scores tool-selection accuracy, and is resumable (a partial or interrupted run never re-spends API calls on already-completed queries). `run_eval_distractor.py` and `run_eval_heldout.py` reuse its scoring functions unmodified against the distractor and held-out sets respectively; `run_eval_heldout.py` additionally records per-query wall-clock latency.

## Multi-turn conversation

Conversation history is persisted in Supabase (`conversations` / `messages` tables) and loaded back on every request, keyed by `conversation_id`. This was verified live, not just by code inspection: a three-turn conversation was sent through the real API, where each follow-up question deliberately omitted the context needed to answer it standing alone. Turn 1 asked for the overtime hourly-rate formula; turn 2 asked "If my base salary is 25,000,000 VND, how much would that be?" — with no mention of "overtime," "hourly rate," or any calculation — and the agent correctly inferred it should apply the previously-retrieved formula, going straight to `calculate_or_verify(25000000 / 176)` without needing to re-search. Turn 3 asked "What document did that formula come from again?" and the agent answered correctly from conversation memory alone, with zero tool calls. All six messages (three user, three agent, each with its full reasoning trace and sources) were confirmed persisted in Supabase in the correct order under the shared conversation id.

## Cost control

Every Gemini Pro call in the project — the agent loop, the baseline, and every LLM-as-judge evaluation call — goes through one shared client wrapper with `thinking_config=ThinkingConfig(thinking_level="low")` hard-coded, not exposed as an environment variable or a caller-overridable parameter. Thinking tokens bill as output tokens at a materially higher rate than a low thinking budget, and this project's total budget target was on the order of $10-15 for the entire evaluation cycle (ingestion embeddings, the 20-query eval running agent + baseline + two judges per query, plus repeated re-evaluation runs across the fix/reverify cycles above). A single hard cap applied uniformly, rather than a per-call-site setting that could drift, was the deliberate way to make that budget constraint actually hold.

## Project structure

```
Autonomous AI Agent/
├── server/
│   ├── app/
│   │   ├── agent/
│   │   │   ├── agent_loop.py            # Self-implemented ReAct loop
│   │   │   ├── tool_registry.py         # Tool registration and dispatch
│   │   │   └── tools/                   # search_knowledge_base, get_document,
│   │   │                                #   list_documents, compare_sections,
│   │   │                                #   calculate_or_verify
│   │   ├── ingestion/                   # text_extractor, ocr_extractor,
│   │   │                                #   semantic_chunker, embedder,
│   │   │                                #   document_ingester
│   │   ├── services/                    # vector_db_service, llm_service,
│   │   │                                #   supabase_service, conversation_service
│   │   ├── routes/                      # chat.py (POST /api/chat), health.py
│   │   ├── utils/                       # logger.py, parse_agent_response.py
│   │   ├── config.py, models.py, main.py
│   ├── knowledge-base/                  # 51 documents: hr/, legal/, ops/
│   ├── scripts/
│   │   ├── seed.py                      # Ingest knowledge-base/ into Qdrant
│   │   └── verify_*.py                  # Manual real-API smoke tests
│   ├── tests/                           # pytest suite -- pure logic, no live calls
│   ├── requirements.txt, pytest.ini, .env.example
├── client/
│   ├── src/
│   │   ├── components/                  # ChatWindow, MessageBubble,
│   │   │                                #   ReasoningStep, InputBar
│   │   ├── services/api.ts
│   │   ├── types/index.ts               # Shared TS types matching server models
│   │   ├── App.tsx, main.tsx
│   ├── package.json, tsconfig.json, .env.example
├── evaluation/
│   ├── test-set.json                    # 20-query evaluation set
│   ├── test-set-distractor.json         # 6-query decoy/superseded-doc set
│   ├── test-set-heldout.json            # 10-query held-out validation set
│   ├── baseline_rag.py                  # Non-agentic single-shot RAG comparison
│   ├── run_eval.py, run_eval_distractor.py, run_eval_heldout.py
│   ├── results.json, results-distractor.json, results-heldout.json,
│   │   results_63chunks_baseline.json
├── requirements.md                      # Full project specification
└── README.md
```

`evaluation/` and `server/tests/` did not exist in the original project plan (`requirements.md`) at the level of detail present here — the distractor set, the resumable/incremental evaluation runner, and the pure-logic pytest suite were all added as the project matured past the original specification.

## Known limitations and future work

- **Retrieval breadth on multi-entity comparisons.** The `q15` limitation above is the clearest open item: when a question spans three or more documents, the agent's search queries don't reliably cover all of them. A reasonable next step is detecting multi-entity comparison questions (e.g., by prompting the model to enumerate the entities/documents it believes are relevant before searching) and issuing one search per entity rather than relying on one or two general queries to happen to surface everything.
- **The "RAG can't compute" argument weakens against frontier models, and that's fine.** `calculate_or_verify` was motivated by the observation that a plain retrieval system returns a formula as prose and leaves the arithmetic to the reader (documented in the computation case study). Increasingly capable frontier LLMs can often do this arithmetic correctly in their own generated text without an explicit tool, which erodes "the baseline literally cannot compute" as a permanent differentiator. The more durable argument is not computational capability but **auditability**: `calculate_or_verify` produces a discrete, inspectable step in the trace — the exact expression evaluated and the exact result — rather than a number that appeared inside a paragraph with no way to verify how it was derived. The citation-verification guard (bug 4/5 above) is the same principle applied to sourcing: an architecture that can catch and correct its own ungrounded claims is a stronger property than one that merely tends to produce correct output.
- **Accuracy-vs-latency is an open tradeoff, not a solved problem.** The agent's ~1.6-1.7x latency multiplier over the baseline (12.59s vs 7.61s average, held-out set) is the direct cost of multi-step reasoning, and it scales with query difficulty — exactly the queries that most need the extra reasoning steps are the slowest. Nothing in the current design (e.g. parallelizing independent tool calls within a single turn, or caching repeated sub-queries within a conversation) attempts to close this gap; it is presented honestly as a cost of the architecture rather than optimized away, and is a reasonable target for future work if latency became a hard product constraint.
- **Gemini's free-tier daily request quota** (250 requests/day for `gemini-3.1-pro-preview`) was hit more than once during full re-evaluation cycles and is a hard external constraint on how quickly a complete evaluation run (agent + baseline + two LLM-as-judge calls per query, times 20-26 queries) can be repeated in a single day.
- **A residual trust assumption in the conversation-history fix (bug 5).** Seeding the citation guard's known-document set from prior agent turns' answers assumes those answers were themselves legitimately grounded when produced — true for any turn generated under this guard, but not independently re-verified against a persistent audit log. A hallucination that entered conversation history before this guard existed would be "trusted" by a later turn recalling it. This is an inherent tradeoff of allowing genuine memory-based recall at all, not a defect introduced by the fix, but worth stating plainly rather than leaving implicit.
