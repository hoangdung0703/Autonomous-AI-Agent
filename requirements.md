# Autonomous AI Agent for Enterprise Knowledge Retrieval
## Project Requirements & Specification

---

## 1. Project Summary

A genuine AI Agent system — not a RAG web app — that autonomously reasons over a corporate knowledge base using a self-implemented ReAct (Reasoning + Acting) loop. Users interact via a minimal single-page chat interface with no login required. The agent's full chain of thought is displayed in real time.

**Stack:** Python 3.11+ + FastAPI · React 19 + Vite + TypeScript · Qdrant Cloud · Supabase (Postgres) · Gemini 3.1 Pro · Gemini Vision (OCR) · gemini-embedding-001

---

## 2. Project Structure

```
enterprise-ai-agent/
├── server/
│   ├── app/
│   │   ├── agent/
│   │   │   ├── agent_loop.py          # Self-implemented ReAct loop
│   │   │   ├── tool_registry.py       # Tool registration and dispatch
│   │   │   └── tools/
│   │   │       ├── search_knowledge_base.py
│   │   │       ├── get_document.py
│   │   │       ├── list_documents.py
│   │   │       ├── compare_sections.py
│   │   │       └── calculate_or_verify.py
│   │   ├── ingestion/
│   │   │   ├── document_ingester.py   # Orchestrates full ingestion pipeline
│   │   │   ├── text_extractor.py      # pdfplumber, python-docx, openpyxl
│   │   │   ├── ocr_extractor.py       # Gemini Vision for scanned PDFs
│   │   │   ├── semantic_chunker.py    # Semantic chunking strategy
│   │   │   └── embedder.py            # gemini-embedding-001 with batching
│   │   ├── services/
│   │   │   ├── vector_db_service.py   # Qdrant client wrapper
│   │   │   ├── llm_service.py         # Gemini 3.1 Pro wrapper (function calling)
│   │   │   ├── supabase_service.py    # Supabase client (conversation history)
│   │   │   └── conversation_service.py
│   │   ├── routes/
│   │   │   ├── chat.py                # POST /api/chat
│   │   │   └── health.py              # GET /api/health
│   │   ├── utils/
│   │   │   ├── logger.py              # Python logging config
│   │   │   └── parse_agent_response.py # Parse Thought/Action/Answer from Gemini response
│   │   ├── config.py                  # Pydantic Settings, env validation
│   │   ├── models.py                  # Pydantic request/response models
│   │   └── main.py                    # FastAPI app entrypoint
│   ├── knowledge-base/                # Pre-loaded documents (seeded at startup)
│   │   ├── hr/
│   │   │   ├── employee-handbook.pdf
│   │   │   ├── leave-policy.pdf
│   │   │   └── benefits-guide.pdf
│   │   ├── legal/
│   │   │   ├── nda-template.pdf
│   │   │   ├── data-privacy-policy.pdf
│   │   │   └── vendor-contract-terms.pdf
│   │   └── ops/
│   │       ├── it-onboarding-guide.pdf
│   │       ├── expense-claim-procedure.pdf
│   │       └── incident-response-playbook.pdf
│   ├── scripts/
│   │   └── seed.py                    # Run once to ingest knowledge-base/ into Qdrant
│   ├── requirements.txt
│   └── .env.example
├── client/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── ReasoningStep.tsx      # Thought / Action / Observation display
│   │   │   └── InputBar.tsx
│   │   ├── services/
│   │   │   └── api.ts
│   │   ├── types/
│   │   │   └── index.ts               # Shared TS types (matches server Pydantic models)
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── .env.example
├── evaluation/
│   ├── test-set.json
│   ├── baseline_rag.py
│   └── run_eval.py
├── .gitignore
└── README.md
```

---

## 3. Environment Variables

### server/.env.example
```
GOOGLE_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.1-pro
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSIONS=768

QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=enterprise_knowledge

SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_service_role_key

PORT=5001
ENVIRONMENT=development
MAX_AGENT_STEPS=6
```

### client/.env.example
```
VITE_API_URL=http://localhost:5001
```

> No Docker required anywhere in this project. Qdrant and Supabase are both cloud-managed (free tier). Only local processes are `uvicorn` (backend) and `npm run dev` (frontend).

---

## 4. Agent — ReAct Loop

### 4.1 Overview

Self-implemented in `agent_loop.py` — NO LangChain AgentExecutor, no agent framework.

- **Max steps:** 6 (env: `MAX_AGENT_STEPS`) — a safety cap, not the normal termination
- **Termination:** model returns plain text (no function call) → treated as Final Answer, OR max steps reached

### 4.2 Control Flow — Gemini Native Function Calling

The Thought/Action/Observation loop is driven by Gemini's structured function calling, not free-text regex parsing. Tool schemas are defined once as Pydantic models and converted to Gemini `FunctionDeclaration` objects — this keeps the tool contract type-safe and is the natural fit between FastAPI/Pydantic and Gemini's function calling API.

**System instruction (sets behavior, not format-parsing):**
```
You are an AI Agent for enterprise knowledge retrieval at VinTech Corp.
Before each tool call, briefly state your reasoning as plain text (this is your "Thought").
Then call the appropriate function.
When you have enough information, respond with plain text only (no function call) —
this is your Final Answer. Always cite the source document and section name in the
Final Answer. If context is insufficient, say so — do not fabricate information.
For numeric questions, retrieve the formula first, then use calculate_or_verify.
```

### 4.3 Loop Logic (pseudocode, Python)

```python
async def run_agent(question: str, conversation_history: list[dict]) -> AgentResult:
    steps: list[Step] = []
    contents = build_initial_contents(question, conversation_history)

    for i in range(settings.MAX_AGENT_STEPS):
        response = await llm_service.generate(contents, tools=TOOL_DECLARATIONS)
        thought_text, function_call = parse_agent_response(response)

        if thought_text:
            steps.append(Step(type="thought", content=thought_text))

        if function_call is None:
            # Model chose to answer directly -> stop
            return AgentResult(answer=thought_text, steps=steps)

        steps.append(Step(type="action", tool=function_call.name, params=function_call.args))
        observation = await tool_registry.execute(function_call.name, function_call.args)
        steps.append(Step(type="observation", content=observation))

        contents = append_turn(contents, function_call, observation)

    return AgentResult(
        answer="Could not complete reasoning within step limit.",
        steps=steps,
    )
```

### 4.4 Self-Correction Requirement

If a tool observation indicates an empty/poor result (e.g. `"No relevant chunks found."`), the model must be able to read this in the next turn and decide to retry with a different query, call `list_documents`, or change the category filter. This is driven entirely by the model reading the Observation text — never hardcoded fallback logic in Python. At least one test case in the evaluation set must demonstrate this behavior end-to-end.

---

## 5. Agent Tools (5 total)

Each tool is a Pydantic model (defines the function-calling schema) + an async Python function (the implementation). Tools never call `eval()`.

### 5.1 search_knowledge_base(query: str, category: Optional[str] = None)
- Embeds query via `gemini-embedding-001`
- Queries Qdrant, optional payload filter: `category == hr | legal | ops`
- Returns top 5 chunks: `document_name`, `section_title`, `excerpt` (400 chars), `similarity_score`

### 5.2 get_document(document_name: str)
- Retrieves all chunks for a document ordered by `chunk_index` (Qdrant scroll/filter by payload)
- Returns concatenated full content
- Used for deep reading of a specific file

### 5.3 list_documents(category: Optional[str] = None)
- Returns unique document names + category + chunk count from Qdrant payload aggregation
- Optional filter by category: hr / legal / ops
- Used for agent orientation before searching

### 5.4 compare_sections(doc1: str, doc2: str, topic: str)
- Runs `search_knowledge_base(topic)` filtered to `doc1` → top 3 chunks
- Runs `search_knowledge_base(topic)` filtered to `doc2` → top 3 chunks
- Returns both result sets side by side for comparison

### 5.5 calculate_or_verify(expression: str, context: str)
- Evaluates a numeric expression safely using **`asteval`** (NOT Python `eval()`)
- `expression`: math string e.g. `"(8/12) * 12"`
- `context`: plain text description e.g. `"annual leave for 8 months"`
- Returns: `{ "expression": ..., "result": ..., "context": ... }`
- **This is the key differentiator from RAG** — agent retrieves the formula, then computes

```python
from asteval import Interpreter

aeval = Interpreter()

def calculate_or_verify(expression: str, context: str) -> dict:
    result = aeval(expression)
    if aeval.error:
        raise ValueError(f"Invalid expression: {expression}")
    return {"expression": expression, "result": result, "context": context}
```

**Example:**
```
Query: "Employee worked 8 months — how many days annual leave?"
→ search_knowledge_base("annual leave formula") → "12 days/year, pro-rated monthly"
→ calculate_or_verify("(8/12)*12", "leave for 8 months") → 8.0
→ Final Answer: "8 days" with citation
```

---

## 6. Document Ingestion Pipeline

### 6.1 Text Extraction Strategy
```
PDF   → try pdfplumber first
        if extracted text < 100 chars → treat as scanned → Gemini Vision OCR
DOCX  → python-docx paragraph extraction
XLSX  → openpyxl → convert each sheet to CSV-like text block
```

### 6.2 Semantic Chunking (semantic_chunker.py)

Split at natural section boundaries, NOT fixed character count. Two pattern sets are tried in order (documents in this project may mix English business-doc headings and Vietnamese legal-style headings):

1. **Pattern set A (generic headings):** `\n#{1,3} `, `\n[A-Z][A-Z ]{5,}\n`, `\n\n`
2. **Pattern set B (VN legal-style):** `\nĐiều \d+`, `\nKhoản \d+`, `\nMục \d+`
3. Try pattern set A first; if fewer than 2 sections detected, retry with pattern set B; if both fail, fall back to paragraph-based splitting (`\n\n`)
4. If a resulting chunk > 1200 chars → split further at sentence boundaries (`. `, `\n`)
5. If a chunk < 150 chars → merge with the next section
6. Store the detected section title with each chunk

**Chunk metadata (Qdrant payload):**
```python
{
    "document_name": "leave-policy.pdf",
    "category": "hr",                      # hr | legal | ops
    "section_title": "Annual Leave Entitlement",
    "chunk_index": 3,
    "page_estimate": 2,                    # chunk_index * avg_chars_per_page estimate
    "char_count": 487,
}
```

### 6.3 Embedding (embedder.py)
- Model: `gemini-embedding-001` via REST (`v1beta`), `output_dimensionality=768` (must match `EMBEDDING_DIMENSIONS` and the Qdrant collection's vector size exactly)
- Batch size: 2, using `asyncio.gather` with `return_exceptions=True`
- Adaptive delay: 500ms → 2000ms when `consecutive_failures > 3`
- Retry: 3 attempts per chunk

### 6.4 Qdrant Collection Format
```python
# Collection created once in seed.py:
client.create_collection(
    collection_name=settings.QDRANT_COLLECTION,
    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
)

# Point upserted per chunk:
PointStruct(
    id=f"{document_name}_chunk_{chunk_index}",   # or a deterministic UUID5 of this string
    vector=[...768 floats],
    payload={
        "document_name": document_name,
        "category": category,
        "section_title": section_title,
        "chunk_index": chunk_index,
        "page_estimate": page_estimate,
        "char_count": char_count,
        "text": chunk_text,   # store raw text in payload for retrieval without a second lookup
    },
)
```

### 6.5 Seed Script (scripts/seed.py)
```
1. Scan knowledge-base/ recursively
2. Detect category from folder name
3. Run ingestion pipeline per file
4. Log: [SUCCESS] leave-policy.pdf -> 24 chunks | [FAIL] reason
5. Print summary: X/Y documents indexed, N total chunks
```

---

## 7. Conversation History (Supabase Postgres)

Supabase is used ONLY as a Postgres database for conversation history — Supabase Auth and Storage are not used. No login, no RLS policies beyond the default.

**Table: `conversations`**
```sql
create table conversations (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz default now()
);
```

**Table: `messages`**
```sql
create table messages (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid references conversations(id),
    role text not null,              -- 'user' | 'agent'
    content text not null,
    steps jsonb,                     -- full Thought/Action/Observation trace, null for user messages
    sources jsonb,                   -- cited sources, null for user messages
    created_at timestamptz default now()
);
```

Accessed from FastAPI via the `supabase-py` client (`supabase_service.py`).

---

## 8. API Specification

### POST /api/chat
**Request:**
```json
{
  "question": "How many days of annual leave for 8 months worked?",
  "conversation_id": "optional-uuid"
}
```

**Response:**
```json
{
  "conversation_id": "uuid",
  "steps": [
    { "type": "thought", "content": "I need to find the leave formula..." },
    { "type": "action", "tool": "search_knowledge_base", "params": { "query": "annual leave formula", "category": "hr" } },
    { "type": "observation", "content": "Found: 12 days/year pro-rated monthly in leave-policy.pdf" },
    { "type": "thought", "content": "Now I can calculate for 8 months." },
    { "type": "action", "tool": "calculate_or_verify", "params": { "expression": "(8/12)*12", "context": "annual leave 8 months" } },
    { "type": "observation", "content": "Result: 8.0" }
  ],
  "answer": "An employee who has worked 8 months is entitled to **8 days** of annual leave.",
  "sources": [
    { "document_name": "leave-policy.pdf", "section_title": "Annual Leave Entitlement", "excerpt": "12 days per calendar year, pro-rated..." }
  ]
}
```

### GET /api/health
```json
{
  "status": "ok",
  "qdrant": "connected",
  "supabase": "connected",
  "gemini": "reachable",
  "documents_indexed": 47
}
```

All request/response bodies are defined as Pydantic models in `app/models.py`, which FastAPI uses to auto-generate OpenAPI docs at `/docs` — useful to show the hội đồng a live, self-documenting API during the demo.

---

## 9. Frontend — Single Page Chat

### 9.1 Components

| Component | Responsibility |
|---|---|
| `App.tsx` | Root — holds `conversationId` state |
| `ChatWindow.tsx` | Scrollable message thread |
| `MessageBubble.tsx` | Renders user or agent message |
| `ReasoningStep.tsx` | Collapsible Thought/Action/Observation panel |
| `InputBar.tsx` | Text input + send, disabled while loading |

### 9.2 UX Flow

```
User types question → press Enter or Send
    ↓
InputBar disabled, "Agent is thinking..." indicator
    ↓
Steps appear one by one:
  🤔 Thought: I need to find...
  🔍 search_knowledge_base("leave formula", "hr")
  📄 Found: 12 days/year, pro-rated...
  🔢 calculate_or_verify("(8/12)*12", ...)
  📊 Result: 8.0
    ↓
Final answer rendered in markdown
Sources shown as small document chips
InputBar re-enabled
```

### 9.3 Design Constraints (Light Theme)
- **Light theme** — white/light gray background for readability in thesis screenshots
- Background: `#FFFFFF` or `#F8F9FA`
- Text: `#1A1A2E` (dark, high contrast)
- Accent: `#4F46E5` (indigo) for buttons and highlights
- ⚠️ NO dark backgrounds — dark UI screenshots in thesis are hard to read when printed
- No sidebar, no header nav, no auth UI
- Reasoning steps: collapsed by default, expand on click
- Mobile responsive

---

## 10. Server Dependencies (requirements.txt)

```
fastapi>=0.115
uvicorn[standard]>=0.30
pydantic>=2.9
pydantic-settings>=2.5
google-genai>=0.3
qdrant-client>=1.11
supabase>=2.9
pdfplumber>=0.11
python-docx>=1.1
openpyxl>=3.1
asteval>=1.0
python-dotenv>=1.0
```

## 11. Client Dependencies

```json
{
  "react": "^19.0.0",
  "vite": "^6.x",
  "typescript": "^5.x",
  "axios": "^1.x",
  "react-markdown": "^9.x",
  "remark-gfm": "^4.x"
}
```

---

## 12. Getting Started (Local Only — No Deployment, No Docker)

> ⚠️ This project runs locally only — no Render/Vercel deployment. Qdrant and Supabase are cloud-managed free tiers; no containers are run on the local machine.

```bash
# 1. Create Qdrant Cloud cluster and Supabase project (web dashboards), note the URLs/keys

# 2. Install backend dependencies
cd server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Install frontend dependencies
cd ../client
npm install

# 4. Configure environment
cp server/.env.example server/.env
cp client/.env.example client/.env
# Fill in: GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, SUPABASE_URL, SUPABASE_KEY

# 5. Run the Supabase SQL from Section 7 in the Supabase SQL Editor (once)

# 6. Add documents to knowledge-base/ folders, then seed Qdrant
cd ../server
python scripts/seed.py

# 7. Start backend
uvicorn app.main:app --reload --port 5001

# 8. Start frontend
cd ../client && npm run dev   # runs on :5173
```

---

## 13. Critical Agent Design Clarifications

These points directly answer "is this a real agent or fake agent?" — hội đồng sẽ hỏi thẳng.

### 13.1 Thought/Action MUST control execution flow — not just UI decoration

**Real agent (what we implement):**
```
Gemini returns a response with a text part (the "Thought") AND/OR a function_call part.
Python reads response.candidates[0].content.parts:
  - if a part has .function_call -> dispatch that tool, execution branches on the model's choice
  - if a part has .text and no function_call anywhere in the response -> treat as Final Answer
```

**Fake agent (what we must NOT do):**
```
Python has a hardcoded tool call sequence.
Gemini is called separately just to generate "Thought: ..." text for display.
```

**Implementation:** Use Gemini **native function calling** (structured output) — not free-text parsing with regex. Tool schemas are declared once as Pydantic models, converted to `FunctionDeclaration` objects, and passed via the `tools` parameter. This is verifiable — the tool call literally cannot happen without the model's decision.

```python
# llm_service.py — Gemini function calling
response = await client.aio.models.generate_content(
    model=settings.GEMINI_MODEL,
    contents=contents,
    config=GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[Tool(function_declarations=TOOL_DECLARATIONS)],
    ),
)

part = response.candidates[0].content.parts[0]
if part.function_call:
    # Model decided to call a tool -- dispatch it
    return ParsedResponse(type="action", tool=part.function_call.name, params=dict(part.function_call.args))
else:
    # Model decided to produce a final answer
    return ParsedResponse(type="final_answer", content=part.text)
```

### 13.2 Stopping condition is model-driven, not always max_steps

Max 6 steps is a safety cap — NOT the normal termination. The model must be able to stop early by choosing to return a final answer instead of a function call.

With Gemini function calling, this is natural:
- Response contains a `function_call` part → loop continues
- Response contains only a `text` part → loop stops, treat as Final Answer

This means the model autonomously decides "I have enough information" — not the code deciding for it. Track and report: average steps per query (should be 2-4, not always 6).

### 13.3 Self-correction behavior — must be demonstrable

The agent must handle empty/poor search results by trying a different approach:

```python
# After tool execution, if the observation indicates an empty result:
# observation = "No relevant chunks found."
# -> The agent sees this in the next turn's contents, and can decide to:
#    - Try different query wording
#    - Call list_documents() to see what's available
#    - Try a different category filter
# This is driven by the model reading the Observation -- never hardcoded fallback logic in Python.
```

**Must have at least 1 test case demonstrating this:**
- Query designed to fail on first search attempt
- Agent retries with a different strategy
- Agent ultimately finds the answer (or correctly says it cannot)
- Screenshot/log of this multi-step correction shown in the thesis

### 13.4 Baseline comparison — mandatory for evaluation

Must implement a simple RAG baseline to compare against:

**Baseline (evaluation/baseline_rag.py):**
```python
async def query(question: str) -> dict:
    embedding = await embed(question)
    chunks = qdrant_client.search(collection_name=..., query_vector=embedding, limit=5)
    answer = await llm_service.generate_simple(build_simple_prompt(question, chunks))
    return {"answer": answer, "steps": 1}  # always 1 step
```

**Evaluation table structure:**

| Query Type | Metric | Baseline RAG | ReAct Agent |
|---|---|---|---|
| Simple factual | Correctness | X% | Y% |
| Multi-hop (cross-doc) | Correctness | X% | Y% |
| Computation required | Correctness | 0% | Y% |
| Avg reasoning steps | Steps | 1 | 2-4 |

Multi-hop and computation queries are where the agent wins decisively — these are the demo cases.

### 13.5 Evaluation metrics — defined precisely

**Task Completion Rate:**
- Create a test set of 20 queries with ground truth answers (manually written)
- Score: binary correct/incorrect per query
- Correct = answer contains the key fact(s) from ground truth
- Measure separately for: simple / multi-hop / computation query types

**Faithfulness (LLM-as-judge):**
```
Prompt to Gemini:
"Given this answer: [answer]
And these source chunks: [retrieved chunks]
Is every factual claim in the answer supported by the source chunks?
Reply: FAITHFUL or UNFAITHFUL, with reason."
```
Score = % of answers rated FAITHFUL. This is reproducible and explainable to hội đồng.

**Reasoning Steps:**
- Record the number of Thought-Action-Observation cycles per query
- Report: min, max, average
- Compare: baseline always = 1 step; agent = variable
- Show: agent uses MORE steps for harder queries (proves adaptive reasoning)

**Tool Selection Accuracy:**
- For each test query, define the "correct" tool sequence (manually annotated)
- Score: % of queries where the agent used the expected tools (order-independent)

---

## 14. Key Design Decisions

| Decision | Rationale |
|---|---|
| Self-implemented ReAct loop | Shows genuine understanding — not a framework black box |
| FastAPI + Pydantic | Tool schemas defined once as Pydantic models map directly to Gemini function-calling declarations — type-safe, self-documenting API |
| Gemini 3.1 Pro | Best-in-class reasoning + native Vision for OCR; default model for all reasoning/eval runs (flash-lite may be used only for quick local iteration, never for final eval or demo) |
| Qdrant Cloud | Dedicated vector database with strong metadata filtering, cloud-managed — no Docker required |
| Supabase (Postgres only) | Simple, cloud-managed relational store for conversation history; Auth/Storage intentionally unused |
| Semantic chunking (dual pattern set) | More precise retrieval than fixed-size splitting; handles both generic and VN legal-style headings |
| calculate_or_verify tool | Proves agent > RAG: handles computation, not just retrieval |
| asteval (not eval()) | Safe expression evaluation — no arbitrary code execution |
| Pre-loaded knowledge base | Focus is agent behavior, not file management UI |
| No auth / no multi-tenant | Eliminates web app scope — thesis is about the agent |
| No Docker anywhere | Both external services are cloud-managed free tiers — removes a demo-day failure point |
| Chain of thought display | Core contribution — transparency impossible with pure RAG |

---

*This file is the single source of truth for project structure and implementation decisions.*
*Update whenever architectural decisions change.*

---

## 15. Evaluation & Results — Detailed Specification

This section defines exactly how to evaluate the agent output to produce credible, defensible results.

### 15.1 Test Set Design (20 queries minimum)

Distribute across 3 query types:

| Type | Count | Description | Expected tool sequence |
|---|---|---|---|
| Simple factual | 8 | Single-document lookup | search → answer |
| Multi-hop | 7 | Requires cross-document reasoning | search + search or compare_sections |
| Computation | 5 | Requires formula retrieval + calculation | search + calculate_or_verify |

**Example test queries:**

Simple:
- "What is VinTech Corp's probationary period duration?"
- "How many days notice is required for resignation?"

Multi-hop:
- "Does the leave policy and the employment contract agree on the number of sick days allowed?"
- "What are the differences between the NDA terms for full-time employees vs vendors?"

Computation:
- "An employee who joined 8 months ago — how many annual leave days are they entitled to?"
- "If an employee works overtime 12 hours this month at 1.5x rate, what is the extra pay for a base salary of 20,000,000 VND?"

**Ground truth:** For each query, manually write the expected correct answer and identify the source document + section. Store in `evaluation/test-set.json`.

---

### 15.2 Automated Evaluation Script

Build `evaluation/run_eval.py`:

```python
# For each test query:
# 1. Run against ReAct Agent -> record answer, steps, tools used
# 2. Run against Baseline RAG -> record answer, steps (always 1)
# 3. Score correctness: LLM-as-judge comparing answer vs ground truth
# 4. Score faithfulness: LLM-as-judge comparing answer vs retrieved chunks
# 5. Write results to evaluation/results.json
```

**LLM-as-judge prompt for correctness:**
```
You are evaluating an AI system's answer.

Question: {question}
Ground truth answer: {ground_truth}
System answer: {system_answer}

Does the system answer correctly address the question and contain the key facts from the ground truth?
Respond with JSON: { "correct": true/false, "reason": "brief explanation" }
```

**LLM-as-judge prompt for faithfulness:**
```
You are evaluating whether an AI answer is grounded in its source documents.

Answer: {answer}
Source chunks used: {retrieved_chunks}

Is every factual claim in the answer supported by the source chunks?
Respond with JSON: { "faithful": true/false, "unsupported_claims": [...] }
```

---

### 15.3 Results Table (fill in after running eval)

**Overall Results:**

| Metric | Baseline RAG | ReAct Agent | Improvement |
|---|---|---|---|
| Task Completion Rate (all) | X% | Y% | +Z% |
| Task Completion — Simple | X% | Y% | +Z% |
| Task Completion — Multi-hop | X% | Y% | +Z% |
| Task Completion — Computation | 0% | Y% | +Y% |
| Answer Faithfulness | X% | Y% | +Z% |
| Avg Reasoning Steps | 1 | X.X | — |

**Per-query detail log:** Store in `evaluation/results.json` — show at least 5 representative queries in the thesis with full reasoning trace.

---

### 15.4 Self-correction Case Study

Document at least 1 case where the agent self-corrects:

```
Query: "What is VinTech's policy on remote work?"
Step 1 — search_knowledge_base("remote work policy") -> No results found
Step 2 — list_documents() -> Agent sees available docs
Step 3 — search_knowledge_base("work from home", "ops") -> Found 2 sections
Step 4 — Final Answer: [correct answer with citation]

Baseline RAG: "No relevant information found." <- INCORRECT
Agent: [correct answer] <- CORRECT via self-correction
```

This is the most powerful demo — show it in both the thesis and the live demo.

---

### 15.5 Computation Case Study

Document at least 1 computation query end-to-end:

```
Query: "Employee worked 8 months — annual leave entitlement?"

Baseline RAG:
  Retrieved: "Employees are entitled to 12 days of annual leave per year,
              pro-rated based on months worked."
  Answer: "12 days per year, pro-rated." <- INCOMPLETE -- no actual number

ReAct Agent:
  Thought: I need to find the leave formula, then calculate for 8 months.
  Action: search_knowledge_base("annual leave pro-rated formula", "hr")
  Observation: "12 days per year, pro-rated monthly = (months/12) x 12"
  Thought: Formula found. Now calculate for 8 months.
  Action: calculate_or_verify("(8/12)*12", "annual leave for 8 months")
  Observation: Result = 8.0
  Final Answer: "8 days" <- COMPLETE with citation and calculation shown
```

This directly answers "RAG cũng làm được mà, agent hơn gì?" — agent computes, RAG only retrieves.

---

### 15.6 Thesis Results Section Structure

Chapter: Results and Discussion should contain:

1. **Test set description** — 20 queries, distribution across types
2. **Overall results table** (Section 15.3 above)
3. **Self-correction case study** with full reasoning trace
4. **Computation case study** with baseline vs agent comparison
5. **Faithfulness analysis** — % faithful, any notable failures
6. **Discussion** — where the agent outperforms the baseline and why, known limitations
7. **Screenshots** — light-theme UI, chain of thought clearly readable