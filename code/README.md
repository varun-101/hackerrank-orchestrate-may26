# HackerRank Orchestrate — Support Ticket Agent

An AI agent that resolves real customer support tickets across three domains (HackerRank, Claude, Visa) using a grounded RAG pipeline. Built for the HackerRank Orchestrate hackathon (May 2026).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r code/requirements.txt

# 2. Set your API key
cp .env.example .env
# edit .env and add your DEEPSEEK_API_KEY

# 3. Build the retrieval index (one-time, ~2–3 min)
cd code
python build_index.py

# 4. Run the pipeline
python main.py                  # process all tickets
python main.py --limit 5        # first 5 tickets only (dev mode)
python main.py --verbose        # per-ticket status lines
```

Output is written to `support_tickets/output.csv`.

---

## Requirements

- Python 3.10+
- A [DeepSeek API key](https://platform.deepseek.com/) — the only required secret
- HuggingFace models are downloaded automatically on first run:
  - `all-MiniLM-L6-v2` (embedding)
  - `BAAI/bge-reranker-base` (cross-encoder reranker)

---

## Project Layout

```
.
├── code/
│   ├── main.py           # entry point — runs the full pipeline
│   ├── preprocessor.py   # Layer 1: guard rails, language detect, LLM classify
│   ├── retriever.py      # Layer 2: chunking, hybrid retrieval, reranker
│   ├── build_index.py    # offline index builder (run once)
│   ├── reasoner.py       # Layer 3: risk router + LLM response generation
│   ├── validator.py      # Layer 4: schema + hallucination guard
│   ├── area_resolver.py  # derives product_area from corpus file paths
│   ├── llm_client.py     # shared DeepSeek client with retry logic
│   ├── observability.py  # per-ticket timing traces
│   └── requirements.txt
├── data/
│   ├── hackerrank/       # HackerRank corpus (.md files)
│   ├── claude/           # Claude corpus (.md files)
│   ├── visa/             # Visa corpus (.md files)
│   └── index/            # generated index shards (created by build_index.py)
├── support_tickets/
│   ├── support_tickets.csv        # input
│   ├── sample_support_tickets.csv # ground-truth examples used as few-shot
│   └── output.csv                 # generated output
└── .env.example
```

---

## Architecture

The pipeline runs five sequential layers per ticket.

### Layer 0 — Ingestion

`main.py` reads `support_tickets.csv` row by row and fans each ticket into the L1→L4 pipeline. Every row is guaranteed an output — if the pipeline errors, a safe escalation is written so no row is silently dropped.

---

### Layer 1 — Pre-processing (`preprocessor.py`)

Four sub-components run in order before any retrieval or LLM call:

**1. Guard Rails (regex, no LLM)**

Regex patterns screen for prompt/command injection in English, French, and Spanish before the ticket reaches any generative model. Patterns include:
- `ignore all previous instructions`, `reveal your system prompt`, `act as`, `jailbreak`, `DAN mode`
- French equivalents: `affiche toutes les règles`, `logique interne`, `documents récupérés`
- Command injections: `rm -rf`, `drop table`, `delete all files`, `sudo`

Injected tickets are short-circuited to `status=escalated, request_type=invalid` — they never reach the LLM.

**2. Language Detection**

`langdetect` identifies the ISO-639-1 code. Deterministic (seed=42).

**3. English Normalisation**

Non-English tickets are translated to English via a DeepSeek call so retrieval and classification work on a uniform input. Guard rails re-run on the translated text to catch foreign-language injections that slipped through step 1.

**4. LLM Classification**

A single DeepSeek `deepseek-chat` call (JSON mode, temperature=0) returns:

| Field | Values |
|---|---|
| `domain` | `HackerRank` \| `Claude` \| `Visa` \| `ambiguous` |
| `request_type` | `product_issue` \| `feature_request` \| `bug` \| `invalid` |
| `risk_score` | `0.0–1.0` (fraud/identity theft → 0.9+, FAQ → 0.0) |
| `risk_reason` | one-sentence explanation |

A `company_hint` derived from the CSV's Company field is passed alongside the ticket to bias domain classification.

---

### Layer 2 — Retrieval (`retriever.py`, `build_index.py`)

**Offline indexing** (`build_index.py`, run once):

- All `.md` files under `data/{domain}/` are cleaned (YAML frontmatter stripped), chunked at ~512 tokens (2048 chars) with 50-token (200-char) overlap snapping to newlines.
- Each chunk is embedded with `all-MiniLM-L6-v2` (sentence-transformers), L2-normalised.
- A BM25 index (`rank-bm25`) is built over the same chunks.
- One `.pkl` shard per domain is saved under `data/index/`.

**Online retrieval** (called per ticket):

1. **Dense retrieval** — cosine similarity via dot product on normalised embeddings, top 20 candidates.
2. **BM25 retrieval** — keyword match, top 20 candidates.
3. **Hybrid merge** — `score = 0.7 × dense + 0.3 × BM25`, union of both candidate sets.
4. **Cross-encoder reranker** (`BAAI/bge-reranker-base`) — scores all candidates as query–chunk pairs, returns top 5.

For `domain=ambiguous`, all three shards are searched and merged before reranking.

---

### Layer 3 — Reasoning (`reasoner.py`)

**Risk Router (Gate 2)**

If `risk_score >= 1.0` or `is_injection=True`, the LLM is never called. A canned escalation message is returned immediately. This ensures fraud, stolen-card, and security-breach tickets never reach a generative model.

**Prompt Builder**

Assembles:
- A domain-specific system persona (HackerRank recruiter support / Claude support / Visa cardholder support)
- Hard rules: ground every claim in excerpts, never fabricate URLs/phone numbers/policies, escalate if unsure
- The top-5 retrieved corpus excerpts (labelled `[Excerpt 1..5 | source/path]`)
- 2–3 few-shot examples from `sample_support_tickets.csv` (one `product_issue/replied`, one `bug/escalated`, one `invalid/replied`)
- A strict JSON output schema for all six fields

**LLM Call**

`deepseek-chat` in `json_object` mode, temperature=0. Returns:

```json
{
  "status": "replied" | "escalated",
  "product_area": "<corpus subdirectory name>",
  "response": "<user-facing answer>",
  "justification": "<which excerpt supports this>",
  "request_type": "product_issue" | "feature_request" | "bug" | "invalid",
  "chunk_id": <1-N>
}
```

---

### Layer 4 — Output Validation (`validator.py`)

Three independent checks run before the result is written to CSV:

**1. Schema Validator** — coerces invalid enum values to safe defaults; escalates if the response body is too short (<20 chars).

**2. Consistency Checker** — flags escalated tickets whose response body is unusually long (>600 chars), surfacing potential misclassification for review.

**3. Hallucination Guard** — two-stage:
- *Heuristic pre-filter*: phone numbers, URLs, and 3+ digit numbers in the response are checked against the corpus verbatim. Any that don't appear trigger an LLM grounding call.
- *LLM grounding call*: a second DeepSeek call asks "is every factual claim in this response supported by these corpus excerpts?" If not grounded, the ticket is escalated with the unsupported claims noted in the justification.

**Product Area Resolution** (`area_resolver.py`)

After reasoning, the `product_area` is overridden with the parent directory name of the top-ranked corpus `.md` file. This makes `product_area` authoritative and consistent — it comes from the file system, not the LLM.

---

### Key Design Decisions

| Decision | Rationale |
|---|---|
| DeepSeek via OpenAI-compatible API | Cost-effective, JSON mode works reliably, swap-friendly |
| Domain-sharded indices | Prevents Visa docs surfacing for HackerRank tickets |
| Regex-only guard rails | LLM cannot be used to guard itself — regex is the safe choice |
| Two escalation gates (L1 risk score + L3 risk router) | High-risk tickets never reach the generative model |
| Cross-encoder reranker after hybrid retrieval | Dramatically improves precision vs. embedding-only retrieval |
| Corpus-path-derived `product_area` | Removes LLM guessing; the file system is the source of truth |
| Hallucination guard as final check | Better to over-escalate than to ship fabricated policy |

---

## Output Schema

| Field | Type | Values |
|---|---|---|
| `issue` | string | original ticket text |
| `subject` | string | ticket subject line |
| `company` | string | HackerRank / Claude / Visa |
| `response` | string | user-facing support reply |
| `product_area` | string | corpus subdirectory (e.g. `managing-tests`) |
| `status` | enum | `replied` \| `escalated` |
| `request_type` | enum | `product_issue` \| `feature_request` \| `bug` \| `invalid` |
| `justification` | string | which excerpt was used, or why it was escalated |
