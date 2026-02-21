# Knowledge Graph RAG Pipeline

A **Knowledge-Graph-based Retrieval-Augmented Generation (RAG)** system built on Neo4j, spaCy, and Groq LLMs, with a full evaluation harness powered by [DeepEval](https://docs.confident-ai.com/).

---

## Table of Contents

1. [Part 1 — How the RAG Pipeline Works](#part-1--how-the-rag-pipeline-works)
   - [System Overview](#system-overview)
   - [Phase 1: Document Ingestion](#phase-1-document-ingestion)
   - [Phase 2: Query & Generation](#phase-2-query--generation)
   - [Thread Isolation](#thread-isolation)
   - [Quick Start (Pipeline)](#quick-start-pipeline)
2. [Part 2 — The Testing Setup](#part-2--the-testing-setup)
   - [Architecture Overview](#architecture-overview)
   - [Step 1 — Generate Synthetic Goldens](#step-1--generate-synthetic-goldens)
   - [Step 2 — Run Evaluation](#step-2--run-evaluation)
   - [DeepEval Metrics: How They Work](#deepeval-metrics-how-they-work)
   - [Problems Encountered & Fixes Applied](#problems-encountered--fixes-applied)
   - [Quick Start (Evaluation)](#quick-start-evaluation)
3. [Tuning the Pipeline](#tuning-the-pipeline)
4. [Project Structure](#project-structure)
5. [Configuration Reference](#configuration-reference)

---

# Part 1 — How the RAG Pipeline Works

## System Overview

This system does not use vector similarity search. Instead, it builds a **symbolic knowledge graph** from the source document and traverses it at query time. This gives more interpretable retrieval — you can always trace which keyword matched which chunk.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     INGESTION PHASE (one-time)                      │
│                                                                     │
│  PDF File                                                           │
│     │                                                               │
│     ▼                                                               │
│  ┌──────────┐    sentence-aware sliding window (600 chars, 150 OL) │
│  │ Chunker  │──────────────────────────────────────────────────────▶│
│  └──────────┘                                                       │
│     │  List[{content, chunk_id}]                                   │
│     ▼                                                               │
│  ┌──────────────┐  spaCy NER + noun-chunks + YAKE + Regex          │
│  │ NER Extractor│──────────────────────────────────────────────────▶│
│  └──────────────┘                                                   │
│     │  Dict[keyword → List[chunk]]                                 │
│     ▼                                                               │
│  ┌────────────────┐  frequency filter (removes too-common keywords)│
│  │ Keyword Filter │──────────────────────────────────────────────── │
│  └────────────────┘                                                 │
│     │  Dict[filtered_keyword → List[chunk]]                        │
│     ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     Neo4j Knowledge Graph                    │  │
│  │                                                              │  │
│  │   (Keyword)──[:APPEARS_IN]──▶(Chunk)                        │  │
│  │       └─────[:SIMILAR_TO]────(Keyword)                      │  │
│  │                                                              │  │
│  │   All nodes tagged with thread_id for document isolation     │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      QUERY PHASE (per-request)                      │
│                                                                     │
│  User Query                                                         │
│     │                                                               │
│     ▼                                                               │
│  ┌────────────────────┐  Groq LLM filters query against all       │
│  │  Keyword Extractor │  graph keywords → matched keyword list     │
│  │  (groq_client.py)  │                                            │
│  └────────────────────┘                                            │
│     │  List[matched_keywords]                                      │
│     ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Graph Retriever                          │   │
│  │   1. Score chunks by # of matched keywords they share       │   │
│  │   2. Return top-scoring chunk(s) as "primary"               │   │
│  │   3. Expand neighborhood up to MAX_DEPTH hops:              │   │
│  │      • Shared-keyword neighbors                             │   │
│  │      • SIMILAR_TO keyword neighbors                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│     │  List[{id, content}]  retrieved chunks                      │
│     ▼                                                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                  Groq Answer Generator                       │  │
│  │  prompt = CONTEXT (chunks joined) + QUERY                    │  │
│  │  model fallback: tries GENERATION_MODELS in order            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│     │  answer: str                                                 │
│     ▼                                                               │
│  Response returned to caller                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Document Ingestion

### 1. Chunking (`chunker2.py`)

The PDF is parsed page-by-page with **pdfplumber**. The chunker is **genuinely sentence-aware** — it uses NLTK's `sent_tokenize` to split each page into a list of complete sentences first, then accumulates whole sentences until the character budget is consumed:

```
Page text
  → sent_tokenize()  →  ["Sentence 1.", "Sentence 2.", "Sentence 3 spans many words.", …]
       ↓
  Accumulate sentences until len(joined) > CHUNK_SIZE
       ↓
  Flush chunk  →  "Pg_no 3: Sentence 1. Sentence 2."
       ↓
  Carry back last N sentences whose total length ≥ CHUNK_OVERLAP  (overlap)
       ↓
  Start new chunk from overlap sentences + next sentence
```

This means:
- A sentence is **never cut in the middle** — boundaries always fall at a sentence end
- The overlap also carries back **complete sentences**, not raw character slices, so every new chunk begins at a clean grammatical boundary
- Each chunk gets an MD5 hash as its `chunk_id`, enabling exact deduplication of subset chunks

| Config | Default | Effect |
|--------|---------|--------|
| `CHUNK_SIZE` | 600 chars | Max characters per chunk. Larger → fewer, broader chunks; smaller → more, focused chunks |
| `CHUNK_OVERLAP` | 150 chars | Minimum tail-characters worth of whole sentences carried into the next chunk. Higher = better cross-boundary context continuity |

Each chunk is stored as `{"content": "Pg_no 3: …", "chunk_id": "<md5>"}`.  
Subset chunks (where one chunk's text is entirely contained in another) are deduplicated before returning.

### 2. Keyword / Entity Extraction (`ner_extractor.py`)

Three complementary extractors run on each chunk and their outputs are merged:

```
Chunk text
    ├── spaCy en_core_web_sm
    │       ├── Named entities  → typed routing:
    │       │       DATE    → normalize to YYYY-MM-DD
    │       │       NUMBER  → keep if ≥4 digits or 10–999
    │       │       PERSON / ORG / GPE / … → lemmatized noun tokens
    │       └── Noun chunks     → lemmatized noun tokens
    ├── YAKE statistical keywords  (disabled by default, easily enabled)
    └── Regex proper-names         (TitleCase, ALL-CAPS multi-word)
          ↓
    Normalize → Post-process → Validate → Deduplicate (substring removal)
          ↓
    Final keyword list
```

The result is a `Dict[keyword, List[chunk_content_str]]` — the keyword-to-chunks map.

### 3. Keyword Filtering (`keyword_filter.py`)

Keywords that appear in **too many chunks** (above a frequency threshold × total chunks) are dropped as uninformative stop-topics. This keeps the graph focused on distinctive concepts.

### 4. Knowledge Graph Construction (`graph_builder2.py`)

Three node/relationship types are written to Neo4j:

```cypher
(:Chunk   {id, content, thread_id})
(:Keyword {name, thread_id})
(:Keyword)-[:APPEARS_IN {thread_id}]->(:Chunk)
```

`thread_id` is attached to every node and relationship so multiple documents can coexist in one Neo4j database without interference.

---

## Phase 2: Query & Generation

### 5. Keyword Extraction at Query Time (`groq_client.py → extract_keywords`)

All `Keyword` names for the target `thread_id` are loaded from Neo4j. The user query is sent to the Groq LLM together with that full keyword list. The model returns only the keywords from that list that are relevant to the query. This avoids embedding models entirely.

### 6. Graph Retrieval (`graph_retriever2.py`)

**Primary retrieval** — a scored Cypher query:
```cypher
MATCH (k:Keyword {thread_id: $tid})-[:APPEARS_IN]→(c:Chunk {thread_id: $tid})
WHERE k.name IN $matched_keywords
WITH c, count(k) AS score
ORDER BY score DESC
LIMIT 1
-- then returns ALL chunks tied at the max score
```

**Neighborhood expansion** — BFS up to `MAX_DEPTH` hops using two link types:
- Chunks sharing a matched keyword (shared-keyword neighbors)
- Chunks reachable via `SIMILAR_TO` keyword edges

### 7. Answer Generation (`groq_client.py → generate_answer`)

Retrieved chunks are concatenated into one context block and sent to the Groq LLM with a structured prompt.  
A `GENERATION_MODELS` fallback list is tried in order if any model hits a rate limit.

---

## Thread Isolation

Every document ingested gets a `thread_id` (UUID or user-supplied string). All Neo4j nodes, relationships, and retrieval queries are scoped to this ID. This means:
- Multiple documents can live in one database simultaneously
- Clearing or re-ingesting one document does not affect others
- Evaluation runs can have their own isolated `thread_id`

---

## Quick Start (Pipeline)

```powershell
# 1. Start Neo4j
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Ingest a PDF
python rag_pipeline.py ingest "electronics-12-02175.pdf" --thread-id paper-001

# 4. Query it
python rag_pipeline.py query "How does LSTM compare to Informer?" --thread-id paper-001
```

---

# Part 2 — The Testing Setup

## Architecture Overview

```
                ┌─────────────────────┐
                │  Source PDF(s)      │
                └──────────┬──────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  generate_goldens.py   │  ← DeepEval Synthesizer
              │  (Groq LLM + ST       │    generates Q&A pairs
              │   Embedder)           │    from the document
              └────────────┬───────────┘
                           │ data/kg_rag_synth_goldens.json
                           ▼
              ┌────────────────────────┐
              │   evaluate_rag.py      │
              │                        │
              │  for each golden:      │
              │    query → RAG pipeline│  ← actual_output
              │    retrieval_context   │  ← retrieved chunks
              │                        │
              │  DeepEval evaluate()   │
              │  ┌────────────────┐    │
              │  │ContextualRel.  │    │  ← Groq as judge LLM
              │  │ AnswerRel.     │    │
              │  │ Faithfulness   │    │
              │  └────────────────┘    │
              └────────────────────────┘
                           │
                           ▼
                  Scores + Pass/Fail
                  printed to terminal
```

---

## Step 1 — Generate Synthetic Goldens

`generate_goldens.py` uses **DeepEval's Synthesizer** to automatically create realistic question-answer pairs from your source PDF, without needing manually authored test cases.

```powershell
python generate_goldens.py --pdf-path "electronics-12-02175.pdf" --num-goldens 20
# → saves to data/kg_rag_synth_goldens.json
```

Internally the synthesizer:
1. Chunks the PDF using the same `chunker2.py` used in ingestion
2. Uses `STEmbeddingModel` (sentence-transformers `all-MiniLM-L6-v2`) to group contexts thematically
3. Uses `GroqLlama3` as both the generator (writes questions + expected answers) and the critic (filters low-quality pairs via `FiltrationConfig`)

Golden format:
```json
{
  "input": "Analyze how integrating LSTM and Informer's strengths …",
  "expected_output": "Integrating LSTM and Informer models enhances …",
  "context": ["Electronics 2023, 12, 2175 …"]
}
```

You can also write goldens manually in the same JSON format — `evaluate_rag.py` loads them identically.

---

## Step 2 — Run Evaluation

```powershell
# Evaluate against the default golden dataset
python evaluate_rag.py

# Ingest a new PDF and evaluate in one step
python evaluate_rag.py --pdf-path electronics-12-02175.pdf --thread-id paper-001

# Evaluate against a specific golden file
python evaluate_rag.py --dataset-path data/debug_golden.json --thread-id paper-001
```

The evaluator:
1. Loads goldens from JSON
2. Runs each query through the full RAG pipeline to get `actual_output` and `retrieval_context`
3. Feeds each `LLMTestCase` through DeepEval metrics, using `GroqLlama3` as the **judge LLM**
4. Prints aggregate Mean / Min / Max scores and pass rates

---

## DeepEval Metrics: How They Work

All three metrics use a **jury/verdict system**: the judge LLM is asked to evaluate individual statements and return `{"verdict": "yes"/"no"}` for each. The final score is the fraction of "yes" verdicts.

### 1. Contextual Relevancy

> *"How much of the retrieved context is actually relevant to the query?"*

**What it tests:** Retrieval quality — are the chunks we return useful?

**Calculation:**
```
score = (# retrieved statements relevant to query) / (total # statements in retrieval_context)
```
The judge LLM reads each sentence/statement from the retrieved chunks and verdicts whether it helps answer the query.

**Why it matters for this pipeline:** Our graph retrieval can over-expand via neighborhood BFS. A low score means we're pulling in too many loosely-related chunks.

---

### 2. Answer Relevancy

> *"How relevant is the actual answer to the original query?"*

**What it tests:** Generation quality — does the LLM stay on topic?

**Calculation:**
```
score = (# statements in actual_output relevant to query) / (total # statements in actual_output)
```
The judge generates several "reverse questions" from the answer, then scores what fraction of them align with the original query.

**Why it matters:** Catches hallucinations about unrelated content or over-verbose answers that wander off topic.

---

### 3. Faithfulness

> *"Is everything in the answer actually supported by the retrieved context?"*

**What it tests:** Grounding quality — does the LLM make up facts not present in the retrieved chunks?

**Calculation:**
```
score = (# claims in actual_output supported by retrieval_context) / (total # claims in actual_output)
```
The judge decomposes the answer into atomic claims, then verdicts each claim against the context.

**Why it matters:** The most critical metric for a RAG system. A faithfulness score below 0.7 means the LLM is inventing information not grounded in your document.

---

### Score Thresholds

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| `ContextualRelevancyMetric` | 0.7 | ≥70% of retrieved content must be relevant |
| `AnswerRelevancyMetric` | 0.7 | ≥70% of the answer must address the query |
| `FaithfulnessMetric` | 0.7 | ≥70% of answer claims must be grounded in context |

---

## Problems Encountered & Fixes Applied

### Problem 1 — JSON Schema Support Not Universal

**Issue:** DeepEval metrics call `a_generate_with_schema(prompt, schema=SomePydanticModel)` which sends a `response_format: {type: "json_schema"}` request to Groq. Several Groq models do not support this format and return `BadRequestError`.

**Fix:** `EVAL_MODELS` in `config.py` was curated to only include models that support JSON schema output. The fallback list in `GroqLlama3` tries each model in sequence.

---

### Problem 2 — Groq Rate Limits Causing Silent Failures

**Issue:** The evaluation runs 3 metric judgements per test case, all firing concurrently. With multiple test cases, this quickly exceeded Groq's tokens-per-minute limit. Errors were silently swallowed and the pipeline appeared to hang.

**Fix:** Added explicit `RateLimitError` handling with linear back-off in `GroqLlama3`:
```python
except RateLimitError:
    wait = 15 * (attempt + 1)  # 15s, 30s, 45s…
    await asyncio.sleep(wait)
```
And a model fallback loop that tries the next model in `EVAL_MODELS` after exhausting retries.

---

### Problem 3 — `asyncio.Semaphore` in Wrong Event Loop (CancelledError → TimeoutError)

**Issue:** A module-level `asyncio.Semaphore` was created at import time, before DeepEval's internal event loop started. When DeepEval wrapped each test case in `asyncio.wait_for(coro, timeout=T)`, the cancellation from an expired deadline propagated into coroutines waiting on `semaphore.acquire()`, causing:
```
asyncio.exceptions.CancelledError → TimeoutError
```

This looked like a timeout problem but was actually a **semaphore-in-wrong-loop** problem. Even a "better positioned" semaphore wouldn't help — any `await`-ing lock inside DeepEval's cancellable context can be cancelled.

**Fix (3-part):**

| Step | Change | File |
|------|--------|------|
| 1 | Removed the semaphore entirely from `a_generate` | `models/groq_llm.py` |
| 2 | Delegated concurrency control to DeepEval's own `AsyncConfig(max_concurrent=2, throttle_value=1)` — this sits *above* the timeout wrapper | `evaluate_rag.py` |
| 3 | Added `DEEPEVAL_DISABLE_TIMEOUTS=1` to `.env` so rate-limit retries (`asyncio.sleep(N)`) are never cancelled by a deadline | `.env` |

---

### Problem 4 — `a_generate` Blocking the Event Loop

**Issue:** The original `a_generate` called the synchronous `generate` which used `time.sleep()` for rate-limit waits. Calling a sync-sleep inside an async context blocks the entire Python event loop, freezing all other concurrent metric evaluations.

**Fix:** Rewrote `a_generate` to use `AsyncGroq` (the async Groq client) and `await asyncio.sleep()` for all waits, making retries fully non-blocking.

---

## Quick Start (Evaluation)

```powershell
# Full pipeline: ingest + generate goldens + evaluate
python rag_pipeline.py ingest "electronics-12-02175.pdf" --thread-id paper-001
python generate_goldens.py --pdf-path "electronics-12-02175.pdf" --num-goldens 20
python evaluate_rag.py --thread-id paper-001

# Fast debug run (single golden, hand-crafted)
python evaluate_rag.py --dataset-path data/debug_golden.json --thread-id paper-001

# Pytest integration
pytest tests/test_rag_eval.py -v
```

---

# Project Structure

```
my_rag/
├── rag_pipeline.py          # Main entry point: ingest + query CLI
├── evaluate_rag.py          # DeepEval evaluation runner
├── generate_goldens.py      # Synthetic Q&A generation from PDF
│
├── chunker2.py              # PDF → sentence-aware overlapping chunks
├── ner_extractor.py         # spaCy + YAKE + Regex keyword extraction
├── keyword_filter.py        # Frequency-based keyword pruning
├── graph_builder2.py        # Neo4j graph builder (Chunk + Keyword nodes)
├── graph_retriever2.py      # Scored retrieval + BFS neighborhood expansion
├── groq_client.py           # Groq API: keyword extraction + answer generation
│
├── models/
│   ├── groq_llm.py          # DeepEval LLM wrapper (GroqLlama3 with async+fallback)
│   └── embedding_model.py   # DeepEval embedding wrapper (sentence-transformers)
│
├── config.py                # All configuration constants and env vars
├── docker-compose.yml       # Neo4j container
├── requirements.txt         # Python dependencies
│
├── data/
│   ├── kg_rag_synth_goldens.json   # Generated golden dataset
│   ├── debug_golden.json           # Hand-crafted debug golden
│   └── chunks.txt                  # Debug: last chunked output
│
└── tests/
    └── test_rag_eval.py     # Pytest test suite
```

---

# Tuning the Pipeline

Every parameter listed here can be set in `config.py` or overridden via an environment variable in `.env`. The defaults are conservative — they produce a small, precise graph. If you find answers are too thin, you can widen retrieval; if answers are noisy, tighten them.

---

## 1. Chunking Parameters

### `CHUNK_SIZE` (default: `600`)

Controls the maximum number of characters per chunk (measured in whole sentences).

| Smaller (e.g. 300) | Larger (e.g. 1200) |
|--------------------|--------------------|
| More granular chunks, more precise keyword-to-chunk mapping | Fewer, broader chunks — each chunk covers more ground |
| Good for question-answering over dense technical text | Good for narrative/story text where context is spread across paragraphs |
| Graph has more nodes → retrieval is more targeted | Graph has fewer nodes → retrieval is less pinpointed but faster |

### `CHUNK_OVERLAP` (default: `150`)

The minimum number of characters worth of tail sentences from the previous chunk that are carried into the next. Because overlap is measured in whole sentences, the actual carried text might be slightly more than this value.

- **Higher overlap** → context that straddles two chunks is preserved in both; answers are less likely to be missing information that appeared near a boundary
- **Lower overlap** → chunks are more independent; less redundancy in the graph

---

## 2. Keyword Frequency Filter — Why It Exists

The NER extractor (`ner_extractor.py`) uses the general-purpose `en_core_web_sm` spaCy model, which was **not fine-tuned for any specific domain**. As a result, it over-extracts: it picks up generic nouns like `"model"`, `"result"`, `"method"`, `"section"`, `"figure"` from every single chunk in a technical paper.

If these generic keywords are kept in the graph, they become **hub nodes** — connected to almost every chunk. When the query hits one of these keywords, the retriever returns a huge, unfocused set of chunks, degrading both precision and the faithfulness score.

```
Without filter (dense graph):
  "model" ──► chunk 1
           ──► chunk 2
           ──► chunk 3  ← almost everything connects to "model"
           ──► chunk 4     → retrieval returns too many chunks
           ──► chunk 5

With filter (sparse graph):
  "lstm informer" ──► chunk 7   ← only chunks specifically about this topic
  "attention mechanism" ──► chunk 12
```

### `FREQUENCY_THRESHOLD` (default: `0.03` = 3%)

A keyword is **removed** if it appears in more than `FREQUENCY_THRESHOLD × total_chunks` chunks.

At the default of 3%, a keyword that appears in just 4 out of 100 chunks is already dropped. This is intentionally aggressive.

| Lower (e.g. 0.01 = 1%) | Higher (e.g. 0.15 = 15%) |
|------------------------|---------------------------|
| Only ultra-specific keywords survive | Domain-level terms like `"neural network"` survive |
| Very sparse graph, very precise retrieval | Denser graph, more connections between chunks |
| Risk: missing relevant chunks that share a generic concept | Risk: hub nodes pollute retrieval with off-topic chunks |
| Good for: domain-specific docs with well-defined jargon | Good for: general-purpose or narrative documents |

**Effect on answer quality:** A lower threshold = more focused context passed to the LLM = higher Faithfulness score but potentially missing broader context. A higher threshold = more context = higher recall but potential hallucination from unrelated chunks.

### `KEEP_LIST` (default: `{}`)

A set of keyword strings that are **always kept**, regardless of how frequently they appear. Use this to protect domain-critical terms that the frequency filter would otherwise drop.

```python
# config.py  — example for a legal document pipeline
KEEP_LIST = {"liability", "negligence", "force majeure"}
```

---

## 3. Graph Retrieval Parameters

### `MAX_DEPTH` (default: `1`)

Controls how many BFS hops the retriever expands from the primary scored chunk(s).

```
MAX_DEPTH = 0:  Return only the highest-scoring primary chunk(s)
MAX_DEPTH = 1:  Primary chunks + their direct graph-neighbors (shared keywords)
MAX_DEPTH = 2:  Primary + neighbors + neighbors-of-neighbors
```

| Lower (`0`) | Higher (`2+`) |
|-------------|---------------|
| Very precise, minimal context | Broad context sweep, many chunks returned |
| High Faithfulness (all content is highly relevant) | Higher recall (edge cases are caught) |
| May miss supporting evidence in adjacent chunks | Risk of pulling in weakly-related chunks → lower Contextual Relevancy score |

**Practical guidance:** For short, focused queries use `MAX_DEPTH=0` or `1`. For multi-hop reasoning questions ("what are the implications of X on Y?") try `MAX_DEPTH=2`.

### `TOP_K_KEYWORDS` (default: `1`)

After the Groq LLM returns a list of matched keywords, `TOP_K_KEYWORDS` controls how many of those keywords are passed to the Cypher scoring query.

> ⚠️ **Note:** The current implementation in `graph_retriever2.py` passes the full matched keyword list to the scoring query and uses count-of-matching-keywords as the score. `TOP_K_KEYWORDS` is kept in config for a planned filter step. Changing it currently has no effect on retrieval — it is a placeholder for future ranked-keyword filtering.

---

## 4. Evaluation Concurrency

### `AsyncConfig(max_concurrent, throttle_value)` in `evaluate_rag.py`

These are set in code (not env vars) because they control DeepEval's internal async executor:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `max_concurrent` | `2` | Max test cases evaluated in parallel. Lower = fewer simultaneous Groq API calls |
| `throttle_value` | `1` | Seconds to wait before spawning the next test case task. Higher = gentler API ramp-up |

If you are hitting Groq rate limits during evaluation, **reduce `max_concurrent` to `1`** and **increase `throttle_value` to `3`** before increasing retry wait times.

---

# Configuration Reference

Key variables in `config.py` (all overridable via `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_SIZE` | 600 | Max chars per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap chars between chunks |
| `FREQUENCY_THRESHOLD` | 0.03 | Keyword filter: max fraction of chunks it may appear in |
| `TOP_K_KEYWORDS` | 1 | Top-k keywords used per retrieval query |
| `MAX_DEPTH` | 1 | BFS hops for neighborhood expansion |
| `EVAL_MODELS` | `[openai/gpt-oss-20b, …]` | Ordered fallback list for DeepEval judge |
| `GENERATION_MODELS` | `[llama-4-scout, …]` | Ordered fallback list for answer generation |
| `EVAL_LLM_MODEL_NAME` | First in `EVAL_MODELS` | Default evaluation model |
| `EVAL_EMBEDDER_MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding model for golden synthesis |
| `DEEPEVAL_DISABLE_TIMEOUTS` | `1` (set in .env) | Disables DeepEval's per-task deadline |
