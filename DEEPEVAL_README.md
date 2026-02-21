# DeepEval Integration for Neo4j Knowledge Graph RAG

This directory contains the DeepEval integration for evaluating the Neo4j-based knowledge graph RAG pipeline.

## Setup

### 1. Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Required: Gemini API Key for LLM evaluation and synthesis
GEMINI_API_KEY=your_gemini_api_key_here

# Neo4j Configuration (defaults shown)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=hello-world
NEO4J_DATABASE=neo4j

# Optional: Evaluation thread ID (for graph isolation)
EVAL_THREAD_ID=eval-thread-default
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install DeepEval, pytest, and all other required dependencies.

### 3. Ensure Neo4j is Running

Start Neo4j using Docker Compose:

```bash
docker-compose up -d
```

Verify it's running:

```bash
docker ps
```

## Usage

### Step 1: Generate Synthetic Goldens

Generate synthetic question-answer pairs from your PDF:

```bash
python generate_goldens.py
```

Options:
- `--pdf-path`: Path to PDF file (default: `Family and Social Class.pdf`)
- `--num-goldens`: Number of goldens to generate (default: 50)
- `--output-path`: Output path for dataset JSON (default: `data/kg_rag_synth_goldens.json`)

Example:
```bash
python generate_goldens.py --num-goldens 100 --output-path data/my_goldens.json
```

### Step 2: Evaluate RAG Pipeline

Run evaluation on your RAG pipeline:

```bash
python evaluate_rag.py
```

Options:
- `--dataset-path`: Path to evaluation dataset (default: `data/kg_rag_synth_goldens.json`)
- `--thread-id`: Thread ID for graph isolation
- `--quiet`: Suppress verbose output

Example:
```bash
python evaluate_rag.py --dataset-path data/my_goldens.json
```

### Step 3: Run Tests (CI/CD Gate)

Run pytest-based tests that assert minimum metric thresholds:

```bash
pytest tests/test_rag_eval.py -v
```

Or run all tests:

```bash
pytest tests/ -v
```

## Files Overview

- **`models.py`**: Gemini model configuration for DeepEval
- **`rag_pipeline.py`**: RAG pipeline entrypoint that wraps existing Neo4j KG-RAG
- **`generate_goldens.py`**: Synthetic golden generation from PDF using DeepEval Synthesizer
- **`evaluate_rag.py`**: Evaluation script using DeepEval metrics (ContextualRelevancy, AnswerRelevancy, Faithfulness)
- **`tests/test_rag_eval.py`**: Pytest test suite with metric threshold assertions

## Metrics

The evaluation uses three DeepEval metrics:

1. **ContextualRelevancyMetric**: Measures how relevant retrieved contexts are to the query
2. **AnswerRelevancyMetric**: Measures how relevant the answer is to the query
3. **FaithfulnessMetric**: Measures whether the answer is faithful to the retrieved contexts (no hallucination)

Default thresholds (configurable in `tests/test_rag_eval.py`):
- Contextual Relevancy: ≥ 0.7
- Answer Relevancy: ≥ 0.7
- Faithfulness: ≥ 0.7

## Workflow

1. **Preprocessing** (one-time): Build knowledge graph from PDF
   ```bash
   python main.py  # This builds the KG for a thread_id
   ```

2. **Generate Goldens**: Create synthetic Q&A pairs
   ```bash
   python generate_goldens.py
   ```

3. **Evaluate**: Run evaluation on RAG pipeline
   ```bash
   python evaluate_rag.py
   ```

4. **Test**: Run pytest to gate CI/CD
   ```bash
   pytest tests/test_rag_eval.py
   ```

## Troubleshooting

### DeepEval API Changes

If you encounter import errors or API mismatches, check the [DeepEval documentation](https://docs.confident-ai.com/) for the latest API. Common adjustments may be needed for:

- `GeminiModel` import path
- `Synthesizer` method names (`generate_goldens_from_docs` vs `generate_goldens_from_contexts`)
- Metric class names or initialization parameters

### Thread ID Issues

The evaluation uses a fixed `thread_id` (default: `eval-thread-default`) to ensure consistent graph access. Make sure:

1. The knowledge graph has been built for this thread_id (run `main.py` first)
2. Or set `EVAL_THREAD_ID` environment variable to match your graph's thread_id

### Missing Dataset

If you get "Dataset not found" errors:

1. Run `python generate_goldens.py` first
2. Or ensure the PDF file exists at the expected path
3. Check that the output directory (`data/`) exists and is writable

## Integration with Existing Code

The DeepEval integration wraps your existing RAG components:

- **`graph_retriever2.py`**: Used by `rag_pipeline.py` for Neo4j retrieval
- **`gemini_client.py`**: Used by `rag_pipeline.py` for answer generation
- **`config.py`**: Used for Neo4j and Gemini configuration

No changes to your existing codebase are required - the integration is modular and non-invasive.
