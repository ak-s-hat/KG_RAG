# Testing and Metrics Guide

Complete guide for running tests and getting evaluation metrics for your RAG pipeline.

## 📋 Prerequisites Checklist

Before running tests, ensure all prerequisites are met:

### 1. ✅ Environment Setup

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Set environment variables** (create `.env` file or export):
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Neo4j (defaults shown - adjust if needed)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=whatever name you decide

# Optional: Evaluation thread ID
EVAL_THREAD_ID=eval-thread-default
```

### 2. ✅ Neo4j Database Running

**Start Neo4j:**
```bash
docker-compose up -d
```

**Verify it's running:**
```bash
docker ps
# Should show neo4j_db container running
```

**Check connection:**
```bash
# Test Neo4j connection (optional)
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('your_uid', 'your_password')); driver.verify_connectivity(); print('✅ Neo4j connected')"
```

### 3. ✅ Knowledge Graph Built

**Build the knowledge graph** (one-time setup):
```bash
python main.py
```

This will:
- Chunk your PDF
- Extract keywords
- Build Neo4j knowledge graph
- Use a `thread_id` (note this thread_id - you'll need it for evaluation)

**Important:** Note the `thread_id` printed by `main.py`. You'll need to set `EVAL_THREAD_ID` to match this, or the evaluation will use the default `eval-thread-default`.

### 4. ✅ Synthetic Goldens Generated

**Generate evaluation dataset:**
```bash
python generate_goldens.py
```

Options:
```bash
# Generate 50 goldens (default)
python generate_goldens.py

# Generate more goldens for better evaluation
python generate_goldens.py --num-goldens 100

# Specify custom paths
python generate_goldens.py --pdf-path "path/to/pdf.pdf" --output-path "data/my_goldens.json"
```

This creates `data/kg_rag_synth_goldens.json` with question-answer pairs.

---

## 🧪 Running Tests

### Option 1: Run All Tests (Recommended)

**Run the complete test suite:**
```bash
pytest tests/test_rag_eval.py -v
```

**Verbose output:**
```bash
pytest tests/test_rag_eval.py -v -s
```

**Run specific test:**
```bash
pytest tests/test_rag_eval.py::test_contextual_relevancy_threshold -v
```

### Option 2: Run Evaluation Script (Get Detailed Metrics)

**Get detailed metrics without assertions:**
```bash
python evaluate_rag.py
```

**With custom dataset:**
```bash
python evaluate_rag.py --dataset-path data/my_goldens.json
```

**With specific thread_id:**
```bash
python evaluate_rag.py --thread-id your-thread-id-here
```

**Quiet mode (less output):**
```bash
python evaluate_rag.py --quiet
```

---

## 📊 Understanding Metrics

### Metrics Explained

1. **ContextualRelevancyMetric** (Target: ≥ 0.7)
   - Measures: How relevant are the retrieved chunks to the query?
   - Score range: 0.0 to 1.0
   - Higher = better retrieval quality

2. **AnswerRelevancyMetric** (Target: ≥ 0.7)
   - Measures: How relevant is the generated answer to the query?
   - Score range: 0.0 to 1.0
   - Higher = answer better addresses the question

3. **FaithfulnessMetric** (Target: ≥ 0.7)
   - Measures: Is the answer faithful to the retrieved contexts (no hallucination)?
   - Score range: 0.0 to 1.0
   - Higher = less hallucination, more grounded in context

### Reading Test Output

**When running `pytest tests/test_rag_eval.py -v`:**
```
tests/test_rag_eval.py::test_dataset_exists PASSED
tests/test_rag_eval.py::test_dataset_loads PASSED
tests/test_rag_eval.py::test_rag_pipeline_runs PASSED
tests/test_rag_eval.py::test_contextual_relevancy_threshold PASSED
tests/test_rag_eval.py::test_answer_relevancy_threshold PASSED
tests/test_rag_eval.py::test_faithfulness_threshold PASSED
tests/test_rag_eval.py::test_all_metrics_above_threshold PASSED
```

**When running `python evaluate_rag.py`:**
```
============================================================
EVALUATION RESULTS
============================================================

ContextualRelevancyMetric:
  Mean Score: 0.823
  Min Score:  0.650
  Max Score:  0.950
  Pass Rate:  85.0%

AnswerRelevancyMetric:
  Mean Score: 0.789
  Min Score:  0.620
  Max Score:  0.920
  Pass Rate:  78.0%

FaithfulnessMetric:
  Mean Score: 0.856
  Min Score:  0.710
  Max Score:  0.980
  Pass Rate:  92.0%
============================================================
```

---

## 🔧 Adjusting Thresholds

**Edit `tests/test_rag_eval.py`** to change minimum thresholds:

```python
# Minimum thresholds for metrics
MIN_CONTEXTUAL_RELEVANCY = 0.7  # Change to 0.8 for stricter tests
MIN_ANSWER_RELEVANCY = 0.7      # Change to 0.75 for stricter tests
MIN_FAITHFULNESS = 0.7          # Change to 0.8 for stricter tests
```

---

## 🚀 Complete Workflow Example

**Step-by-step execution:**

```bash
# 1. Ensure Neo4j is running
docker-compose up -d

# 2. Build knowledge graph (note the thread_id)
python main.py
# Output: 📌 Thread ID: abc123-def456-...

# 3. Set thread_id for evaluation (optional, if different from default)
export EVAL_THREAD_ID=abc123-def456-...

# 4. Generate synthetic goldens
python generate_goldens.py --num-goldens 50

# 5. Run evaluation to see metrics
python evaluate_rag.py

# 6. Run tests to verify thresholds
pytest tests/test_rag_eval.py -v
```

---

## 🐛 Troubleshooting

### Error: "Dataset not found"
**Solution:** Run `python generate_goldens.py` first

### Error: "Neo4j connection failed"
**Solution:** 
- Check Neo4j is running: `docker ps`
- Verify credentials in `.env` file
- Test connection: `docker-compose logs neo4j`

### Error: "No keywords extracted from query"
**Solution:** 
- Ensure knowledge graph is built for the correct `thread_id`
- Check that `EVAL_THREAD_ID` matches the thread_id used in `main.py`

### Error: "GEMINI_API_KEY not found"
**Solution:**
- Set `GEMINI_API_KEY` in `.env` file
- Or export: `export GEMINI_API_KEY=your_key`

### Tests failing with low scores
**Solution:**
- Check if knowledge graph has data: Query Neo4j browser
- Verify PDF was processed correctly
- Consider adjusting thresholds if scores are consistently low
- Review retrieved contexts: Add debug prints in `rag_pipeline.py`

---

## 📈 Interpreting Results

### Good Performance
- All metrics ≥ 0.7: ✅ RAG pipeline is performing well
- Mean scores ≥ 0.8: ✅ Excellent performance
- Pass rate ≥ 90%: ✅ Very consistent

### Needs Improvement
- Any metric < 0.7: ⚠️ Review retrieval or generation
- Mean scores 0.5-0.7: ⚠️ Consider tuning:
  - Keyword extraction
  - Graph traversal depth (MAX_DEPTH)
  - Chunk size/overlap
  - Prompt engineering

### Poor Performance
- Any metric < 0.5: ❌ Significant issues
- Check:
  - Knowledge graph completeness
  - Query-keyword matching
  - Context retrieval quality
  - Answer generation prompts

---

## 🔄 CI/CD Integration

**Example GitHub Actions workflow:**

```yaml
name: RAG Evaluation

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Start Neo4j
        run: docker-compose up -d
      - name: Build knowledge graph
        run: python main.py
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
      - name: Generate goldens
        run: python generate_goldens.py --num-goldens 20
      - name: Run tests
        run: pytest tests/test_rag_eval.py -v
        env:
          EVAL_THREAD_ID: ${{ github.run_id }}
```

---

## 📝 Quick Reference

| Command | Purpose |
|---------|---------|
| `python generate_goldens.py` | Generate synthetic Q&A pairs |
| `python evaluate_rag.py` | Get detailed metrics |
| `pytest tests/test_rag_eval.py -v` | Run tests with assertions |
| `pytest tests/test_rag_eval.py::test_all_metrics_above_threshold -v` | Run single test |

---

## 🎯 Next Steps

1. **Baseline Evaluation**: Run with default settings to establish baseline
2. **Iterate**: Adjust RAG pipeline based on low-scoring metrics
3. **Monitor**: Run tests regularly to catch regressions
4. **Optimize**: Tune thresholds based on your domain requirements
