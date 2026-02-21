"""
RAG Evaluation Script using DeepEval Metrics.
Evaluates the Neo4j-based knowledge graph RAG pipeline.
"""
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    from deepeval.dataset import Golden, EvaluationDataset
    from deepeval.test_case import LLMTestCase
    from deepeval import evaluate
    from deepeval.evaluate import AsyncConfig
    from deepeval.metrics import (
        ContextualRelevancyMetric,
        AnswerRelevancyMetric,
        FaithfulnessMetric,
    )
except ImportError:
    raise ImportError(
        "DeepEval not installed. Install with: pip install deepeval"
    )

from models.groq_llm import GroqLlama3
from rag_pipeline import rag_pipeline, ingest_document


DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "kg_rag_synth_goldens.json"
)


def load_dataset(path: Optional[str] = None) -> EvaluationDataset:
    """
    Load evaluation dataset from JSON file.
    
    Args:
        path: Path to dataset JSON file. Uses DEFAULT_DATASET_PATH if None.
    
    Returns:
        EvaluationDataset instance
    """
    if path is None:
        path = DEFAULT_DATASET_PATH
    
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            f"Please run generate_goldens.py first to create synthetic goldens."
        )
    
    try:
        return EvaluationDataset.load(path)
    except (AttributeError, Exception):
        print("⚠️  EvaluationDataset.load failed, attempting manual load...")
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        goldens = []
        for item in data:
            # Handle different serialization formats
            if "input" in item and "expected_output" in item:
                goldens.append(Golden(**item))
            elif "query" in item: # fallback if keys are different
                goldens.append(Golden(input=item["query"], expected_output=item.get("expected_output")))
        
        return EvaluationDataset(goldens=goldens)


def build_test_cases(
    dataset: EvaluationDataset,
    thread_id: Optional[str] = None
) -> List[LLMTestCase]:
    """
    Build LLMTestCase objects from evaluation dataset by running RAG pipeline.
    
    Args:
        dataset: EvaluationDataset containing goldens
        thread_id: Optional thread_id for graph isolation
    
    Returns:
        List of LLMTestCase objects with actual outputs from RAG pipeline
    """
    test_cases = []
    
    print(f"🔄 Building test cases from {len(dataset.goldens)} goldens...")
    
    for i, golden in enumerate(dataset.goldens, 1):
        query = golden.input
        expected = getattr(golden, "expected_output", None)
        
        if expected is None:
            print(f"⚠️  Warning: Golden {i} has no expected_output, skipping...")
            continue
        
        print(f"  [{i}/{len(dataset.goldens)}] Processing: {query[:60]}...")
        
        # Run RAG pipeline
        try:
            actual_output, retrieved_contexts = rag_pipeline(query, thread_id=thread_id)
        except Exception as e:
            print(f"  ❌ Error processing query: {e}")
            # Create a test case with error message
            actual_output = f"Error: {str(e)}"
            retrieved_contexts = []
        
        # Create test case
        tc = LLMTestCase(
            input=query,
            actual_output=actual_output,
            retrieval_context=retrieved_contexts,
            expected_output=expected,
        )
        test_cases.append(tc)
    
    print(f"✅ Built {len(test_cases)} test cases")
    return test_cases


def run_eval(
    dataset_path: Optional[str] = None,
    pdf_path: Optional[str] = None,
    thread_id: Optional[str] = None,
    metrics: Optional[List] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run evaluation on RAG pipeline using DeepEval metrics.
    
    Args:
        dataset_path: Path to evaluation dataset JSON
        thread_id: Optional thread_id for graph isolation
        metrics: Optional list of metric instances. Uses default metrics if None.
        verbose: Whether to print detailed results
    
    Returns:
        Dictionary containing evaluation results
    """
    # Load dataset
    dataset = load_dataset(dataset_path)

    # Ingest document if provided
    if pdf_path:
        # Generate a unique thread_id if not provided to avoid conflicts
        if thread_id is None:
            import uuid
            thread_id = str(uuid.uuid4())
            print(f"🆔 Generated new thread_id for evaluation: {thread_id}")
        
        print(f"📥 Ingesting document: {pdf_path}")
        try:
            ingest_document(pdf_path, thread_id)
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            raise e
    
    # Build test cases
    test_cases = build_test_cases(dataset, thread_id=thread_id)
    
    if not test_cases:
        raise ValueError("No test cases generated. Check your dataset.")
    
    # Initialize metrics if not provided
    if metrics is None:
        judge_llm = GroqLlama3()
        
        metrics = [
            ContextualRelevancyMetric(
                model=judge_llm,
                threshold=0.7,
            ),
            AnswerRelevancyMetric(
                model=judge_llm,
                threshold=0.7,
            ),
            FaithfulnessMetric(
                model=judge_llm,
                threshold=0.7,
            ),
        ]
    
    print(f"\n📊 Running evaluation with {len(metrics)} metrics...")
    print(f"   Metrics: {[m.__class__.__name__ for m in metrics]}\n")
    
    # Run evaluation
    # max_concurrent=2: only 2 test cases evaluated in parallel (reduces Groq burst)
    # throttle_value=1: 1-second gap between spawning new test case tasks
    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        async_config=AsyncConfig(max_concurrent=2, throttle_value=1),
    )
    
    # Print results
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Aggregate scores by metric
        metric_scores = {}
        for metric in metrics:
            metric_name = metric.__class__.__name__
            scores = [
                getattr(tc, f"{metric_name.lower()}_score", None)
                for tc in test_cases
            ]
            scores = [s for s in scores if s is not None]
            if scores:
                metric_scores[metric_name] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "scores": scores,
                }
        
        for metric_name, stats in metric_scores.items():
            print(f"\n{metric_name}:")
            print(f"  Mean Score: {stats['mean']:.3f}")
            print(f"  Min Score:  {stats['min']:.3f}")
            print(f"  Max Score:  {stats['max']:.3f}")
            print(f"  Pass Rate:  {sum(1 for s in stats['scores'] if s >= 0.7) / len(stats['scores']) * 100:.1f}%")
        
        print("\n" + "="*60)
    
    return {
        "test_cases": test_cases,
        "metrics": metrics,
        "results": results,
        "metric_scores": metric_scores if verbose else None,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate RAG pipeline using DeepEval metrics"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to evaluation dataset JSON (default: data/kg_rag_synth_goldens.json)"
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=None,
        help="Path to PDF file to ingest before evaluation (e.g. electronics-12-02175.pdf)"
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="Thread ID for graph isolation (default: uses EVAL_THREAD_ID env var or eval-thread-default)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    run_eval(
        dataset_path=args.dataset_path,
        pdf_path=args.pdf_path,
        thread_id=args.thread_id,
        verbose=not args.quiet,
    )
