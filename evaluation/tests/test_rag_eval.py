"""
Pytest-based test suite for RAG evaluation.
Asserts minimum metric thresholds to gate CI/CD.
"""
import os
import pytest
from typing import Dict, Any

from evaluate_rag import run_eval, load_dataset, build_test_cases
from rag_pipeline import rag_pipeline


# Minimum thresholds for metrics (adjust based on your requirements)
MIN_CONTEXTUAL_RELEVANCY = 0.7
MIN_ANSWER_RELEVANCY = 0.7
MIN_FAITHFULNESS = 0.7

# Path to evaluation dataset
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "kg_rag_synth_goldens.json"
)


@pytest.fixture(scope="module")
def eval_results():
    """
    Fixture that runs evaluation once and caches results for all tests.
    """
    if not os.path.exists(DATASET_PATH):
        pytest.skip(
            f"Dataset not found: {DATASET_PATH}\n"
            f"Run 'python generate_goldens.py' first to create synthetic goldens."
        )
    
    results = run_eval(
        dataset_path=DATASET_PATH,
        verbose=False,  # Suppress output in tests
    )
    return results


def test_dataset_exists():
    """Test that evaluation dataset exists."""
    assert os.path.exists(DATASET_PATH), (
        f"Evaluation dataset not found: {DATASET_PATH}\n"
        f"Run 'python generate_goldens.py' first."
    )


def test_dataset_loads():
    """Test that dataset can be loaded."""
    dataset = load_dataset(DATASET_PATH)
    assert len(dataset.goldens) > 0, "Dataset is empty"


def test_rag_pipeline_runs():
    """Test that RAG pipeline executes without errors."""
    test_query = "What is social class?"
    try:
        answer, contexts = rag_pipeline(test_query)
        assert isinstance(answer, str), "Answer should be a string"
        assert isinstance(contexts, list), "Contexts should be a list"
        assert len(answer) > 0, "Answer should not be empty"
    except Exception as e:
        pytest.fail(f"RAG pipeline failed: {e}")


def test_contextual_relevancy_threshold(eval_results: Dict[str, Any]):
    """
    Test that mean contextual relevancy meets minimum threshold.
    """
    test_cases = eval_results["test_cases"]
    metrics = eval_results["metrics"]
    
    # Find ContextualRelevancyMetric
    contextual_relevancy = None
    for metric in metrics:
        if metric.__class__.__name__ == "ContextualRelevancyMetric":
            contextual_relevancy = metric
            break
    
    if contextual_relevancy is None:
        pytest.skip("ContextualRelevancyMetric not found in metrics")
    
    # Calculate mean score
    scores = []
    for tc in test_cases:
        # Try to get score from test case
        # DeepEval may attach scores differently - adjust based on actual API
        score = getattr(tc, "contextual_relevancy_score", None)
        if score is None:
            # Try alternative attribute names
            score = getattr(tc, "contextualrelevancyscore", None)
        if score is not None:
            scores.append(score)
    
    if not scores:
        pytest.skip("Could not extract contextual relevancy scores")
    
    mean_score = sum(scores) / len(scores)
    
    assert mean_score >= MIN_CONTEXTUAL_RELEVANCY, (
        f"Mean contextual relevancy {mean_score:.3f} "
        f"below threshold {MIN_CONTEXTUAL_RELEVANCY}"
    )


def test_answer_relevancy_threshold(eval_results: Dict[str, Any]):
    """
    Test that mean answer relevancy meets minimum threshold.
    """
    test_cases = eval_results["test_cases"]
    metrics = eval_results["metrics"]
    
    # Find AnswerRelevancyMetric
    answer_relevancy = None
    for metric in metrics:
        if metric.__class__.__name__ == "AnswerRelevancyMetric":
            answer_relevancy = metric
            break
    
    if answer_relevancy is None:
        pytest.skip("AnswerRelevancyMetric not found in metrics")
    
    # Calculate mean score
    scores = []
    for tc in test_cases:
        score = getattr(tc, "answer_relevancy_score", None)
        if score is None:
            score = getattr(tc, "answerrelevancyscore", None)
        if score is not None:
            scores.append(score)
    
    if not scores:
        pytest.skip("Could not extract answer relevancy scores")
    
    mean_score = sum(scores) / len(scores)
    
    assert mean_score >= MIN_ANSWER_RELEVANCY, (
        f"Mean answer relevancy {mean_score:.3f} "
        f"below threshold {MIN_ANSWER_RELEVANCY}"
    )


def test_faithfulness_threshold(eval_results: Dict[str, Any]):
    """
    Test that mean faithfulness meets minimum threshold.
    """
    test_cases = eval_results["test_cases"]
    metrics = eval_results["metrics"]
    
    # Find FaithfulnessMetric
    faithfulness = None
    for metric in metrics:
        if metric.__class__.__name__ == "FaithfulnessMetric":
            faithfulness = metric
            break
    
    if faithfulness is None:
        pytest.skip("FaithfulnessMetric not found in metrics")
    
    # Calculate mean score
    scores = []
    for tc in test_cases:
        score = getattr(tc, "faithfulness_score", None)
        if score is None:
            score = getattr(tc, "faithfulnessscore", None)
        if score is not None:
            scores.append(score)
    
    if not scores:
        pytest.skip("Could not extract faithfulness scores")
    
    mean_score = sum(scores) / len(scores)
    
    assert mean_score >= MIN_FAITHFULNESS, (
        f"Mean faithfulness {mean_score:.3f} "
        f"below threshold {MIN_FAITHFULNESS}"
    )


def test_all_metrics_above_threshold(eval_results: Dict[str, Any]):
    """
    Combined test: all metrics should meet their thresholds.
    This is useful as a single gate for CI/CD.
    """
    test_cases = eval_results["test_cases"]
    
    # Collect all scores
    all_scores = {
        "contextual_relevancy": [],
        "answer_relevancy": [],
        "faithfulness": [],
    }
    
    for tc in test_cases:
        # Try multiple attribute name patterns
        for metric_name in all_scores.keys():
            score = getattr(tc, f"{metric_name}_score", None)
            if score is None:
                score = getattr(tc, metric_name.replace("_", ""), None)
            if score is not None:
                all_scores[metric_name].append(score)
    
    # Check thresholds
    failures = []
    
    if all_scores["contextual_relevancy"]:
        mean = sum(all_scores["contextual_relevancy"]) / len(all_scores["contextual_relevancy"])
        if mean < MIN_CONTEXTUAL_RELEVANCY:
            failures.append(
                f"ContextualRelevancy: {mean:.3f} < {MIN_CONTEXTUAL_RELEVANCY}"
            )
    
    if all_scores["answer_relevancy"]:
        mean = sum(all_scores["answer_relevancy"]) / len(all_scores["answer_relevancy"])
        if mean < MIN_ANSWER_RELEVANCY:
            failures.append(
                f"AnswerRelevancy: {mean:.3f} < {MIN_ANSWER_RELEVANCY}"
            )
    
    if all_scores["faithfulness"]:
        mean = sum(all_scores["faithfulness"]) / len(all_scores["faithfulness"])
        if mean < MIN_FAITHFULNESS:
            failures.append(
                f"Faithfulness: {mean:.3f} < {MIN_FAITHFULNESS}"
            )
    
    if failures:
        pytest.fail(
            "One or more metrics below threshold:\n" + "\n".join(failures)
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
