"""
Synthetic Golden Generation from PDF using DeepEval Synthesizer.
Generates question-answer pairs from the source PDF for evaluation.
"""
import os
import json
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv
try:
    from deepeval.synthesizer import Synthesizer
    from deepeval.synthesizer.config import ContextConstructionConfig, FiltrationConfig
    from deepeval.dataset import Golden, EvaluationDataset
except ImportError:
    raise ImportError(
        "DeepEval not installed. Install with: pip install deepeval"
    )

from models.groq_llm import GroqLlama3
from models.embedding_model import STEmbeddingModel
from config import DOCUMENTS_DIR, DATA_DIR


# Default paths from config
DEFAULT_PDF_PATH = DOCUMENTS_DIR
DEFAULT_OUTPUT_PATH = os.path.join(DATA_DIR, "kg_rag_synth_goldens.json")


def extract_pdf_text(pdf_path: str) -> str:
    """
    Extract text from PDF file.
    Uses existing chunker2 module if available, otherwise falls back to pdfplumber.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        Extracted text as string
    """
    try:
        from chunker2 import chunk_pdf
        chunks = chunk_pdf(pdf_path)
        # Join all chunks into a single text
        return "\n\n".join(chunks)
    except Exception as e:
        print(f"Warning: Could not use chunker2, falling back to pdfplumber: {e}")
        import pdfplumber
        text_parts = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts)


def create_synthetic_goldens(
    pdf_path: Optional[str] = None,
    num_goldens: int = 10,
    output_path: Optional[str] = None,
    use_document_path: bool = True
) -> EvaluationDataset:
    """
    Generate synthetic golden question-answer pairs from PDF.
    
    Args:
        pdf_path: Path to PDF file. Uses DEFAULT_PDF_PATH if None.
        num_goldens: Number of synthetic goldens to generate
        output_path: Path to save the dataset. Uses DEFAULT_OUTPUT_PATH if None.
        use_document_path: If True, use Synthesizer's from_docs method.
                          If False, extract text first and use from_contexts.
    
    Returns:
        EvaluationDataset containing synthetic goldens
    """
    if pdf_path is None:
        pdf_path = DEFAULT_PDF_PATH
    
    if output_path is None:
        output_path = DEFAULT_OUTPUT_PATH
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}\n"
            f"Please ensure the PDF exists or update DEFAULT_PDF_PATH in this file."
        )
    
    print(f"📄 Loading PDF: {pdf_path}")
    print(f"🤖 Using Groq and Sentence-Transformers for synthesis...")
    
    # 1. Instantiate custom models
    llm = GroqLlama3()
    embedder = STEmbeddingModel()

    # 2. Configure filtration to use Groq as critic
    filtration_conf = FiltrationConfig(
        critic_model=llm,
    )

    # 3. Configure context construction
    context_conf = ContextConstructionConfig(
        embedder=embedder,
        critic_model=llm,
        max_contexts_per_document=3,
        max_context_length=5, # Reduce from 29 to 5 
        chunk_size=500
    )

    # 4. Instantiate Synthesizer
    synthesizer = Synthesizer(
        model=llm,
        filtration_config=filtration_conf
    )
    
    try:
        # Use only generate_goldens_from_docs as requested
        synthetic_goldens = synthesizer.generate_goldens_from_docs(
            document_paths=[pdf_path],
            context_construction_config=context_conf,
            include_expected_output=True
        )

        # Fallback: If from_docs returns 0 results, try manual extraction
        if not synthetic_goldens:
            print("⚠️  generate_goldens_from_docs returned 0 results. Attempting fallback with manual text extraction...")
            text = extract_pdf_text(pdf_path)
            if not text or not text.strip():
                raise ValueError("Could not extract text from PDF for fallback.")
            
            # Create contexts manually
            chunk_size = 2000
            contexts = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            print(f"   Extracted {len(contexts)} contexts manually.")
            
            synthetic_goldens = synthesizer.generate_goldens_from_contexts(
                contexts=contexts,
                include_expected_output=True,
                max_goldens_per_context=max(1, int(num_goldens / len(contexts)) + 1)
            )
        
        # Manually limit to desired number
        if len(synthetic_goldens) > num_goldens:
            synthetic_goldens = synthetic_goldens[:num_goldens]
            
    except Exception as e:
        # If all methods fail, provide helpful error
        raise NotImplementedError(
            f"DeepEval Synthesizer API error: {e}\n"
            f"Ensure GROQ_API_KEY is set and the PDF path is correct."
        ) from e
    
    # Create dataset
    dataset = EvaluationDataset(goldens=synthetic_goldens)
    
    # Save to disk
    # dataset.save(output_path) # AttributeError: 'EvaluationDataset' object has no attribute 'save'
    
    # Manual save to JSON
    data_to_save = []
    for golden in synthetic_goldens:
        # Serialize Golden objects (Pydantic models)
        if hasattr(golden, "model_dump"):
            data_to_save.append(golden.model_dump())
        elif hasattr(golden, "dict"):
            data_to_save.append(golden.dict())
        else:
            data_to_save.append(golden.__dict__)
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4)
        
    print(f"✅ Generated {len(synthetic_goldens)} synthetic goldens")
    print(f"💾 Saved to: {output_path}")
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic golden question-answer pairs from PDF"
    )
    parser.add_argument(
        "--pdf-path",
        type=str,
        default=r"D:\ML\my_rag\electronics-12-02175.pdf",
        help="Path to PDF file (default: Family and Social Class.pdf)"
    )
    parser.add_argument(
        "--num-goldens",
        type=int,
        default=5,
        help="Number of synthetic goldens to generate (default: 10)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output path for dataset JSON (default: data/kg_rag_synth_goldens.json)"
    )
    parser.add_argument(
        "--use-contexts",
        action="store_true",
        help="Extract text first instead of using document path directly"
    )
    
    args = parser.parse_args()
    
    create_synthetic_goldens(
        pdf_path=args.pdf_path,
        num_goldens=args.num_goldens,
        output_path=args.output_path,
        use_document_path=not args.use_contexts
    )