"""
RAG Pipeline Entrypoint for DeepEval Integration.
Wraps the existing Neo4j-based knowledge graph RAG system.
"""
import os
import uuid
from typing import List, Tuple, Optional
from neo4j import GraphDatabase

import sys
from pathlib import Path

# Add root directory to sys.path to allow running this script directly
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)
from retriever.graph_retriever2 import GraphRetriever
from generator.groq_client import generate_answer
from ingestion.chunker2 import chunk_pdf
from ingestion.ner_extractor import map_keywords_to_chunks
from ingestion.keyword_filter import filter_keys
from ingestion.graph_builder2 import KnowledgeGraphBuilder


# Global thread_id for evaluation - in production this would come from frontend
# For evaluation, we use a fixed thread_id to ensure consistent graph access
EVAL_THREAD_ID = os.getenv("EVAL_THREAD_ID", "eval-thread-default")


def retrieve_from_kg(query: str, thread_id: Optional[str] = None) -> List[str]:
    """
    Retrieve relevant chunks from Neo4j knowledge graph for a given query.
    
    Args:
        query: User query string
        thread_id: Optional thread_id for graph isolation. Uses EVAL_THREAD_ID if None.
    
    Returns:
        List of chunk content strings (retrieval context)
    """
    if thread_id is None:
        thread_id = EVAL_THREAD_ID
    
    retriever = GraphRetriever(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_pass=NEO4J_PASSWORD,
        neo4j_db=NEO4J_DATABASE,
        thread_id=thread_id
    )
    
    try:
        retrieved_chunks = retriever.retrieve(query)
        # Extract content strings from chunk dictionaries
        contexts = [chunk["content"] for chunk in retrieved_chunks]
        return contexts
    finally:
        retriever.close()


def generate_with_groq(query: str, contexts: List[str]) -> str:
    """
    Generate answer using Groq API with retrieved contexts.
    
    Args:
        query: User query string
        contexts: List of context strings retrieved from knowledge graph
    
    Returns:
        Generated answer string
    """
    # Convert contexts to the format expected by generate_answer
    # (list of dicts with 'content' key)
    chunks = [{"content": ctx} for ctx in contexts]
    answer = generate_answer(query, chunks)
    return answer


def rag_pipeline(query: str, thread_id: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Complete RAG pipeline: retrieve from KG and generate answer.
    
    This is the main entrypoint for DeepEval evaluation.
    
    Args:
        query: User query string
        thread_id: Optional thread_id for graph isolation. Uses EVAL_THREAD_ID if None.
    
    Returns:
        Tuple of (answer: str, retrieval_contexts: List[str])
        - answer: Generated answer from Groq
        - retrieval_contexts: List of chunk content strings used for generation
    """
    contexts = retrieve_from_kg(query, thread_id=thread_id)
    answer = generate_with_groq(query, contexts)
    return answer, contexts


def ingest_document(pdf_path: str, thread_id: str):
    """
    Ingest a PDF document into the knowledge graph with a specific thread_id.
    
    Args:
        pdf_path: Path to the PDF file.
        thread_id: Unique identifier for this document/session.
    """
    print(f"🚀 Starting ingestion for thread_id={thread_id} from {pdf_path}")
    
    # 1. Chunk PDF
    chunks = chunk_pdf(pdf_path)
    print(f"✅ Loaded {len(chunks)} chunks.")

    # 2. Extract keywords
    key_chunk_map = map_keywords_to_chunks(chunks)
    print(f"✅ Extracted {len(key_chunk_map.keys())} unique keywords")
    
    # 3. Filter keywords
    filtered_map = filter_keys(key_chunk_map, len(chunks))
    print(f"✅ Filtered to {len(filtered_map.keys())} unique keywords")

    # 4. Build Knowledge Graph
    kg = KnowledgeGraphBuilder()
    kg.clear_graph(thread_id)
    kg.build_graph_from_map(filtered_map, thread_id)
    kg.close()
    print(f"🎉 Knowledge graph built successfully for thread_id={thread_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG Pipeline CLI - query the knowledge graph or ingest new documents."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- query sub-command ---
    query_parser = subparsers.add_parser("query", help="Run a RAG query against the knowledge graph.")
    query_parser.add_argument("text", type=str, help="Query string to send to the RAG pipeline.")
    query_parser.add_argument(
        "--thread-id", type=str, default=None,
        help="Thread ID to scope graph retrieval (default: EVAL_THREAD_ID env var)."
    )

    # --- ingest sub-command ---
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a PDF document into the knowledge graph.")
    ingest_parser.add_argument("pdf_path", type=str, help="Path to the PDF file to ingest.")
    ingest_parser.add_argument(
        "--thread-id", type=str, default=None,
        help="Thread ID to tag the ingested document (auto-generated UUID if not provided)."
    )

    args = parser.parse_args()

    if args.command == "query":
        answer, contexts = rag_pipeline(args.text, thread_id=args.thread_id)
        print(f"\nQuery: {args.text}")
        print(f"\nRetrieved {len(contexts)} context chunk(s)")
        print(f"\nAnswer:\n{answer}")

    elif args.command == "ingest":
        thread_id = args.thread_id
        if thread_id is None:
            thread_id = str(uuid.uuid4())
            print(f"🆔 No --thread-id provided. Using generated ID: {thread_id}")
        ingest_document(args.pdf_path, thread_id)
        print(f"\n✅ Ingestion complete. Use --thread-id {thread_id} to query this document.")

