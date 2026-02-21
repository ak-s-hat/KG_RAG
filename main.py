import os
from time import time
import uuid

from config import DOCUMENTS_DIR, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
from chunker2 import chunk_pdf
from ner_extractor import map_keywords_to_chunks
from keyword_filter import filter_keys
from graph_builder2 import KnowledgeGraphBuilder
from graph_retriever2 import GraphRetriever
from gemini_client import generate_answer
import torch


def verify_gpu():
    if torch.cuda.is_available():
        print("✅ PyTorch detected CUDA GPU")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("❌ No GPU detected; running on CPU")


if __name__ == "__main__":
    start_time = time()
    verify_gpu()

    # Generate thread_id for this session
    # In production, this should come from the frontend when a new chat is opened
    thread_id = str(uuid.uuid4())
    print(f"📌 Thread ID: {thread_id}\n")

    # --- 1. Chunk PDF ---ok 
    # Use the PDF file in the root directory
    pdf_path = os.path.join(os.path.dirname(__file__), "Family and Social Class.pdf")
    if not os.path.exists(pdf_path):
        # Fallback to DOCUMENTS_DIR if PDF not found in root
        pdf_path = os.path.join(DOCUMENTS_DIR, "sample.pdf")
    chunks = chunk_pdf(pdf_path)
    print(f"Loaded {len(chunks)} chunks.")

    # --- 2. Extract keywords from entire doc ---
    key_chunk_map = map_keywords_to_chunks(chunks)
    print(f"NER keywords {len(key_chunk_map.keys())} unique keywords/entities")
    filtered_map = filter_keys(key_chunk_map,len(chunks))
    print(f"Filterd {len(filtered_map.keys())} unique keywords/entities")
    keywords = sorted(filtered_map.keys())
    # for key in keywords:
    #     print(key)

    # --- 3. Build Knowledge Graph ---
    kg = KnowledgeGraphBuilder()
    kg.clear_graph(thread_id)
    print("Cleared existing graph.")
    kg.build_graph_from_map(filtered_map, thread_id)
    kg.close()
    print("Knowledge graph built successfully.")
    print(f"\nTotal Preprocessing: {time() - start_time:.2f} seconds")

    # --- 4. Query the Graph using semantic retriever ---    
    retriever = GraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, thread_id)
    query_text = "how do social classes affect the lives of individuals?"
    retrieved_chunks = retriever.retrieve(query_text)

    print(f"\nRetrieved {len(retrieved_chunks)} chunks for query: '{query_text}'\n")
    # for c in retrieved_chunks:
    #     print(f"Chunk ID: {c['id']}\nContent: {c['content']}\n---")
    retriever.close()

    # --- 5. Augmented Generation ---
    answer = generate_answer(query_text,retrieved_chunks)
    print('----------')
    print('ANSWER:')
    print(answer)
    print('----------')

     # --- 4. Query the Graph using semantic retriever ---    
    retriever = GraphRetriever(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE, thread_id)
    query_text = "based on the last question help me device a marketing strategy to target the elderly population."
    retrieved_chunks = retriever.retrieve(query_text)

    print(f"\nRetrieved {len(retrieved_chunks)} chunks for query: '{query_text}'\n")
    # for c in retrieved_chunks:
    #     print(f"Chunk ID: {c['id']}\nContent: {c['content']}\n---")
    retriever.close()

    # --- 5. Augmented Generation ---
    answer = generate_answer(query_text,retrieved_chunks)
    print('----------')
    print('ANSWER:')
    print(answer)
    print('----------')

    print(f"\nTotal Script Execution Time: {time() - start_time:.2f} seconds")
