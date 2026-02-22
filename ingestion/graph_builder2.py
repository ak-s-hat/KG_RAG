# graph_builder.py

from neo4j import GraphDatabase
import sys
from pathlib import Path

# Add root directory to sys.path to allow running this script directly
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from typing import List, Dict
from config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    NEO4J_DATABASE,
)


class KnowledgeGraphBuilder:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD, database=NEO4J_DATABASE):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def clear_graph(self, thread_id: str):
        """
        Clears only the graph data belonging to a specific thread_id.
        """
        print(f"🧹 Clearing graph for thread_id={thread_id}...")
        with self.driver.session(database=self.database) as session:
            session.run("MATCH (n {thread_id: $thread_id}) DETACH DELETE n", thread_id=thread_id)
        print(f"✅ Cleared graph for thread_id={thread_id}")

    def build_graph_from_map(self, keyword_to_chunks_map: Dict[str, List[str]], thread_id: str):
        """
        Builds a knowledge graph from a map of keywords to chunks.
        All nodes and relationships are tagged with thread_id.
        """
        print(f"Building graph for thread_id={thread_id} with {len(keyword_to_chunks_map)} keywords...")

        # --- 1. Extract unique chunks and keywords ---
        all_keywords = list(keyword_to_chunks_map.keys())
        unique_chunks_set = set(chunk for chunks in keyword_to_chunks_map.values() for chunk in chunks)
        all_chunks = list(unique_chunks_set)
        chunk_to_id = {chunk: i for i, chunk in enumerate(all_chunks)}

        # --- 2. Push nodes and relationships to Neo4j ---
        with self.driver.session(database=self.database) as session:
            # Create Chunk nodes
            print("Creating Chunk nodes...")
            for i, chunk in enumerate(all_chunks):
                session.run(
                    """
                    MERGE (c:Chunk {id: $id, thread_id: $thread_id})
                    SET c.content = $content
                    """,
                    id=i, content=chunk, thread_id=thread_id
                )

            # Create Keyword nodes
            print("Creating Keyword nodes...")
            for kw in all_keywords:
                session.run(
                    """
                    MERGE (k:Keyword {name: $name, thread_id: $thread_id})
                    """,
                    name=kw, thread_id=thread_id
                )

            # Create APPEARS_IN relationships
            print("Creating APPEARS_IN relationships...")
            for kw, chunks in keyword_to_chunks_map.items():
                for chunk in chunks:
                    chunk_id = chunk_to_id[chunk]
                    session.run(
                        """
                        MATCH (k:Keyword {name: $kw_name, thread_id: $thread_id})
                        MATCH (c:Chunk {id: $c_id, thread_id: $thread_id})
                        MERGE (k)-[:APPEARS_IN {thread_id: $thread_id}]->(c)
                        """,
                        kw_name=kw, c_id=chunk_id, thread_id=thread_id
                    )

        print(f"✅ Graph built for thread_id={thread_id} with {len(all_chunks)} chunks and {len(all_keywords)} keywords.")
