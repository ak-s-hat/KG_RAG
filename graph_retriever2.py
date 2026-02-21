from neo4j import GraphDatabase
from config import MAX_DEPTH
from groq_client import extract_keywords  # wrapper for Groq API


class GraphRetriever:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pass, neo4j_db, thread_id: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
        self.database = neo4j_db
        self.thread_id = thread_id

    def close(self):
        self.driver.close()

    def get_keywords_for_thread(self, thread_id: str):
        """
        Retrieve all Keyword node names for a specific thread_id.
        """
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                MATCH (k:Keyword {thread_id: $thread_id})
                RETURN DISTINCT k.name AS name
            """, {"thread_id": thread_id})
            return [r["name"] for r in result]
        
    # --- Core Retrieval ---
    def retrieve(self, query: str):
        # ✅ 1. Fetch all keywords for this thread from Neo4j
        graph_keywords = self.get_keywords_for_thread(self.thread_id)

        # ✅ 2. Extract query-specific keywords via Gemini using available graph keywords
        query_keywords = extract_keywords(query, graph_keywords)
        print(f'Extracted keywords: {query_keywords}')
        if not query_keywords:
            print("⚠ No keywords extracted from query")
            return []

        matched_keywords = query_keywords
        if not matched_keywords:
            print("⚠ No matches found for query keywords")
            return []

        # 3. Retrieve primary chunks connected to matched keywords (scored)
        with self.driver.session(database=self.database) as session:
            result = session.run("""
                // Part 1: Calculate score per chunk
                MATCH (k:Keyword {thread_id: $thread_id})-[:APPEARS_IN {thread_id: $thread_id}]->(c:Chunk {thread_id: $thread_id})
                WHERE k.name IN $keywords
                WITH c, count(k) AS score
                ORDER BY score DESC
                LIMIT 1
                WITH score AS max_score

                // Part 2: Get all chunks with same top score
                MATCH (k2:Keyword {thread_id: $thread_id})-[:APPEARS_IN {thread_id: $thread_id}]->(c2:Chunk {thread_id: $thread_id})
                WHERE k2.name IN $keywords
                WITH c2, count(k2) AS final_score, max_score
                WHERE final_score = max_score
                RETURN c2.id AS id, c2.content AS content
            """, {
                "keywords": matched_keywords,
                "thread_id": self.thread_id
            })
            
            primary_chunks = [{"id": r["id"], "content": r["content"]} for r in result]

        # 4. Expand neighborhood up to MAX_DEPTH
        retrieved_chunks = primary_chunks.copy()
        visited = set([c["id"] for c in primary_chunks])
        frontier = [c["id"] for c in primary_chunks]

        for depth in range(MAX_DEPTH):
            if not frontier:
                break
            with self.driver.session(database=self.database) as session:
                result = session.run("""
                    // Case A: shared keyword
                    MATCH (c:Chunk {thread_id: $thread_id})<-[:APPEARS_IN {thread_id: $thread_id}]-(k:Keyword {thread_id: $thread_id})-[:APPEARS_IN {thread_id: $thread_id}]->(n:Chunk {thread_id: $thread_id})
                    WHERE c.id IN $frontier
                    RETURN DISTINCT n.id AS id, n.content AS content
                    UNION
                    // Case B: keyword similarity
                    MATCH (c:Chunk {thread_id: $thread_id})<-[:APPEARS_IN {thread_id: $thread_id}]-(k1:Keyword {thread_id: $thread_id})-[:SIMILAR_TO {thread_id: $thread_id}]-(k2:Keyword {thread_id: $thread_id})-[:APPEARS_IN {thread_id: $thread_id}]->(n:Chunk {thread_id: $thread_id})
                    WHERE c.id IN $frontier
                    RETURN DISTINCT n.id AS id, n.content AS content
                """, {"frontier": frontier, "thread_id": self.thread_id})

                neighbors = [{"id": r["id"], "content": r["content"]} for r in result if r["id"] not in visited]

            retrieved_chunks.extend(neighbors)
            visited.update([n["id"] for n in neighbors])
            frontier = [n["id"] for n in neighbors]

        return retrieved_chunks