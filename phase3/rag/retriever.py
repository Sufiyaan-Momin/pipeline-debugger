"""
rag/retriever.py
─────────────────────────────────────────────────────────────────
PHASE 3 — RAG Retriever

This module provides the search interface your Phase 4 agent
will call as a tool. Given a query string, it returns the most
relevant chunks from ChromaDB across all collections.

Think of this as the "Google search" for your knowledge base.
When the agent sees a failed DAG, it calls this with the error
message and gets back relevant incidents, runbooks, and schema docs.

HOW VECTOR SEARCH WORKS:
  1. Your query ("column geo_region does not exist") is embedded
     into a 384-dim vector by sentence-transformers
  2. ChromaDB computes cosine similarity between your query vector
     and every stored chunk vector
  3. The top-k closest chunks are returned — these are semantically
     similar even if they don't share exact keywords

TESTING:
  python rag/retriever.py
  # Runs built-in test queries so you can see what gets retrieved
─────────────────────────────────────────────────────────────────
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import textwrap

import chromadb
from chromadb.utils import embedding_functions


# ── Data model ────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A single search result.
    The agent receives a list of these and uses them as context.
    """
    text:        str            # the actual chunk text
    source_type: str            # "incident", "runbook", "schema_doc", "sql_model"
    source_file: str            # which file it came from
    distance:    float          # cosine distance (lower = more similar, 0 = identical)
    metadata:    dict           # all other metadata

    @property
    def relevance_score(self) -> float:
        """Convert distance to 0-1 score (1 = most relevant)."""
        return round(1 - self.distance, 3)

    def __str__(self):
        return (
            f"[{self.source_type.upper()}] {self.source_file} "
            f"(relevance: {self.relevance_score})\n"
            f"{textwrap.shorten(self.text, width=200)}"
        )


# ── Retriever class ───────────────────────────────────────────────

class PipelineKnowledgeRetriever:
    """
    Main retriever class. Wraps ChromaDB and provides
    clean search methods the agent can call as tools.

    Usage:
        retriever = PipelineKnowledgeRetriever("./data/chroma_db")
        results = retriever.search("column does not exist in raw_orders")
    """

    COLLECTIONS = ["incidents", "runbooks", "schema_docs", "sql_models"]
    EMBED_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, chroma_path: str = "./data/chroma_db"):
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.EMBED_MODEL
        )
        self._collections = {}
        self._load_collections()

    def _load_collections(self):
        """Load all collections that exist in ChromaDB."""
        for name in self.COLLECTIONS:
            try:
                self._collections[name] = self.client.get_collection(
                    name=name,
                    embedding_function=self.embed_fn
                )
            except Exception:
                pass  # collection doesn't exist yet — skip silently

        if not self._collections:
            raise RuntimeError(
                "No ChromaDB collections found. "
                "Run 'python rag/ingest.py' first to build the knowledge base."
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        source_types: Optional[list[str]] = None,
        error_type: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Search the knowledge base for chunks relevant to `query`.

        Args:
            query:        Natural language search query
            top_k:        Number of results per collection
            source_types: Filter to specific source types
                          e.g. ["runbook", "incident"] to skip schema docs
            error_type:   If known, filter incidents by error_type metadata

        Returns:
            List of RetrievedChunk objects sorted by relevance (best first)

        Example:
            results = retriever.search(
                query="column geo_region does not exist",
                top_k=3,
                source_types=["incident", "runbook"]
            )
        """
        all_results = []

        # Decide which collections to search
        collections_to_search = {
            name: col
            for name, col in self._collections.items()
            if source_types is None or name.replace("_", "").replace("docs", "") in source_types
               or name in source_types
        }

        for col_name, collection in collections_to_search.items():
            if collection.count() == 0:
                continue

            # Build optional metadata filter
            where_filter = None
            if error_type and col_name == "incidents":
                where_filter = {"error_type": {"$eq": error_type}}

            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=min(top_k, collection.count()),
                    where=where_filter,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception as e:
                print(f"  Warning: search failed in collection '{col_name}': {e}")
                continue

            # Unpack ChromaDB response format
            docs       = results["documents"][0]
            metadatas  = results["metadatas"][0]
            distances  = results["distances"][0]

            for doc, meta, dist in zip(docs, metadatas, distances):
                all_results.append(RetrievedChunk(
                    text        = doc,
                    source_type = meta.get("source_type", col_name),
                    source_file = meta.get("source_file", ""),
                    distance    = dist,
                    metadata    = meta,
                ))

        # Sort all results across all collections by relevance
        all_results.sort(key=lambda x: x.distance)
        return all_results

    def search_incidents(self, query: str, error_type: str = None, top_k: int = 3) -> list[RetrievedChunk]:
        """Convenience: search only the incidents collection."""
        return self.search(query, top_k=top_k, source_types=["incidents"], error_type=error_type)

    def search_runbooks(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Convenience: search only the runbooks collection."""
        return self.search(query, top_k=top_k, source_types=["runbooks"])

    def search_schema(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Convenience: search only schema docs."""
        return self.search(query, top_k=top_k, source_types=["schema_docs"])

    def get_context_for_error(self, error_message: str, error_type: str = None) -> str:
        """
        HIGH-LEVEL METHOD: Given an error message, return a formatted
        context string ready to inject into an LLM prompt.

        This is what the Phase 4 agent calls before sending to the LLM:
            context = retriever.get_context_for_error(
                "column geo_region does not exist",
                error_type="schema_mismatch"
            )
            prompt = SYSTEM_PROMPT + context + "\\nNow diagnose this error..."

        Returns a formatted string like:
            === RELEVANT INCIDENTS ===
            [INCIDENT] dag_02... (relevance: 0.91)
            Schema mismatch on raw_orders. Column geo_region...

            === RUNBOOK GUIDANCE ===
            [RUNBOOK] schema_mismatch.md (relevance: 0.88)
            A schema mismatch occurs when our pipeline expects...

            === SCHEMA DOCUMENTATION ===
            [SCHEMA_DOC] schema.yml (relevance: 0.83)
            DBT SOURCE TABLE: pipeline_db.raw_orders...
        """
        sections = []

        # Past incidents — most diagnostic
        incidents = self.search_incidents(error_message, error_type=error_type, top_k=3)
        if incidents:
            lines = ["=== RELEVANT PAST INCIDENTS ==="]
            for r in incidents:
                lines.append(f"\n[Relevance: {r.relevance_score}] From: {r.source_file}")
                lines.append(r.text)
            sections.append("\n".join(lines))

        # Runbook guidance — how to fix it
        runbooks = self.search_runbooks(error_message, top_k=2)
        if runbooks:
            lines = ["=== RUNBOOK GUIDANCE ==="]
            for r in runbooks:
                lines.append(f"\n[Relevance: {r.relevance_score}] From: {r.source_file}")
                lines.append(r.text)
            sections.append("\n".join(lines))

        # Schema docs — what columns should exist
        schema = self.search_schema(error_message, top_k=2)
        if schema:
            lines = ["=== SCHEMA DOCUMENTATION ==="]
            for r in schema:
                lines.append(f"\n[Relevance: {r.relevance_score}] From: {r.source_file}")
                lines.append(r.text)
            sections.append("\n".join(lines))

        if not sections:
            return "No relevant context found in knowledge base."

        return "\n\n".join(sections)

    def collection_stats(self) -> dict:
        """Return count of chunks in each collection."""
        return {
            name: col.count()
            for name, col in self._collections.items()
        }


# ── Test harness ──────────────────────────────────────────────────

def run_test_queries(retriever: PipelineKnowledgeRetriever):
    """
    Run a set of test queries and print results.
    Use this to verify your knowledge base is working correctly
    BEFORE connecting the agent.

    HOW TO INTERPRET RESULTS:
      Relevance 0.85+  = very strong match, will definitely help the agent
      Relevance 0.70+  = good match, useful context
      Relevance 0.50+  = weak match, might be noise
      Relevance <0.50  = probably irrelevant, check your documents
    """
    test_queries = [
        # Schema error queries
        {
            "label": "Schema mismatch — column renamed",
            "query": "column does not exist in raw_orders geo_region",
            "expected_sources": ["schema_mismatch.md", "dag_02"],
        },
        # Data quality queries
        {
            "label": "Data quality — NULL order IDs",
            "query": "null order_id primary key violation data quality failed",
            "expected_sources": ["data_quality.md", "dag_03"],
        },
        # Timeout queries
        {
            "label": "Timeout — slow query",
            "query": "airflow task timeout execution limit exceeded slow query",
            "expected_sources": ["dag_04"],
        },
        # Schema lookup
        {
            "label": "Schema lookup — what columns does raw_orders have?",
            "query": "raw_orders columns schema definition",
            "expected_sources": ["schema.yml"],
        },
        # Fix lookup
        {
            "label": "Fix lookup — how do I fix a schema mismatch?",
            "query": "how to fix column not found rename upstream schema mismatch",
            "expected_sources": ["schema_mismatch.md"],
        },
    ]

    print("\n" + "="*60)
    print("RAG RETRIEVER TEST")
    print("="*60)
    print(f"\nKnowledge base stats: {retriever.collection_stats()}")

    for test in test_queries:
        print(f"\n{'─'*60}")
        print(f"QUERY: {test['label']}")
        print(f"  \"{test['query']}\"")
        print(f"Expected sources: {test['expected_sources']}")
        print("Results:")

        results = retriever.search(test["query"], top_k=3)

        if not results:
            print("  ⚠  No results returned — is the knowledge base empty?")
            continue

        for i, r in enumerate(results, 1):
            source_short = Path(r.source_file).name if r.source_file else "unknown"
            print(f"\n  {i}. [{r.source_type}] {source_short} (relevance: {r.relevance_score})")
            # Print first 200 chars of the chunk
            preview = r.text[:200].replace('\n', ' ')
            print(f"     {preview}...")

    print("\n" + "="*60)
    print("FULL CONTEXT BLOCK (what the agent sees for a schema error):")
    print("="*60)
    context = retriever.get_context_for_error(
        "column geo_region does not exist in raw_orders",
        error_type="schema_mismatch"
    )
    print(context[:2000])
    if len(context) > 2000:
        print(f"\n... [{len(context) - 2000} more characters]")


if __name__ == "__main__":
    import sys

    chroma_path = "./data/chroma_db"
    if len(sys.argv) > 1:
        chroma_path = sys.argv[1]

    try:
        retriever = PipelineKnowledgeRetriever(chroma_path)
        run_test_queries(retriever)
    except RuntimeError as e:
        print(f"\nError: {e}")
        print("Solution: Run 'python rag/ingest.py' to build the knowledge base first.")
        sys.exit(1)