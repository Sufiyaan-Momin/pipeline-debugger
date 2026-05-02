"""
rag/ingest.py
─────────────────────────────────────────────────────────────────
PHASE 3 — RAG Knowledge Base Ingestion

This is the script that BUILDS your agent's brain.
It reads every document your agent needs to know about,
splits them into chunks, embeds them as vectors, and
stores everything in ChromaDB.

After running this, your agent can answer:
  "When did we last see a schema mismatch on raw_orders?"
  "What does the runbook say about NULL order_ids?"
  "What columns does stg_orders expose?"

HOW EMBEDDINGS WORK (the key concept):
  A sentence like "column geo_region does not exist"
  gets converted into a list of ~384 numbers, e.g.:
    [0.23, -0.71, 0.05, 0.88, ...]
  This is its "meaning fingerprint".
  When the agent searches "schema error on orders table",
  ChromaDB finds the chunks whose fingerprints are
  mathematically closest — that's semantic search.

INSTALL (run once):
  pip install chromadb sentence-transformers pyyaml

RUN:
  python rag/ingest.py
  # or with a custom data directory:
  python rag/ingest.py --data-dir ./data

WHAT IT INGESTS:
  1. Incident JSON files  (from scripts/log_exporter.py)
  2. Runbook markdown     (from docs/runbooks/)
  3. dbt schema.yml       (column definitions + tests)
  4. dbt SQL models       (the actual transformation logic)
─────────────────────────────────────────────────────────────────
"""

import argparse
import json
import re
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Iterator

import yaml
import chromadb
from chromadb.utils import embedding_functions


# ── Config ────────────────────────────────────────────────────────

CHROMA_PATH   = "./data/chroma_db"   # where ChromaDB persists to disk
EMBED_MODEL   = "all-MiniLM-L6-v2"  # 384-dim, fast, free, runs locally
                                      # good enough for this project
                                      # upgrade to 'all-mpnet-base-v2' for
                                      # better quality (slower)

# Chunk sizes by document type — tuned for each source
CHUNK_CONFIG = {
    "incident":  {"size": 600,  "overlap": 100},
    "runbook":   {"size": 500,  "overlap": 80},
    "schema":    {"size": 300,  "overlap": 50},   # schema chunks are naturally small
    "sql_model": {"size": 400,  "overlap": 60},
}

# One collection per document type — lets us filter searches by source
COLLECTIONS = ["incidents", "runbooks", "schema_docs", "sql_models"]


# ── Chunking ──────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks.

    WHY OVERLAP?
      If a key sentence falls at the boundary between two chunks,
      we'd lose context. Overlap ensures every sentence appears
      fully in at least one chunk.

    Example with chunk_size=20, overlap=5:
      "The cat sat on the mat near the door"
      Chunk 1: "The cat sat on the"
      Chunk 2: "on the mat near the"   ← 'on the' repeated
      Chunk 3: "near the door"
    """
    if not text.strip():
        return []

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Move forward by (chunk_size - overlap) to create the overlap
        start += chunk_size - overlap

    return [c for c in chunks if len(c.strip()) > 30]  # skip tiny fragments


def make_doc_id(source: str, index: int) -> str:
    """
    Create a stable, unique ID for each chunk.
    Uses a hash of the source path + index so re-running
    ingest.py updates existing chunks instead of duplicating.
    """
    raw = f"{source}::{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


# ── Document loaders ──────────────────────────────────────────────

def load_incidents(data_dir: Path) -> Iterator[dict]:
    """
    Load incident JSON files exported by scripts/log_exporter.py.
    Each incident becomes a rich text document combining:
      - error type and detail
      - the log excerpt (most diagnostic)
      - the suggested fix
    """
    incidents_dir = data_dir / "incidents"
    if not incidents_dir.exists():
        print(f"  ⚠  No incidents/ folder at {incidents_dir} — skipping")
        print("     Run scripts/log_exporter.py first to generate incidents")
        return

    json_files = list(incidents_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("_")]  # skip _all_incidents.json

    print(f"  Found {len(json_files)} incident files")

    for path in json_files:
        with open(path) as f:
            incident = json.load(f)

        # Build a human-readable document from the structured JSON
        # This reads naturally and embeds better than raw JSON
        doc_text = f"""
INCIDENT REPORT
===============
DAG: {incident.get('dag_id', 'unknown')}
Task: {incident.get('task_id', 'unknown')}
Error type: {incident.get('error_type', 'unknown')}
Date: {incident.get('start_date', 'unknown')}

ERROR DETAIL:
{incident.get('error_detail', 'No detail available')}

SUGGESTED FIX:
{incident.get('suggested_fix', 'No fix recorded')}

LOG EXCERPT (last 2000 chars):
{incident.get('log_excerpt', '')[-2000:]}

RESOLUTION:
{incident.get('resolution') or 'Not yet resolved'}
        """.strip()

        cfg = CHUNK_CONFIG["incident"]
        chunks = chunk_text(doc_text, cfg["size"], cfg["overlap"])

        for i, chunk in enumerate(chunks):
            yield {
                "text": chunk,
                "metadata": {
                    "source_type": "incident",
                    "source_file": path.name,
                    "dag_id":      incident.get("dag_id", ""),
                    "task_id":     incident.get("task_id", ""),
                    "error_type":  incident.get("error_type", ""),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "ingested_at": datetime.utcnow().isoformat(),
                },
                "id": make_doc_id(path.name, i),
            }


def load_runbooks(docs_dir: Path) -> Iterator[dict]:
    """
    Load markdown runbooks from docs/runbooks/.
    These contain team knowledge: how to diagnose and fix each error type.

    SMART CHUNKING: We split on ## headers so each chunk stays
    within one section (Symptoms, Diagnosis, Fix procedures, etc.)
    This keeps related info together and improves retrieval quality.
    """
    runbooks_dir = docs_dir / "runbooks"
    if not runbooks_dir.exists():
        print(f"  ⚠  No runbooks/ folder at {runbooks_dir} — skipping")
        return

    md_files = list(runbooks_dir.glob("*.md"))
    print(f"  Found {len(md_files)} runbook files")

    for path in md_files:
        content = path.read_text()

        # Split on markdown headers (## or ###) for natural section boundaries
        sections = re.split(r'\n(?=#{1,3} )', content)
        sections = [s.strip() for s in sections if s.strip()]

        for i, section in enumerate(sections):
            # Each section may still be large — chunk it further
            cfg = CHUNK_CONFIG["runbook"]
            sub_chunks = chunk_text(section, cfg["size"], cfg["overlap"])

            for j, chunk in enumerate(sub_chunks):
                # Extract section heading for metadata
                first_line = chunk.split('\n')[0].strip('#').strip()

                yield {
                    "text": chunk,
                    "metadata": {
                        "source_type":  "runbook",
                        "source_file":  path.name,
                        "runbook_name": path.stem,
                        "section":      first_line[:80],
                        "section_index": i,
                        "chunk_index":  j,
                        "ingested_at":  datetime.utcnow().isoformat(),
                    },
                    "id": make_doc_id(f"{path.name}::{i}", j),
                }


def load_dbt_schema(dbt_dir: Path) -> Iterator[dict]:
    """
    Load dbt schema.yml files.
    These define every column, its description, and its data quality tests.
    This is gold for your agent — it knows exactly what each column means.

    We flatten the YAML hierarchy into readable text chunks so the
    embedding model can understand them semantically.
    """
    schema_files = list(dbt_dir.rglob("schema.yml"))
    print(f"  Found {len(schema_files)} dbt schema files")

    for path in schema_files:
        with open(path) as f:
            schema = yaml.safe_load(f)

        # ── Sources (raw tables) ──────────────────────────────────
        for source in schema.get("sources", []):
            source_name = source.get("name", "")
            source_desc = source.get("description", "")

            # One chunk per table
            for table in source.get("tables", []):
                table_name  = table.get("name", "")
                table_desc  = table.get("description", "")
                columns     = table.get("columns", [])

                col_lines = []
                for col in columns:
                    col_name = col.get("name", "")
                    col_desc = col.get("description", "")
                    tests    = [str(t) if isinstance(t, str) else list(t.keys())[0]
                                for t in col.get("tests", [])]
                    col_lines.append(
                        f"  Column '{col_name}': {col_desc}"
                        + (f" [tests: {', '.join(tests)}]" if tests else "")
                    )

                doc_text = f"""
DBT SOURCE TABLE: {source_name}.{table_name}
Description: {table_desc}
Columns:
{chr(10).join(col_lines)}
                """.strip()

                cfg = CHUNK_CONFIG["schema"]
                chunks = chunk_text(doc_text, cfg["size"], cfg["overlap"])

                for i, chunk in enumerate(chunks):
                    yield {
                        "text": chunk,
                        "metadata": {
                            "source_type":  "schema_doc",
                            "source_file":  str(path),
                            "table_name":   table_name,
                            "schema_type":  "source",
                            "chunk_index":  i,
                            "ingested_at":  datetime.utcnow().isoformat(),
                        },
                        "id": make_doc_id(f"source::{table_name}", i),
                    }

        # ── Models (transformed tables) ───────────────────────────
        for model in schema.get("models", []):
            model_name = model.get("name", "")
            model_desc = model.get("description", "")
            columns    = model.get("columns", [])

            col_lines = []
            for col in columns:
                col_name = col.get("name", "")
                col_desc = col.get("description", "")
                tests    = [str(t) if isinstance(t, str) else list(t.keys())[0]
                            for t in col.get("tests", [])]
                col_lines.append(
                    f"  Column '{col_name}': {col_desc}"
                    + (f" [tests: {', '.join(tests)}]" if tests else "")
                )

            doc_text = f"""
DBT MODEL: {model_name}
Description: {model_desc}
Columns:
{chr(10).join(col_lines)}
            """.strip()

            cfg = CHUNK_CONFIG["schema"]
            chunks = chunk_text(doc_text, cfg["size"], cfg["overlap"])

            for i, chunk in enumerate(chunks):
                yield {
                    "text": chunk,
                    "metadata": {
                        "source_type":  "schema_doc",
                        "source_file":  str(path),
                        "table_name":   model_name,
                        "schema_type":  "model",
                        "chunk_index":  i,
                        "ingested_at":  datetime.utcnow().isoformat(),
                    },
                    "id": make_doc_id(f"model::{model_name}", i),
                }


def load_sql_models(dbt_dir: Path) -> Iterator[dict]:
    """
    Load dbt SQL model files.
    The agent can use these to understand what transformations happen
    and propose fixes when a transformation breaks.
    """
    sql_files = list(dbt_dir.rglob("*.sql"))
    print(f"  Found {len(sql_files)} SQL model files")

    for path in sql_files:
        content = path.read_text()

        # Extract the docstring comment at the top if present
        comment_match = re.match(r'^--[^\n]*\n((?:--[^\n]*\n)*)', content)
        header_comment = comment_match.group(0) if comment_match else ""

        doc_text = f"""
SQL MODEL: {path.stem}
File: {path}

{content}
        """.strip()

        cfg = CHUNK_CONFIG["sql_model"]
        chunks = chunk_text(doc_text, cfg["size"], cfg["overlap"])

        for i, chunk in enumerate(chunks):
            yield {
                "text": chunk,
                "metadata": {
                    "source_type": "sql_model",
                    "source_file": str(path),
                    "model_name":  path.stem,
                    "chunk_index": i,
                    "ingested_at": datetime.utcnow().isoformat(),
                },
                "id": make_doc_id(str(path), i),
            }


# ── ChromaDB setup ────────────────────────────────────────────────

def get_or_create_collection(client: chromadb.Client, name: str, embed_fn):
    """
    Get existing collection or create a new one.
    ChromaDB collections are like tables — each holds a
    set of (text, embedding, metadata) triples.
    """
    try:
        collection = client.get_collection(name=name, embedding_function=embed_fn)
        print(f"  Using existing collection '{name}' ({collection.count()} docs)")
    except Exception:
        collection = client.create_collection(
            name=name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for text
        )
        print(f"  Created new collection '{name}'")
    return collection


def upsert_batch(collection, documents: list[dict], batch_size: int = 100):
    """
    Insert or update documents in batches.
    Upsert = update if ID exists, insert if not.
    This makes re-running ingest.py safe — no duplicates.
    """
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        collection.upsert(
            ids       =[d["id"]            for d in batch],
            documents =[d["text"]          for d in batch],
            metadatas =[d["metadata"]      for d in batch],
        )
        print(f"    Upserted {min(i + batch_size, total)}/{total} chunks")


# ── Main ──────────────────────────────────────────────────────────

def main(base_dir: Path):
    print("\n" + "="*60)
    print("PHASE 3 — RAG Knowledge Base Ingestion")
    print("="*60)

    # Paths relative to the project root
    data_dir = base_dir / "data"
    docs_dir = base_dir / "docs"
    dbt_dir  = base_dir / "dbt_project"

    # ── Initialize ChromaDB ───────────────────────────────────────
    print(f"\nInitializing ChromaDB at: {base_dir / 'data/chroma_db'}")
    client = chromadb.PersistentClient(path=str(base_dir / "data/chroma_db"))

    # sentence-transformers runs locally — no API key needed
    # First run downloads ~90MB model from HuggingFace (one time only)
    print(f"Loading embedding model: {EMBED_MODEL}")
    print("  (First run downloads ~90MB — subsequent runs are instant)")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # ── Ingest incidents ──────────────────────────────────────────
    print("\n[1/4] Ingesting incidents...")
    incident_docs = list(load_incidents(data_dir))
    if incident_docs:
        col = get_or_create_collection(client, "incidents", embed_fn)
        upsert_batch(col, incident_docs)
        print(f"  ✓ {len(incident_docs)} incident chunks ingested")

    # ── Ingest runbooks ───────────────────────────────────────────
    print("\n[2/4] Ingesting runbooks...")
    runbook_docs = list(load_runbooks(docs_dir))
    if runbook_docs:
        col = get_or_create_collection(client, "runbooks", embed_fn)
        upsert_batch(col, runbook_docs)
        print(f"  ✓ {len(runbook_docs)} runbook chunks ingested")

    # ── Ingest dbt schema ─────────────────────────────────────────
    print("\n[3/4] Ingesting dbt schema docs...")
    schema_docs = list(load_dbt_schema(dbt_dir))
    if schema_docs:
        col = get_or_create_collection(client, "schema_docs", embed_fn)
        upsert_batch(col, schema_docs)
        print(f"  ✓ {len(schema_docs)} schema chunks ingested")

    # ── Ingest SQL models ─────────────────────────────────────────
    print("\n[4/4] Ingesting SQL models...")
    sql_docs = list(load_sql_models(dbt_dir))
    if sql_docs:
        col = get_or_create_collection(client, "sql_models", embed_fn)
        upsert_batch(col, sql_docs)
        print(f"  ✓ {len(sql_docs)} SQL chunks ingested")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("INGESTION COMPLETE")
    print("="*60)
    total = len(incident_docs) + len(runbook_docs) + len(schema_docs) + len(sql_docs)
    print(f"Total chunks stored: {total}")
    print(f"ChromaDB location:   {base_dir / 'data/chroma_db'}")
    print("\nCollections:")
    for name in COLLECTIONS:
        try:
            col = client.get_collection(name, embedding_function=embed_fn)
            print(f"  {name:<15} {col.count()} chunks")
        except Exception:
            print(f"  {name:<15} (empty)")

    print("\nNext step: Run rag/retriever.py to test search")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Root directory of the project (default: current directory)"
    )
    args = parser.parse_args()
    main(args.base_dir)