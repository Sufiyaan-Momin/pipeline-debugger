"""
phase6/eval.py
─────────────────────────────────────────────────────────────────
PHASE 6 — RAG Retrieval Evaluation

This script measures how well your knowledge base retrieves
relevant documents for different error types. This is what
separates a portfolio project from a tutorial — you're
evaluating your system like an ML engineer, not just demoing it.

METRICS EXPLAINED:

  Relevance Score (0-1):
    Cosine similarity between query and retrieved chunk.
    > 0.70 = strong match
    > 0.50 = useful context
    < 0.50 = noise

  Precision@K:
    Of the top K results, what fraction are from the
    expected source? P@3 = 0.67 means 2 of 3 results
    are from the right document.

  Mean Reciprocal Rank (MRR):
    Where does the first correct result appear?
    MRR=1.0 means the correct doc is always #1.
    MRR=0.5 means it's usually #2.

  Hit Rate:
    Does the correct document appear anywhere in the
    top K results? Binary yes/no averaged across queries.

RUN:
  cd D:\pipeline-debugger
  python phase6/eval.py

  # Verbose mode (shows all retrieved chunks):
  python phase6/eval.py --verbose

  # Save results to JSON:
  python phase6/eval.py --output phase6/eval_results.json
─────────────────────────────────────────────────────────────────
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3.rag.retriever import PipelineKnowledgeRetriever


# ── Evaluation dataset ────────────────────────────────────────────
# Each test case has:
#   query:            what the agent would search for
#   expected_sources: files that SHOULD appear in top results
#   error_type:       for filtered search testing
#   description:      human-readable label

EVAL_DATASET = [
    # ── Schema mismatch queries ───────────────────────────────────
    {
        "id": "schema_01",
        "description": "Direct column-not-found error",
        "query": "column geo_region does not exist in raw_orders UndefinedColumn",
        "expected_sources": ["schema_mismatch.md"],
        "error_type": "schema_mismatch",
        "category": "schema",
    },
    {
        "id": "schema_02",
        "description": "How to fix schema mismatch",
        "query": "how to fix column not found rename upstream schema mismatch pipeline",
        "expected_sources": ["schema_mismatch.md"],
        "error_type": "schema_mismatch",
        "category": "schema",
    },
    {
        "id": "schema_03",
        "description": "Schema validation failure",
        "query": "SCHEMA MISMATCH expected columns not found actual columns raw_orders",
        "expected_sources": ["schema_mismatch.md"],
        "error_type": "schema_mismatch",
        "category": "schema",
    },

    # ── Data quality queries ──────────────────────────────────────
    {
        "id": "quality_01",
        "description": "NULL primary key error",
        "query": "null order_id primary key violation data quality gate failed",
        "expected_sources": ["data_quality.md"],
        "error_type": "data_quality_failure",
        "category": "quality",
    },
    {
        "id": "quality_02",
        "description": "Negative amount check",
        "query": "negative amount positive value check failed raw_orders",
        "expected_sources": ["data_quality.md"],
        "error_type": "data_quality_failure",
        "category": "quality",
    },
    {
        "id": "quality_03",
        "description": "Referential integrity failure",
        "query": "referential integrity foreign key customer_id does not exist orphaned records",
        "expected_sources": ["data_quality.md"],
        "error_type": "data_quality_failure",
        "category": "quality",
    },

    # ── Timeout queries ───────────────────────────────────────────
    {
        "id": "timeout_01",
        "description": "Airflow task timeout",
        "query": "AirflowTaskTimeout task killed execution timeout exceeded slow query",
        "expected_sources": ["timeout.md"],
        "error_type": "task_timeout",
        "category": "timeout",
    },
    {
        "id": "timeout_02",
        "description": "Missing index causing slow query",
        "query": "missing index sequential scan large table GROUP BY slow aggregation",
        "expected_sources": ["timeout.md"],
        "error_type": "task_timeout",
        "category": "timeout",
    },
    {
        "id": "timeout_03",
        "description": "How to fix timeout",
        "query": "CREATE INDEX CONCURRENTLY fix timeout query optimization postgres",
        "expected_sources": ["timeout.md"],
        "error_type": "task_timeout",
        "category": "timeout",
    },

    # ── Schema lookup queries ─────────────────────────────────────
    {
        "id": "schema_lookup_01",
        "description": "What columns does raw_orders have",
        "query": "raw_orders table columns schema definition order_id customer_id",
        "expected_sources": ["schema.yml"],
        "error_type": None,
        "category": "schema_lookup",
    },
    {
        "id": "schema_lookup_02",
        "description": "stg_orders model definition",
        "query": "stg_orders dbt model staging cleaned standardized orders",
        "expected_sources": ["schema.yml"],
        "error_type": None,
        "category": "schema_lookup",
    },
]


# ── Metrics calculation ───────────────────────────────────────────

@dataclass
class QueryResult:
    query_id:         str
    description:      str
    query:            str
    expected_sources: list[str]
    top_relevance:    float       = 0.0
    mean_relevance:   float       = 0.0
    precision_at_3:   float       = 0.0
    precision_at_5:   float       = 0.0
    hit_rate_at_5:    float       = 0.0
    mrr:              float       = 0.0
    retrieved:        list[dict]  = field(default_factory=list)


@dataclass
class EvalSummary:
    total_queries:       int
    mean_top_relevance:  float
    mean_precision_at_3: float
    mean_precision_at_5: float
    mean_hit_rate:       float
    mean_mrr:            float
    by_category:         dict
    pass_rate:           float   # % of queries with top relevance > 0.60


def source_matches(source_file: str, expected_sources: list[str]) -> bool:
    """Check if a retrieved chunk's source file matches any expected source."""
    source_lower = source_file.lower()
    return any(exp.lower() in source_lower for exp in expected_sources)


def evaluate_query(
    retriever: PipelineKnowledgeRetriever,
    test_case: dict,
    top_k: int = 5,
    verbose: bool = False,
) -> QueryResult:
    """Run a single query and compute all metrics."""

    results = retriever.search(
        query=test_case["query"],
        top_k=top_k,
    )

    expected = test_case["expected_sources"]

    # Relevance scores
    top_relevance  = results[0].relevance_score if results else 0.0
    mean_relevance = sum(r.relevance_score for r in results) / len(results) if results else 0.0

    # Precision@K — what fraction of top K are from expected sources
    def precision_at_k(k):
        top = results[:k]
        if not top:
            return 0.0
        hits = sum(1 for r in top if source_matches(r.source_file, expected))
        return hits / k

    # Hit rate — does ANY result in top K match?
    hit_at_5 = any(source_matches(r.source_file, expected) for r in results[:5])

    # MRR — reciprocal rank of first correct result
    mrr = 0.0
    for i, r in enumerate(results[:10], 1):
        if source_matches(r.source_file, expected):
            mrr = 1.0 / i
            break

    # Build result object
    result = QueryResult(
        query_id         = test_case["id"],
        description      = test_case["description"],
        query            = test_case["query"],
        expected_sources = expected,
        top_relevance    = top_relevance,
        mean_relevance   = mean_relevance,
        precision_at_3   = precision_at_k(3),
        precision_at_5   = precision_at_k(5),
        hit_rate_at_5    = 1.0 if hit_at_5 else 0.0,
        mrr              = mrr,
        retrieved        = [
            {
                "rank":        i + 1,
                "source_file": Path(r.source_file).name if r.source_file else "",
                "source_type": r.source_type,
                "relevance":   r.relevance_score,
                "is_correct":  source_matches(r.source_file, expected),
                "preview":     r.text[:120].replace("\n", " "),
            }
            for i, r in enumerate(results[:top_k])
        ],
    )

    if verbose:
        print(f"\n  Query: \"{test_case['query'][:70]}...\"")
        print(f"  Expected: {expected}")
        for r in result.retrieved:
            correct_marker = "✓" if r["is_correct"] else " "
            print(f"  {correct_marker} [{r['rank']}] {r['source_file']:<30} rel={r['relevance']:.3f}  {r['preview'][:60]}...")

    return result


def compute_summary(results: list[QueryResult]) -> EvalSummary:
    """Aggregate metrics across all queries."""
    n = len(results)
    if n == 0:
        return EvalSummary(0, 0, 0, 0, 0, 0, {}, 0)

    # Overall metrics
    mean_top_rel  = sum(r.top_relevance    for r in results) / n
    mean_p3       = sum(r.precision_at_3   for r in results) / n
    mean_p5       = sum(r.precision_at_5   for r in results) / n
    mean_hit      = sum(r.hit_rate_at_5    for r in results) / n
    mean_mrr      = sum(r.mrr              for r in results) / n
    pass_rate     = sum(1 for r in results if r.top_relevance > 0.60) / n

    # By category
    categories = {}
    for result in results:
        cat = next(
            (t["category"] for t in EVAL_DATASET if t["id"] == result.query_id),
            "unknown"
        )
        if cat not in categories:
            categories[cat] = {"queries": 0, "top_relevance": 0, "hit_rate": 0}
        categories[cat]["queries"]      += 1
        categories[cat]["top_relevance"] += result.top_relevance
        categories[cat]["hit_rate"]      += result.hit_rate_at_5

    for cat in categories:
        n_cat = categories[cat]["queries"]
        categories[cat]["mean_top_relevance"] = round(categories[cat]["top_relevance"] / n_cat, 3)
        categories[cat]["mean_hit_rate"]      = round(categories[cat]["hit_rate"]      / n_cat, 3)
        del categories[cat]["top_relevance"]
        del categories[cat]["hit_rate"]

    return EvalSummary(
        total_queries       = n,
        mean_top_relevance  = round(mean_top_rel, 3),
        mean_precision_at_3 = round(mean_p3, 3),
        mean_precision_at_5 = round(mean_p5, 3),
        mean_hit_rate       = round(mean_hit, 3),
        mean_mrr            = round(mean_mrr, 3),
        by_category         = categories,
        pass_rate           = round(pass_rate, 3),
    )


def print_results(results: list[QueryResult], summary: EvalSummary):
    """Print a formatted evaluation report."""

    print("\n" + "="*65)
    print("  RAG RETRIEVAL EVALUATION REPORT")
    print("="*65)

    # Per-query table
    print(f"\n{'ID':<15} {'Description':<35} {'Top Rel':>8} {'P@3':>6} {'MRR':>6} {'Hit':>5}")
    print("─" * 75)

    for r in results:
        rel_color = "✓" if r.top_relevance >= 0.60 else "△" if r.top_relevance >= 0.40 else "✗"
        print(
            f"{r.query_id:<15} "
            f"{r.description[:34]:<35} "
            f"{rel_color} {r.top_relevance:.3f} "
            f"{r.precision_at_3:>6.2f} "
            f"{r.mrr:>6.2f} "
            f"{'✓' if r.hit_rate_at_5 else '✗':>5}"
        )

    # Summary
    print("\n" + "─"*75)
    print(f"\n  OVERALL METRICS ({summary.total_queries} queries)")
    print(f"  {'Mean Top Relevance':<30} {summary.mean_top_relevance:.3f}  {'✓ PASS' if summary.mean_top_relevance >= 0.55 else '✗ NEEDS IMPROVEMENT'}")
    print(f"  {'Mean Precision@3':<30} {summary.mean_precision_at_3:.3f}")
    print(f"  {'Mean Precision@5':<30} {summary.mean_precision_at_5:.3f}")
    print(f"  {'Mean Hit Rate@5':<30} {summary.mean_hit_rate:.3f}  {'✓ PASS' if summary.mean_hit_rate >= 0.70 else '✗ NEEDS IMPROVEMENT'}")
    print(f"  {'Mean MRR':<30} {summary.mean_mrr:.3f}  {'✓ PASS' if summary.mean_mrr >= 0.50 else '✗ NEEDS IMPROVEMENT'}")
    print(f"  {'Pass Rate (rel > 0.60)':<30} {summary.pass_rate:.1%}")

    # By category
    print(f"\n  BY CATEGORY")
    print(f"  {'Category':<20} {'Queries':>8} {'Mean Rel':>10} {'Hit Rate':>10}")
    print(f"  {'─'*50}")
    for cat, stats in summary.by_category.items():
        print(f"  {cat:<20} {stats['queries']:>8} {stats['mean_top_relevance']:>10.3f} {stats['mean_hit_rate']:>10.1%}")

    # Interpretation
    print(f"\n  INTERPRETATION")
    if summary.mean_top_relevance >= 0.65:
        print("  🟢 EXCELLENT — Your knowledge base retrieves highly relevant")
        print("     context. The agent will generate accurate diagnoses.")
    elif summary.mean_top_relevance >= 0.50:
        print("  🟡 GOOD — Retrieval is working but could be improved.")
        print("     Consider adding more detailed runbooks and incidents.")
    else:
        print("  🔴 NEEDS WORK — Retrieval quality is low.")
        print("     Re-run ingest.py and add more documents to the knowledge base.")

    print("\n  HOW TO IMPROVE SCORES:")
    print("  1. Add more resolved incidents to data/incidents/ (more = better)")
    print("  2. Make runbooks more detailed with specific error messages")
    print("  3. Add column-level descriptions to schema.yml")
    print("  4. Upgrade embedding model to 'all-mpnet-base-v2' for better quality")
    print("="*65)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--verbose", action="store_true", help="Show retrieved chunks per query")
    parser.add_argument("--output",  type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--chroma",  type=str, default="phase3/data/chroma_db", help="ChromaDB path")
    args = parser.parse_args()

    # Load retriever
    print("Loading knowledge base...")
    try:
        retriever = PipelineKnowledgeRetriever(args.chroma)
        stats = retriever.collection_stats()
        print(f"Collections: {stats}")
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Run 'python phase3/rag/ingest.py --base-dir phase3' first.")
        sys.exit(1)

    # Run evaluation
    print(f"\nRunning {len(EVAL_DATASET)} evaluation queries...")
    if args.verbose:
        print("(verbose mode — showing all retrieved chunks)\n")

    results = []
    for i, test_case in enumerate(EVAL_DATASET, 1):
        print(f"  [{i:2}/{len(EVAL_DATASET)}] {test_case['id']}: {test_case['description']}")
        result = evaluate_query(retriever, test_case, top_k=5, verbose=args.verbose)
        results.append(result)

    # Compute and print summary
    summary = compute_summary(results)
    print_results(results, summary)

    # Save to JSON if requested
    if args.output:
        output = {
            "summary":  asdict(summary),
            "results":  [asdict(r) for r in results],
            "dataset":  EVAL_DATASET,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()