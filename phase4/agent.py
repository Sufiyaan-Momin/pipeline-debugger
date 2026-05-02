"""
phase4/agent.py
─────────────────────────────────────────────────────────────────
PHASE 4 — Self-Healing Pipeline Monitoring Agent

This is the brain of the entire project. It ties together:
  - Airflow REST API  (what failed?)
  - ChromaDB RAG      (what do we know about this error?)
  - Groq LLM          (what does it mean and how do we fix it?)
  - Incident reports  (structured output saved to disk)

HOW IT WORKS (ReAct loop):
  1. OBSERVE  — poll Airflow for failed tasks
  2. RETRIEVE — search ChromaDB for relevant incidents/runbooks
  3. REASON   — send error + context to LLM, ask for diagnosis
  4. ACT      — save report, print to console, optionally alert Slack
  5. REPEAT   — wait 30 seconds, go back to step 1

RUN:
  # Make sure GROQ_API_KEY is set first:
  $env:GROQ_API_KEY = "gsk_..."

  # Run the agent (polls continuously):
  python phase4/agent.py

  # Run once and exit (good for testing):
  python phase4/agent.py --once

  # Run on a specific DAG only:
  python phase4/agent.py --dag dag_02_schema_mismatch
─────────────────────────────────────────────────────────────────
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from groq import Groq

# Add project root to path so we can import the retriever
sys.path.insert(0, str(Path(__file__).parent.parent))
from phase3.rag.retriever import PipelineKnowledgeRetriever


# ── Config ────────────────────────────────────────────────────────

AIRFLOW_BASE_URL  = "http://localhost:8080/api/v1"
AIRFLOW_AUTH      = ("admin", "admin")
CHROMA_PATH       = "phase3/data/chroma_db"
REPORTS_DIR       = Path("phase4/reports")
POLL_INTERVAL_SEC = 30        # how often to check Airflow
GROQ_MODEL        = "llama-3.3-70b-versatile"  # fast, free, very capable

# Track which failures we've already diagnosed so we don't repeat
_diagnosed_runs: set = set()


# ── Airflow API helpers ───────────────────────────────────────────

def get_all_dags() -> list[dict]:
    """Fetch all DAGs from Airflow."""
    resp = requests.get(f"{AIRFLOW_BASE_URL}/dags", auth=AIRFLOW_AUTH)
    resp.raise_for_status()
    return resp.json().get("dags", [])


def get_recent_dag_runs(dag_id: str, limit: int = 5) -> list[dict]:
    """Get most recent runs for a DAG."""
    resp = requests.get(
        f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns",
        params={"limit": limit, "order_by": "-start_date"},
        auth=AIRFLOW_AUTH,
    )
    resp.raise_for_status()
    return resp.json().get("dag_runs", [])


def get_failed_tasks(dag_id: str, run_id: str) -> list[dict]:
    """Get all failed task instances for a specific run."""
    resp = requests.get(
        f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{run_id}/taskInstances",
        auth=AIRFLOW_AUTH,
    )
    resp.raise_for_status()
    tasks = resp.json().get("task_instances", [])
    return [t for t in tasks if t.get("state") == "failed"]


def get_task_log(dag_id: str, run_id: str, task_id: str) -> str:
    """Fetch raw log text for a failed task."""
    resp = requests.get(
        f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{run_id}/taskInstances/{task_id}/logs/1",
        auth=AIRFLOW_AUTH,
        headers={"Accept": "text/plain"},
    )
    if resp.status_code == 200:
        return resp.text[-4000:]  # last 4000 chars — most relevant
    return f"[Log unavailable: HTTP {resp.status_code}]"


# ── Error classification ──────────────────────────────────────────

def classify_error(log_text: str) -> str:
    """
    Quick rule-based classifier to give the LLM a head start.
    The LLM will refine this classification in its diagnosis.
    """
    log_lower = log_text.lower()
    if "undefinedcolumn" in log_lower or ("column" in log_lower and "does not exist" in log_lower):
        return "schema_mismatch"
    if "assertionerror" in log_lower or "data quality gate failed" in log_lower or "not null check failed" in log_lower:
        return "data_quality_failure"
    if "airflowtasktimeout" in log_lower or "timeout" in log_lower:
        return "task_timeout"
    if "connection refused" in log_lower or "could not connect" in log_lower:
        return "connection_failure"
    return "unknown"


# ── LLM diagnosis ─────────────────────────────────────────────────

def diagnose_with_llm(
    client: Groq,
    retriever: PipelineKnowledgeRetriever,
    dag_id: str,
    task_id: str,
    error_type: str,
    log_excerpt: str,
) -> dict:
    """
    Core agent function: given a failure, retrieve context and ask
    the LLM to produce a structured diagnosis.

    Returns a dict with:
      - error_type
      - root_cause
      - confidence (high/medium/low)
      - recommended_fix
      - relevant_runbook
      - prevention_tip
      - raw_response
    """
    print(f"\n  🔍 Retrieving context from knowledge base...")
    context = retriever.get_context_for_error(log_excerpt, error_type=error_type)

    # Build the prompt
    system_prompt = """You are an expert data engineering assistant specializing in 
diagnosing Apache Airflow pipeline failures. You have deep knowledge of:
- Common pipeline failure patterns (schema drift, data quality issues, timeouts)
- dbt data transformations and schema documentation
- Postgres database operations and optimization
- Incident response and root cause analysis

When given a pipeline failure, you analyze the error log and relevant context
to produce a clear, actionable diagnosis. Always be specific and concrete.
Never say "it could be many things" — pick the most likely root cause."""

    user_prompt = f"""A pipeline task has failed. Please diagnose it and provide a fix.

## Failed Task
- DAG: {dag_id}
- Task: {task_id}
- Initial error classification: {error_type}

## Error Log (last 4000 chars)
```
{log_excerpt}
```

## Relevant Knowledge Base Context
{context}

## Your Task
Analyze the failure and respond with ONLY a JSON object in this exact format:
{{
  "error_type": "the specific error category (schema_mismatch/data_quality_failure/task_timeout/connection_failure/unknown)",
  "root_cause": "1-2 sentence specific explanation of WHY this failed",
  "confidence": "high/medium/low",
  "recommended_fix": "exact steps to fix this, including any SQL or code changes needed",
  "relevant_runbook": "which runbook applies (schema_mismatch.md/data_quality.md/timeout.md/none)",
  "prevention_tip": "one specific thing to do to prevent this in future",
  "needs_immediate_action": true/false
}}

Respond with ONLY the JSON. No preamble, no explanation outside the JSON."""

    print(f"  🤖 Sending to Groq LLM ({GROQ_MODEL})...")

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,   # low temperature = more deterministic diagnosis
            max_tokens=1000,
        )

        raw_text = response.choices[0].message.content.strip()

        # Parse the JSON response
        # Strip markdown code fences if the LLM added them
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        diagnosis = json.loads(raw_text)
        diagnosis["raw_response"] = raw_text
        return diagnosis

    except json.JSONDecodeError as e:
        print(f"  ⚠  LLM response wasn't valid JSON: {e}")
        return {
            "error_type": error_type,
            "root_cause": "LLM response could not be parsed",
            "confidence": "low",
            "recommended_fix": "Check the raw_response field for LLM output",
            "relevant_runbook": "none",
            "prevention_tip": "none",
            "needs_immediate_action": False,
            "raw_response": response.choices[0].message.content if 'response' in dir() else "no response",
        }
    except Exception as e:
        print(f"  ⚠  LLM call failed: {e}")
        return {
            "error_type": error_type,
            "root_cause": f"LLM call failed: {str(e)}",
            "confidence": "low",
            "recommended_fix": "Check GROQ_API_KEY and network connection",
            "relevant_runbook": "none",
            "prevention_tip": "none",
            "needs_immediate_action": False,
            "raw_response": str(e),
        }


# ── Report generation ─────────────────────────────────────────────

def save_report(
    dag_id: str,
    task_id: str,
    run_id: str,
    log_excerpt: str,
    diagnosis: dict,
) -> Path:
    """
    Save a structured incident report to disk.
    These reports are what you show in your portfolio demo.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Clean filename for Windows
    safe_run = run_id.replace(":", "-").replace("+", "")[:50]
    filename = f"{dag_id}__{task_id}__{safe_run}.json"
    report_path = REPORTS_DIR / filename

    report = {
        "report_generated_at": datetime.utcnow().isoformat(),
        "dag_id":     dag_id,
        "task_id":    task_id,
        "run_id":     run_id,
        "diagnosis":  diagnosis,
        "log_excerpt": log_excerpt[-2000:],
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report_path


def print_report(dag_id: str, task_id: str, diagnosis: dict):
    """Print a formatted diagnosis report to the console."""
    confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(
        diagnosis.get("confidence", "low"), "⚪"
    )
    action_emoji = "🚨" if diagnosis.get("needs_immediate_action") else "ℹ️"

    print("\n" + "="*60)
    print(f"  INCIDENT DIAGNOSIS REPORT")
    print("="*60)
    print(f"  DAG:        {dag_id}")
    print(f"  Task:       {task_id}")
    print(f"  Error type: {diagnosis.get('error_type', 'unknown')}")
    print(f"  Confidence: {confidence_emoji} {diagnosis.get('confidence', 'unknown').upper()}")
    print(f"  Action:     {action_emoji} {'IMMEDIATE ACTION REQUIRED' if diagnosis.get('needs_immediate_action') else 'Monitor'}")
    print()
    print(f"  ROOT CAUSE:")
    print(f"  {diagnosis.get('root_cause', 'Unknown')}")
    print()
    print(f"  RECOMMENDED FIX:")
    # Word-wrap the fix to 55 chars
    fix = diagnosis.get('recommended_fix', 'None')
    words = fix.split()
    line, lines = [], []
    for word in words:
        if sum(len(w)+1 for w in line) + len(word) > 55:
            lines.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        lines.append(" ".join(line))
    for l in lines:
        print(f"  {l}")
    print()
    print(f"  RUNBOOK:    {diagnosis.get('relevant_runbook', 'none')}")
    print(f"  PREVENTION: {diagnosis.get('prevention_tip', 'none')[:80]}")
    print("="*60)


# ── Main monitoring loop ──────────────────────────────────────────

def run_monitoring_cycle(
    groq_client: Groq,
    retriever: PipelineKnowledgeRetriever,
    target_dag: Optional[str] = None,
) -> int:
    """
    One monitoring cycle: check all DAGs for failures and diagnose them.
    Returns the number of new failures diagnosed.
    """
    try:
        dags = get_all_dags()
    except requests.RequestException as e:
        print(f"⚠  Could not connect to Airflow: {e}")
        print("   Is Airflow running? Try: docker compose up -d")
        return 0

    if target_dag:
        dags = [d for d in dags if d["dag_id"] == target_dag]

    new_diagnoses = 0

    for dag in dags:
        dag_id = dag["dag_id"]

        try:
            runs = get_recent_dag_runs(dag_id, limit=3)
        except requests.RequestException:
            continue

        for run in runs:
            run_id    = run["dag_run_id"]
            run_state = run.get("state", "")

            if run_state != "failed":
                continue

            # Get failed tasks in this run
            try:
                failed_tasks = get_failed_tasks(dag_id, run_id)
            except requests.RequestException:
                continue

            for task in failed_tasks:
                task_id  = task["task_id"]
                cache_key = f"{dag_id}::{run_id}::{task_id}"

                # Skip if already diagnosed this session
                if cache_key in _diagnosed_runs:
                    continue

                print(f"\n🔴 NEW FAILURE DETECTED")
                print(f"   DAG:  {dag_id}")
                print(f"   Task: {task_id}")
                print(f"   Run:  {run_id}")

                # Fetch the log
                log_excerpt = get_task_log(dag_id, run_id, task_id)
                error_type  = classify_error(log_excerpt)
                print(f"   Initial classification: {error_type}")

                # Diagnose with LLM
                diagnosis = diagnose_with_llm(
                    groq_client, retriever,
                    dag_id, task_id, error_type, log_excerpt
                )

                # Print and save
                print_report(dag_id, task_id, diagnosis)
                report_path = save_report(dag_id, task_id, run_id, log_excerpt, diagnosis)
                print(f"\n  📄 Report saved: {report_path}")

                # Mark as diagnosed
                _diagnosed_runs.add(cache_key)
                new_diagnoses += 1

    return new_diagnoses


def main():
    parser = argparse.ArgumentParser(description="Pipeline monitoring agent")
    parser.add_argument("--once",  action="store_true", help="Run one cycle and exit")
    parser.add_argument("--dag",   type=str, default=None, help="Monitor a specific DAG only")
    args = parser.parse_args()

    # ── Validate API key ──────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable not set.")
        print("Run: $env:GROQ_API_KEY = 'gsk_your_key_here'")
        sys.exit(1)

    # ── Initialize clients ────────────────────────────────────────
    print("Initializing monitoring agent...")
    groq_client = Groq(api_key=api_key)

    print("Loading RAG knowledge base...")
    try:
        retriever = PipelineKnowledgeRetriever(CHROMA_PATH)
        stats = retriever.collection_stats()
        print(f"Knowledge base loaded: {stats}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("Run 'python phase3/rag/ingest.py --base-dir phase3' first")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  PIPELINE MONITORING AGENT — ACTIVE")
    print(f"{'='*60}")
    print(f"  Airflow:    {AIRFLOW_BASE_URL}")
    print(f"  LLM:        {GROQ_MODEL} (Groq)")
    print(f"  KB chunks:  {sum(stats.values())}")
    print(f"  Poll every: {POLL_INTERVAL_SEC}s")
    print(f"  Target DAG: {args.dag or 'all'}")
    print(f"  Reports:    {REPORTS_DIR}/")
    print(f"{'='*60}")

    # ── Run ───────────────────────────────────────────────────────
    if args.once:
        print("\nRunning single monitoring cycle...")
        count = run_monitoring_cycle(groq_client, retriever, args.dag)
        print(f"\nDone. {count} new failure(s) diagnosed.")
    else:
        print("\nMonitoring... (Ctrl+C to stop)\n")
        cycle = 0
        while True:
            cycle += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle #{cycle} — checking Airflow...")
            count = run_monitoring_cycle(groq_client, retriever, args.dag)
            if count == 0:
                print(f"  No new failures found.")
            print(f"  Next check in {POLL_INTERVAL_SEC}s...")
            time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()