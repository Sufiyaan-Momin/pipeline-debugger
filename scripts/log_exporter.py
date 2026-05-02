"""
scripts/log_exporter.py
───────────────────────
Reads Airflow task logs and exports them to a structured JSON file
that your Phase 3 RAG agent will ingest into ChromaDB.

WHAT THIS DOES:
  1. Calls the Airflow REST API to get recent DAG runs
  2. For each failed run, fetches the task logs
  3. Saves structured incident records to data/incidents/

HOW TO RUN:
  Make sure Airflow is running (docker compose up -d), then:
  pip install requests
  python scripts/log_exporter.py

AIRFLOW REST API DOCS:
  http://localhost:8080/api/v1/ui  (interactive docs while Airflow is running)

This script is the BRIDGE between Phase 2 (pipeline) and Phase 3 (RAG agent).
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import requests

# ── Config ────────────────────────────────────────────────────────
AIRFLOW_BASE_URL = "http://localhost:8080/api/v1"
AIRFLOW_USER = "admin"
AIRFLOW_PASS = "admin"
OUTPUT_DIR = Path("data/incidents")

AUTH = (AIRFLOW_USER, AIRFLOW_PASS)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────

def get_all_dags() -> list[dict]:
    """Fetch list of all DAGs from Airflow."""
    resp = requests.get(f"{AIRFLOW_BASE_URL}/dags", auth=AUTH)
    resp.raise_for_status()
    return resp.json()["dags"]


def get_dag_runs(dag_id: str, limit: int = 10) -> list[dict]:
    """Get recent runs for a DAG."""
    resp = requests.get(
        f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns",
        params={"limit": limit, "order_by": "-start_date"},
        auth=AUTH,
    )
    resp.raise_for_status()
    return resp.json()["dag_runs"]


def get_task_instances(dag_id: str, dag_run_id: str) -> list[dict]:
    """Get all task instances for a specific DAG run."""
    resp = requests.get(
        f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances",
        auth=AUTH,
    )
    resp.raise_for_status()
    return resp.json()["task_instances"]


def get_task_log(dag_id: str, dag_run_id: str, task_id: str, try_number: int = 1) -> str:
    """Fetch raw log text for a specific task attempt."""
    resp = requests.get(
        f"{AIRFLOW_BASE_URL}/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{try_number}",
        auth=AUTH,
        headers={"Accept": "text/plain"},
    )
    if resp.status_code == 200:
        return resp.text
    return f"[Log not available: HTTP {resp.status_code}]"


def classify_error(log_text: str) -> dict:
    """
    Simple rule-based error classifier.
    In Phase 4, your LLM agent will replace/augment this.

    Returns:
      error_type: category of error
      error_detail: specific message extracted from log
      suggested_fix: initial fix hint (your RAG agent will improve this)
    """
    log_lower = log_text.lower()

    # Schema errors
    if "undefinedcolumn" in log_lower or "column" in log_lower and "does not exist" in log_lower:
        return {
            "error_type": "schema_mismatch",
            "error_detail": extract_error_line(log_text, "column"),
            "suggested_fix": "Check if upstream column was renamed. Compare schema.yml with actual table columns.",
        }

    # Data quality errors
    if "assertionerror" in log_lower or "not null check" in log_lower or "data quality" in log_lower:
        return {
            "error_type": "data_quality_failure",
            "error_detail": extract_error_line(log_text, "FAILED"),
            "suggested_fix": "Run quality checks manually. Check raw_orders for NULL order_ids or negative amounts.",
        }

    # Timeout errors
    if "timeout" in log_lower or "airflowtasktimeout" in log_lower:
        return {
            "error_type": "task_timeout",
            "error_detail": "Task exceeded execution_timeout",
            "suggested_fix": "Check for missing indexes. Run EXPLAIN ANALYZE on the failing query. Consider increasing timeout if data volume grew.",
        }

    # Connection errors
    if "connection refused" in log_lower or "could not connect" in log_lower:
        return {
            "error_type": "connection_failure",
            "error_detail": extract_error_line(log_text, "connection"),
            "suggested_fix": "Check database is running. Verify Airflow connection settings in Admin > Connections.",
        }

    return {
        "error_type": "unknown",
        "error_detail": "Could not classify error automatically",
        "suggested_fix": "Review full log manually",
    }


def extract_error_line(log_text: str, keyword: str) -> str:
    """Pull the first line containing a keyword — good for error messages."""
    for line in log_text.split("\n"):
        if keyword.lower() in line.lower():
            return line.strip()[:300]   # cap at 300 chars
    return "No matching line found"


# ── Main export logic ─────────────────────────────────────────────

def export_failed_runs():
    """
    Main function: find all failed tasks across all DAGs
    and save structured incident records to disk.
    """
    print("Connecting to Airflow...")
    dags = get_all_dags()
    print(f"Found {len(dags)} DAGs")

    all_incidents = []

    for dag in dags:
        dag_id = dag["dag_id"]
        print(f"\nChecking DAG: {dag_id}")

        try:
            runs = get_dag_runs(dag_id, limit=5)
        except requests.HTTPError as e:
            print(f"  Could not fetch runs: {e}")
            continue

        for run in runs:
            run_id = run["dag_run_id"]
            run_state = run["state"]

            if run_state not in ("failed", "success"):
                print(f"  Run {run_id}: {run_state} (skipping)")
                continue

            print(f"  Run {run_id}: {run_state}")
            task_instances = get_task_instances(dag_id, run_id)

            for task in task_instances:
                if task["state"] != "failed":
                    continue

                task_id = task["task_id"]
                print(f"    Failed task: {task_id}")

                # Fetch the log
                log_text = get_task_log(dag_id, run_id, task_id)

                # Classify the error
                classification = classify_error(log_text)

                # Build the incident record
                incident = {
                    "incident_id": f"{dag_id}__{task_id}__{run_id}",
                    "dag_id": dag_id,
                    "task_id": task_id,
                    "run_id": run_id,
                    "run_state": run_state,
                    "start_date": run.get("start_date"),
                    "end_date": run.get("end_date"),
                    "error_type": classification["error_type"],
                    "error_detail": classification["error_detail"],
                    "suggested_fix": classification["suggested_fix"],
                    "log_excerpt": log_text[-3000:],   # last 3000 chars (most relevant)
                    "exported_at": datetime.utcnow().isoformat(),
                    # This field is empty now — your agent will fill it in Phase 4
                    "resolution": None,
                    "resolution_confirmed": False,
                }

                all_incidents.append(incident)

                # Save individual incident file (one per failure)
                safe_id = incident['incident_id'].replace(":", "-").replace("+", "").replace("T00-00-00", "")
                incident_path = OUTPUT_DIR / f"{safe_id}.json"
                with open(incident_path, "w") as f:
                    json.dump(incident, f, indent=2, default=str)
                print(f"    Saved: {incident_path}")

    # Also save a combined file for easy browsing
    combined_path = OUTPUT_DIR / "_all_incidents.json"
    with open(combined_path, "w") as f:
        json.dump(all_incidents, f, indent=2, default=str)

    print(f"\n{'='*50}")
    print(f"Export complete: {len(all_incidents)} incidents saved to {OUTPUT_DIR}/")
    print(f"Combined file: {combined_path}")
    print("\nNext step: Run Phase 3 RAG ingestion on the data/incidents/ folder")


if __name__ == "__main__":
    export_failed_runs()