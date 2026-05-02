"""
dag_03_data_quality_failure.py
──────────────────────────────
Simulates a DATA QUALITY failure.

Real-world scenario:
  A new data source started sending corrupted records:
  - Some order_ids are NULL (can't be primary key)
  - Some amounts are negative (invalid — refunds handled separately)
  - customer_ids reference customers that don't exist (referential integrity broken)

What your monitoring agent should detect:
  - Task: run_quality_checks
  - Error type: AssertionError / DataQualityError
  - Root cause: upstream data quality degraded
  - Fix: add NOT NULL constraint upstream, alert source team, quarantine bad records

This is a GREAT type of error to handle because:
  - It's extremely common in real pipelines
  - The fix requires domain knowledge (not just SQL)
  - Your RAG agent can retrieve the runbook for "null order_id" incidents
"""

from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

logger = logging.getLogger(__name__)

default_args = {
    "owner": "data-team",
    "retries": 0,
    "email_on_failure": False,
}


def inject_bad_data(**context):
    """
    Inserts deliberately bad records into raw_orders to simulate
    a corrupt upstream feed. In production this would happen automatically
    from a broken upstream system.

    Run this task first, then the quality check will fail.
    """
    hook = PostgresHook(postgres_conn_id="pipeline_db")
    conn = hook.get_conn()
    cursor = conn.cursor()

    bad_records = [
        # NULL order_id — can't be used as primary key
        (None,   1,   50.00, 'completed', datetime.now(), 'US'),
        # Negative amount — invalid order value
        ('ORD-BAD-001', 2, -999.99, 'completed', datetime.now(), 'EU'),
        # customer_id 9999 doesn't exist in raw_customers
        ('ORD-BAD-002', 9999, 75.00, 'completed', datetime.now(), 'US'),
        # Another NULL order_id
        (None,   3,  120.00, 'pending',   datetime.now(), 'APAC'),
    ]

    cursor.executemany("""
        INSERT INTO raw_orders (order_id, customer_id, amount, status, created_at, region)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, bad_records)
    conn.commit()
    cursor.close()

    logger.warning(f"Injected {len(bad_records)} bad records into raw_orders")
    logger.warning("These will cause data quality checks to fail")


def run_quality_checks(**context):
    """
    Runs dbt-style data quality tests:
      - not_null on order_id
      - positive_values on amount
      - referential integrity: customer_id must exist in raw_customers

    This task will FAIL and your agent should diagnose why.
    """
    hook = PostgresHook(postgres_conn_id="pipeline_db")
    errors = []

    # ── Test 1: Not null on order_id ─────────────────────────────
    null_orders = hook.get_first("""
        SELECT COUNT(*) FROM raw_orders WHERE order_id IS NULL
    """)[0]
    if null_orders > 0:
        msg = f"NOT NULL CHECK FAILED: {null_orders} rows have NULL order_id in raw_orders"
        logger.error(msg)
        errors.append(msg)
    else:
        logger.info("✓ Not null check passed: order_id")

    # ── Test 2: Positive amounts ──────────────────────────────────
    negative_amounts = hook.get_first("""
        SELECT COUNT(*) FROM raw_orders WHERE amount < 0
    """)[0]
    if negative_amounts > 0:
        msg = f"POSITIVE VALUE CHECK FAILED: {negative_amounts} rows have negative amount in raw_orders"
        logger.error(msg)
        errors.append(msg)
    else:
        logger.info("✓ Positive value check passed: amount")

    # ── Test 3: Referential integrity ─────────────────────────────
    orphaned = hook.get_first("""
        SELECT COUNT(*)
        FROM raw_orders o
        LEFT JOIN raw_customers c ON o.customer_id = c.customer_id
        WHERE o.customer_id IS NOT NULL
          AND c.customer_id IS NULL
    """)[0]
    if orphaned > 0:
        msg = f"REFERENTIAL INTEGRITY FAILED: {orphaned} orders reference non-existent customer_ids"
        logger.error(msg)
        errors.append(msg)
    else:
        logger.info("✓ Referential integrity check passed: customer_id")

    # ── Raise if any checks failed ────────────────────────────────
    if errors:
        error_summary = "\n".join(errors)
        raise AssertionError(
            f"DATA QUALITY GATE FAILED — {len(errors)} check(s) failed:\n{error_summary}\n"
            f"Pipeline halted. Bad records must be quarantined before loading to staging."
        )

    logger.info("All data quality checks passed ✓")


def quarantine_bad_records(**context):
    """
    In a real pipeline you'd move bad records to a quarantine table.
    This task only runs if quality checks pass (it won't in our failure scenario).
    """
    logger.info("Quality checks passed — proceeding with clean data load")


def cleanup_injected_data(**context):
    """
    Helper: removes the bad records we injected.
    Run this manually to reset the scenario and try again.
    Trigger just this task: Airflow UI → dag_03 → cleanup_injected_data → Run
    """
    hook = PostgresHook(postgres_conn_id="pipeline_db")
    conn = hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM raw_orders WHERE order_id LIKE 'ORD-BAD-%' OR order_id IS NULL")
    conn.commit()
    cursor.close()
    logger.info("Cleaned up injected bad records ✓")


with DAG(
    dag_id="dag_03_data_quality_failure",
    description="Failure: null order_ids + negative amounts + referential integrity",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["failure", "data-quality"],
) as dag:

    inject   = PythonOperator(task_id="inject_bad_data",         python_callable=inject_bad_data)
    checks   = PythonOperator(task_id="run_quality_checks",      python_callable=run_quality_checks)
    load     = PythonOperator(task_id="load_clean_records",      python_callable=quarantine_bad_records)
    cleanup  = PythonOperator(task_id="cleanup_injected_data",   python_callable=cleanup_injected_data,
                              trigger_rule="all_done")  # runs even if checks fail

    inject >> checks >> load
    checks >> cleanup  # always runs so you can reset