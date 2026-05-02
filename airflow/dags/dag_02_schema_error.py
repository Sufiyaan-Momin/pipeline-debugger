"""
dag_02_schema_error.py
──────────────────────
Simulates a SCHEMA MISMATCH failure.

Real-world scenario:
  An upstream team renamed a column in raw_orders.
  They changed `region` → `geo_region` without telling anyone.
  Our pipeline breaks because it tries to SELECT a column that no longer exists.

What your monitoring agent should detect:
  - Task: validate_orders or load_to_staging
  - Error type: UndefinedColumn / KeyError
  - Root cause: schema drift — upstream column renamed
  - Fix: update SELECT query or coordinate with upstream team

How to trigger the failure:
  1. In Airflow UI, find dag_02_schema_mismatch
  2. Toggle it ON
  3. Click Trigger DAG ▶
  4. Watch validate_orders fail
  5. Check the logs — your agent will read these

LEARNING NOTE:
  This is one of the most common real-world pipeline failures.
  dbt handles this better by auto-documenting schema — that's Phase 3.
"""

from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

logger = logging.getLogger(__name__)

default_args = {
    "owner": "data-team",
    "retries": 0,           # No retries — we want it to fail fast
    "email_on_failure": False,
}


def extract_with_wrong_schema(**context):
    """
    This task queries a column name that doesn't exist.
    In a real scenario: upstream renamed 'region' to 'geo_region'
    and didn't tell the data team.
    """
    hook = PostgresHook(postgres_conn_id="pipeline_db")

    # ❌ BUG: 'geo_region' does not exist — should be 'region'
    # This will throw: psycopg2.errors.UndefinedColumn
    try:
        records = hook.get_records("""
            SELECT order_id, customer_id, amount, status, created_at, geo_region
            FROM raw_orders
        """)
        context["ti"].xcom_push(key="records", value=records)
        return records
    except Exception as e:
        logger.error("=" * 60)
        logger.error("SCHEMA ERROR DETECTED")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Details: {str(e)}")
        logger.error("Expected column 'geo_region' but column may be named 'region'")
        logger.error("Upstream schema may have changed without notification")
        logger.error("=" * 60)
        raise   # Re-raise so Airflow marks the task as FAILED


def validate_schema_exists(**context):
    """Checks that expected columns are present before doing work."""
    hook = PostgresHook(postgres_conn_id="pipeline_db")

    # Get actual columns from the table
    actual_columns = hook.get_records("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'raw_orders'
        ORDER BY ordinal_position
    """)
    actual_col_names = [row[0] for row in actual_columns]
    logger.info(f"Actual columns in raw_orders: {actual_col_names}")

    # Columns our pipeline EXPECTS
    expected_columns = ['order_id', 'customer_id', 'amount', 'status', 'created_at', 'geo_region']

    missing = [col for col in expected_columns if col not in actual_col_names]
    if missing:
        raise ValueError(
            f"SCHEMA MISMATCH: Expected columns {missing} not found in raw_orders. "
            f"Actual columns: {actual_col_names}. "
            f"Possible cause: upstream renamed 'region' to 'geo_region' or vice versa."
        )


def load_task(**context):
    """Won't run because extract failed. Shows dependency chain in logs."""
    logger.info("This task would load data, but it never runs due to upstream failure.")


with DAG(
    dag_id="dag_02_schema_mismatch",
    description="Failure: column renamed upstream — schema drift",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # Manual trigger only
    catchup=False,
    tags=["failure", "schema-error"],
) as dag:

    t1 = PythonOperator(task_id="validate_schema_exists", python_callable=validate_schema_exists)
    t2 = PythonOperator(task_id="extract_with_wrong_schema", python_callable=extract_with_wrong_schema)
    t3 = PythonOperator(task_id="load_to_staging", python_callable=load_task)

    t1 >> t2 >> t3