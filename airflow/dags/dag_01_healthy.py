"""
dag_01_healthy.py
─────────────────
A pipeline that works perfectly every time.
This is your baseline — what a healthy run looks like.

What it does:
  1. Extracts raw_orders from Postgres
  2. Validates the data (row count, null checks)
  3. Loads clean records into stg_orders
  4. Calculates daily revenue into mart_daily_revenue

Run it: In Airflow UI → toggle ON → Trigger DAG ▶
"""

from datetime import datetime, timedelta
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

logger = logging.getLogger(__name__)

# ── DAG default args ─────────────────────────────────────────────
default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

# ── Task functions ────────────────────────────────────────────────

def extract_raw_orders(**context):
    """Pull orders from raw table, push to XCom for next task."""
    hook = PostgresHook(postgres_conn_id="pipeline_db")
    records = hook.get_records("""
        SELECT order_id, customer_id, amount, status, created_at, region
        FROM raw_orders
        WHERE status != 'failed'
    """)
    logger.info(f"Extracted {len(records)} raw orders")
    # Push to XCom so downstream tasks can use it
    context["ti"].xcom_push(key="raw_order_count", value=len(records))
    return records


def validate_orders(**context):
    """
    Data quality checks. This task intentionally passes every time.
    Compare with dag_02_schema_error.py to see what failure looks like.
    """
    ti = context["ti"]
    records = ti.xcom_pull(task_ids="extract_raw_orders")

    if not records:
        raise ValueError("No records extracted — something is wrong upstream!")

    # Check 1: row count
    assert len(records) > 0, "Row count check failed: 0 rows found"
    logger.info(f"✓ Row count check passed: {len(records)} rows")

    # Check 2: no null order_ids
    null_ids = [r for r in records if r[0] is None]
    assert len(null_ids) == 0, f"Null order_id check failed: {len(null_ids)} nulls"
    logger.info("✓ Null check passed: no null order_ids")

    # Check 3: amounts are positive
    bad_amounts = [r for r in records if r[2] is not None and r[2] < 0]
    assert len(bad_amounts) == 0, f"Amount check failed: {len(bad_amounts)} negative amounts"
    logger.info("✓ Amount check passed: all amounts positive")

    logger.info("All validation checks passed ✓")


def load_to_staging(**context):
    """Insert validated records into stg_orders."""
    ti = context["ti"]
    records = ti.xcom_pull(task_ids="extract_raw_orders")

    hook = PostgresHook(postgres_conn_id="pipeline_db")
    conn = hook.get_conn()
    cursor = conn.cursor()

    # Clear today's data before re-loading (idempotent)
    cursor.execute("DELETE FROM stg_orders WHERE _loaded_at::date = CURRENT_DATE")

    insert_sql = """
        INSERT INTO stg_orders (order_id, customer_id, amount, status, created_at, region)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (order_id) DO UPDATE SET
            status = EXCLUDED.status,
            _loaded_at = NOW()
    """
    cursor.executemany(insert_sql, records)
    conn.commit()
    cursor.close()

    logger.info(f"Loaded {len(records)} records into stg_orders")


def calculate_daily_revenue(**context):
    """Aggregate stg_orders into mart_daily_revenue."""
    hook = PostgresHook(postgres_conn_id="pipeline_db")
    conn = hook.get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO mart_daily_revenue (report_date, region, total_revenue, order_count)
        SELECT
            created_at::date AS report_date,
            region,
            SUM(amount)      AS total_revenue,
            COUNT(*)         AS order_count
        FROM stg_orders
        WHERE status = 'completed'
        GROUP BY created_at::date, region
        ON CONFLICT (report_date) DO UPDATE SET
            total_revenue = EXCLUDED.total_revenue,
            order_count   = EXCLUDED.order_count,
            _updated_at   = NOW()
    """)
    conn.commit()
    cursor.close()

    logger.info("Daily revenue mart updated successfully ✓")


# ── DAG definition ────────────────────────────────────────────────

with DAG(
    dag_id="dag_01_healthy_pipeline",
    description="Baseline healthy pipeline — works every time",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["baseline", "healthy"],
) as dag:

    t1 = PythonOperator(task_id="extract_raw_orders",     python_callable=extract_raw_orders)
    t2 = PythonOperator(task_id="validate_orders",        python_callable=validate_orders)
    t3 = PythonOperator(task_id="load_to_staging",        python_callable=load_to_staging)
    t4 = PythonOperator(task_id="calculate_daily_revenue",python_callable=calculate_daily_revenue)

    # Define order: extract → validate → load → aggregate
    t1 >> t2 >> t3 >> t4