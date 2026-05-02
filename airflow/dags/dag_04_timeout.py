"""
dag_04_timeout_failure.py
─────────────────────────
Simulates a TIMEOUT failure.

Real-world scenario:
  A query that used to take 2 seconds now takes forever because:
  - Someone dropped an index on a large table
  - Data volume grew 10x without query optimization
  - A lock is held by another transaction

What your monitoring agent should detect:
  - Task: slow_aggregation_query
  - Error type: AirflowTaskTimeout
  - Root cause: missing index / data volume increase / lock contention
  - Fix: add index, optimize query, or investigate blocking transactions

LEARNING NOTE:
  Airflow uses execution_timeout on the task level to catch these.
  In production, slow queries are often the hardest to debug because
  they don't produce a clear error — just silence, then a timeout.
"""

from datetime import datetime, timedelta
import logging
import time

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook

logger = logging.getLogger(__name__)

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(seconds=30),
    "email_on_failure": False,
}


def fast_extract(**context):
    """This task completes quickly — no issues here."""
    hook = PostgresHook(postgres_conn_id="pipeline_db")
    count = hook.get_first("SELECT COUNT(*) FROM raw_orders")[0]
    logger.info(f"Fast extract: found {count} orders ✓")
    return count


def slow_aggregation_query(**context):
    """
    Simulates a query that times out.

    In real life this would be a massive GROUP BY or window function
    over millions of rows with no index. We simulate it with sleep().

    The task has execution_timeout=10s, but this function sleeps for 30s.
    Airflow will kill it and mark it FAILED with AirflowTaskTimeout.
    """
    logger.info("Starting aggregation query...")
    logger.info("Query: SELECT region, COUNT(*), SUM(amount) FROM raw_orders GROUP BY region")
    logger.info("WARNING: This query is running on a table with no index on 'region'")
    logger.info("Estimated rows to scan: 50,000,000")

    # Simulate a slow query — in production this would be an actual slow SQL query
    for i in range(1, 31):
        logger.info(f"Query running... {i}s elapsed (timeout at 10s)")
        time.sleep(1)

    # This line is never reached — Airflow kills us first
    logger.info("Query completed (you should never see this)")


def post_aggregation(**context):
    """Would store results — never runs due to timeout."""
    logger.info("This task would store aggregation results.")


with DAG(
    dag_id="dag_04_timeout_failure",
    description="Failure: query timeout due to missing index / data volume",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["failure", "timeout"],
) as dag:

    t1 = PythonOperator(
        task_id="fast_extract",
        python_callable=fast_extract,
    )

    t2 = PythonOperator(
        task_id="slow_aggregation_query",
        python_callable=slow_aggregation_query,
        execution_timeout=timedelta(seconds=10),   # Will timeout after 10s
    )

    t3 = PythonOperator(
        task_id="post_aggregation",
        python_callable=post_aggregation,
    )

    t1 >> t2 >> t3