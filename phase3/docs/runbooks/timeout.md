# Runbook: Task Timeout Failures

**Error type:** `task_timeout`
**Severity:** Medium

## What is a task timeout?
Task ran longer than execution_timeout and Airflow killed it.
Causes:
1. Missing database index, data volume grew too large
2. Lock contention from another transaction
3. New JOIN added without testing at scale

## Symptoms
airflow.exceptions.AirflowTaskTimeout: Timeout, PID: 12345

## Diagnosis
```sql
SELECT pid, now() - query_start AS duration, query
FROM pg_stat_activity
WHERE (now() - query_start) > interval '30 seconds';

SELECT tablename, seq_scan, idx_scan, n_live_tup
FROM pg_stat_user_tables ORDER BY seq_scan DESC LIMIT 10;

EXPLAIN ANALYZE
SELECT region, COUNT(*), SUM(amount) FROM raw_orders GROUP BY region;
```

## Fix
```sql
CREATE INDEX CONCURRENTLY idx_raw_orders_region ON raw_orders(region);
ANALYZE raw_orders;
```

## Past incidents
- 2024-02-14: raw_orders.region index dropped during migration. Fixed in 45 min.
- 2024-01-30: Runaway cross join after bad dbt merge. Reverted in 2 hours.

## Prevention
- Every GROUP BY column on tables over 100k rows needs an index
- Review EXPLAIN ANALYZE before merging any aggregation query