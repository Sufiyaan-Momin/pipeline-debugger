# Runbook: Data Quality Failures

**Error type:** `data_quality_failure`
**Severity:** High

## What is a data quality failure?
Data arrived with invalid values violating our contracts.
Common causes:
1. NULL primary keys from upstream service bug
2. Negative amounts from broken refund logic
3. Orphaned foreign keys after customer deletion

## Symptoms
AssertionError: DATA QUALITY GATE FAILED
NOT NULL CHECK FAILED: 2 rows have NULL order_id
POSITIVE VALUE CHECK FAILED: 1 rows have negative amount
REFERENTIAL INTEGRITY FAILED: 1 orders reference non-existent customer_ids

## Diagnosis
```sql
SELECT * FROM raw_orders WHERE order_id IS NULL LIMIT 10;
SELECT * FROM raw_orders WHERE amount < 0 LIMIT 10;
SELECT o.* FROM raw_orders o
LEFT JOIN raw_customers c ON o.customer_id = c.customer_id
WHERE c.customer_id IS NULL LIMIT 10;
```

## Fix
```sql
CREATE TABLE IF NOT EXISTS quarantine_orders
    AS SELECT * FROM raw_orders WHERE 1=0;
INSERT INTO quarantine_orders
    SELECT * FROM raw_orders WHERE order_id IS NULL;
DELETE FROM raw_orders WHERE order_id IS NULL;
```

## Prevention
- Run dbt not_null and relationships tests before mart models
- Route bad records to quarantine instead of halting pipeline