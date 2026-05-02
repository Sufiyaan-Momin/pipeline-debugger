# Runbook: Schema Mismatch Errors

**Error type:** `schema_mismatch`
**Severity:** High

## What is a schema mismatch?
A schema mismatch occurs when our pipeline expects a column that no longer
exists in the source database. Almost always caused by an upstream team
renaming or dropping a column without notifying data engineering.

## Symptoms
psycopg2.errors.UndefinedColumn: column "geo_region" does not exist
ValueError: SCHEMA MISMATCH: Expected columns ['geo_region'] not found

## Diagnosis steps
1. Identify the missing column from the error message
2. Check actual table schema:
```sql
SELECT column_name FROM information_schema.columns
WHERE table_name = 'raw_orders' ORDER BY ordinal_position;
```
3. Compare with schema.yml
4. Search Slack #platform-eng for rename announcements

## Known incidents
- 2024-03-15: Platform team renamed region to geo_region temporarily.
  Failed for 2 hours. Fixed by reverting column name.
- 2024-01-08: customer_tier moved from raw_orders to raw_customers.
  Fix: updated SELECT to join raw_customers.

## Fix
```sql
SELECT geo_region AS region FROM raw_orders
```

## Prevention
- Add schema validation as first task in every DAG
- Subscribe to #data-schema-changes Slack channel