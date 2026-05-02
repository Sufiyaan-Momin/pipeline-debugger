WITH source AS (
    SELECT * FROM {{ source('pipeline_db', 'raw_orders') }}
),
cleaned AS (
    SELECT
        TRIM(order_id)                          AS order_id,
        customer_id,
        COALESCE(amount, 0.00)::NUMERIC(10, 2)  AS amount,
        UPPER(TRIM(status))                     AS status,
        created_at::TIMESTAMP                   AS created_at,
        DATE_TRUNC('day', created_at)::DATE     AS created_date,
        UPPER(TRIM(region))                     AS region,
        CURRENT_TIMESTAMP                       AS _loaded_at
    FROM source
    WHERE order_id IS NOT NULL
),
with_flags AS (
    SELECT *,
        CASE
            WHEN amount < 0        THEN 'negative_amount'
            WHEN amount = 0        THEN 'zero_amount'
            WHEN status NOT IN ('COMPLETED','PENDING','FAILED','CANCELLED') THEN 'unknown_status'
            ELSE NULL
        END AS _data_quality_flag
    FROM cleaned
)
SELECT * FROM with_flags