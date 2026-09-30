-- ============================================================
-- CUSTOMER VALUE CLASSIFICATION USING SNOWFLAKE ML
-- End-to-End Machine Learning Workflow
-- ============================================================


-- ============================================================
-- 1. ENVIRONMENT SETUP
-- ============================================================

CREATE DATABASE IF NOT EXISTS CUSTOMER_CHURN_DB;

USE DATABASE CUSTOMER_CHURN_DB;

CREATE SCHEMA IF NOT EXISTS ML_SCHEMA;

USE SCHEMA ML_SCHEMA;

CREATE WAREHOUSE IF NOT EXISTS ML_WH
WAREHOUSE_SIZE = 'X-SMALL'
AUTO_SUSPEND = 60
AUTO_RESUME = TRUE;

USE WAREHOUSE ML_WH;


-- ============================================================
-- 2. EXPLORE SNOWFLAKE SAMPLE DATA
-- ============================================================

SHOW DATABASES;

SHOW SCHEMAS IN DATABASE SNOWFLAKE_SAMPLE_DATA;

SHOW TABLES IN SCHEMA SNOWFLAKE_SAMPLE_DATA.TPCH_SF1;

SELECT *
FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER
LIMIT 10;

SELECT *
FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS
LIMIT 10;


-- ============================================================
-- 3. CREATE ML DATASET
-- ============================================================

CREATE OR REPLACE TABLE CUSTOMER_ML_DATA AS

WITH CUSTOMER_ORDER_STATS AS (
    SELECT
        C.C_CUSTKEY,
        C.C_ACCTBAL,
        C.C_MKTSEGMENT,

        COUNT(O.O_ORDERKEY) AS TOTAL_ORDERS,

        COALESCE(
            SUM(O.O_TOTALPRICE),
            0
        ) AS TOTAL_SPENDING,

        COALESCE(
            AVG(O.O_TOTALPRICE),
            0
        ) AS AVG_ORDER_VALUE

    FROM SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER C

    LEFT JOIN SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS O
        ON C.C_CUSTKEY = O.O_CUSTKEY

    GROUP BY
        C.C_CUSTKEY,
        C.C_ACCTBAL,
        C.C_MKTSEGMENT
),

AVERAGE_SPENDING AS (
    SELECT
        AVG(TOTAL_SPENDING) AS AVG_SPENDING
    FROM CUSTOMER_ORDER_STATS
)

SELECT
    C.C_CUSTKEY AS CUSTOMER_ID,
    C.C_ACCTBAL,
    C.C_MKTSEGMENT,
    C.TOTAL_ORDERS,
    C.TOTAL_SPENDING,
    C.AVG_ORDER_VALUE,

    CASE
        WHEN C.TOTAL_SPENDING > A.AVG_SPENDING
        THEN 1
        ELSE 0
    END AS HIGH_VALUE

FROM CUSTOMER_ORDER_STATS C

CROSS JOIN AVERAGE_SPENDING A;


-- Check dataset

SELECT *
FROM CUSTOMER_ML_DATA
LIMIT 10;

SELECT COUNT(*) AS TOTAL_CUSTOMERS
FROM CUSTOMER_ML_DATA;

SELECT
    HIGH_VALUE,
    COUNT(*) AS CUSTOMER_COUNT
FROM CUSTOMER_ML_DATA
GROUP BY HIGH_VALUE
ORDER BY HIGH_VALUE;


-- ============================================================
-- 4. TRAIN / TEST SPLIT
-- ============================================================

CREATE OR REPLACE TABLE CUSTOMER_TRAIN AS

SELECT
    CUSTOMER_ID,
    C_ACCTBAL,
    C_MKTSEGMENT,
    TOTAL_ORDERS,
    AVG_ORDER_VALUE,
    HIGH_VALUE

FROM CUSTOMER_ML_DATA

WHERE UNIFORM(
    0::FLOAT,
    1::FLOAT,
    RANDOM()
) < 0.8;


CREATE OR REPLACE TABLE CUSTOMER_TEST AS

SELECT
    CUSTOMER_ID,
    C_ACCTBAL,
    C_MKTSEGMENT,
    TOTAL_ORDERS,
    AVG_ORDER_VALUE,
    HIGH_VALUE

FROM CUSTOMER_ML_DATA

WHERE CUSTOMER_ID NOT IN (
    SELECT CUSTOMER_ID
    FROM CUSTOMER_TRAIN
);


-- Check split

SELECT
    'TRAIN' AS DATASET,
    COUNT(*) AS ROW_COUNT
FROM CUSTOMER_TRAIN

UNION ALL

SELECT
    'TEST' AS DATASET,
    COUNT(*) AS ROW_COUNT
FROM CUSTOMER_TEST;


-- Check class distribution

SELECT
    'TRAIN' AS DATASET,
    HIGH_VALUE,
    COUNT(*) AS COUNT

FROM CUSTOMER_TRAIN

GROUP BY HIGH_VALUE

UNION ALL

SELECT
    'TEST' AS DATASET,
    HIGH_VALUE,
    COUNT(*) AS COUNT

FROM CUSTOMER_TEST

GROUP BY HIGH_VALUE

ORDER BY DATASET, HIGH_VALUE;


-- ============================================================
-- 5. FEATURE ENGINEERING
-- ============================================================

-- Training features

CREATE OR REPLACE TABLE CUSTOMER_TRAIN_FEATURES AS

SELECT
    CUSTOMER_ID,
    C_ACCTBAL,
    TOTAL_ORDERS,
    AVG_ORDER_VALUE,

    CASE
        WHEN C_MKTSEGMENT = 'AUTOMOBILE'
        THEN 1 ELSE 0
    END AS SEG_AUTOMOBILE,

    CASE
        WHEN C_MKTSEGMENT = 'BUILDING'
        THEN 1 ELSE 0
    END AS SEG_BUILDING,

    CASE
        WHEN C_MKTSEGMENT = 'FURNITURE'
        THEN 1 ELSE 0
    END AS SEG_FURNITURE,

    CASE
        WHEN C_MKTSEGMENT = 'HOUSEHOLD'
        THEN 1 ELSE 0
    END AS SEG_HOUSEHOLD,

    CASE
        WHEN C_MKTSEGMENT = 'MACHINERY'
        THEN 1 ELSE 0
    END AS SEG_MACHINERY,

    HIGH_VALUE

FROM CUSTOMER_TRAIN;


-- Testing features

CREATE OR REPLACE TABLE CUSTOMER_TEST_FEATURES AS

SELECT
    CUSTOMER_ID,
    C_ACCTBAL,
    TOTAL_ORDERS,
    AVG_ORDER_VALUE,

    CASE
        WHEN C_MKTSEGMENT = 'AUTOMOBILE'
        THEN 1 ELSE 0
    END AS SEG_AUTOMOBILE,

    CASE
        WHEN C_MKTSEGMENT = 'BUILDING'
        THEN 1 ELSE 0
    END AS SEG_BUILDING,

    CASE
        WHEN C_MKTSEGMENT = 'FURNITURE'
        THEN 1 ELSE 0
    END AS SEG_FURNITURE,

    CASE
        WHEN C_MKTSEGMENT = 'HOUSEHOLD'
        THEN 1 ELSE 0
    END AS SEG_HOUSEHOLD,

    CASE
        WHEN C_MKTSEGMENT = 'MACHINERY'
        THEN 1 ELSE 0
    END AS SEG_MACHINERY,

    HIGH_VALUE

FROM CUSTOMER_TEST;


-- Check features

SELECT *
FROM CUSTOMER_TRAIN_FEATURES
LIMIT 10;


-- ============================================================
-- 6. TRAIN SNOWFLAKE ML CLASSIFICATION MODEL
-- ============================================================

CREATE OR REPLACE SNOWFLAKE.ML.CLASSIFICATION
    CUSTOMER_VALUE_MODEL(

    INPUT_DATA =>
        SYSTEM$REFERENCE(
            'TABLE',
            'CUSTOMER_TRAIN_FEATURES'
        ),

    TARGET_COLNAME => 'HIGH_VALUE'
);


-- ============================================================
-- 7. TEST MODEL / GENERATE PREDICTIONS
-- ============================================================

SELECT
    *,
    CUSTOMER_VALUE_MODEL!PREDICT(
        INPUT_DATA => {*}
    ) AS PREDICTION

FROM CUSTOMER_TEST_FEATURES

LIMIT 10;


-- ============================================================
-- 8. CALCULATE ACCURACY
-- ============================================================

WITH PREDICTIONS AS (

    SELECT
        HIGH_VALUE,

        CUSTOMER_VALUE_MODEL!PREDICT(
            INPUT_DATA => {*}
        ):class::INT AS PREDICTED_VALUE

    FROM CUSTOMER_TEST_FEATURES
)

SELECT

    COUNT(*) AS TOTAL_TEST_ROWS,

    SUM(
        CASE
            WHEN HIGH_VALUE = PREDICTED_VALUE
            THEN 1
            ELSE 0
        END
    ) AS CORRECT_PREDICTIONS,

    ROUND(
        100.0 *
        SUM(
            CASE
                WHEN HIGH_VALUE = PREDICTED_VALUE
                THEN 1
                ELSE 0
            END
        )
        / COUNT(*),
        2
    ) AS ACCURACY_PERCENT

FROM PREDICTIONS;


-- ============================================================
-- 9. CONFUSION MATRIX
-- ============================================================

WITH PREDICTIONS AS (

    SELECT
        HIGH_VALUE,

        CUSTOMER_VALUE_MODEL!PREDICT(
            INPUT_DATA => {*}
        ):class::INT AS PREDICTED_VALUE

    FROM CUSTOMER_TEST_FEATURES
)

SELECT
    HIGH_VALUE AS ACTUAL,
    PREDICTED_VALUE AS PREDICTED,
    COUNT(*) AS COUNT

FROM PREDICTIONS

WHERE PREDICTED_VALUE IS NOT NULL

GROUP BY
    HIGH_VALUE,
    PREDICTED_VALUE

ORDER BY
    ACTUAL,
    PREDICTED;


-- ============================================================
-- 10. PRECISION, RECALL AND F1 SCORE
-- ============================================================

WITH METRICS AS (

    SELECT
        15249 AS TN,
        11 AS FP,
        7 AS FN,
        14641 AS TP
)

SELECT

    ROUND(
        100.0 * TP / (TP + FP),
        2
    ) AS PRECISION_PERCENT,

    ROUND(
        100.0 * TP / (TP + FN),
        2
    ) AS RECALL_PERCENT,

    ROUND(
        100.0 * (2 * TP)
        / (2 * TP + FP + FN),
        2
    ) AS F1_SCORE_PERCENT

FROM METRICS;


-- ============================================================
-- 11. CREATE FINAL PREDICTION TABLE
-- ============================================================

CREATE OR REPLACE TABLE CUSTOMER_PREDICTIONS AS

SELECT

    CUSTOMER_ID,

    HIGH_VALUE AS ACTUAL_VALUE,

    CUSTOMER_VALUE_MODEL!PREDICT(
        INPUT_DATA => {*}
    ):class::INT AS PREDICTED_VALUE,

    CUSTOMER_VALUE_MODEL!PREDICT(
        INPUT_DATA => {*}
    ):probability:"1"::FLOAT
        AS HIGH_VALUE_PROBABILITY

FROM CUSTOMER_TEST_FEATURES;


-- Check predictions

SELECT *
FROM CUSTOMER_PREDICTIONS
LIMIT 10;


-- ============================================================
-- 12. FINAL PREDICTION SUMMARY
-- ============================================================

SELECT
    PREDICTED_VALUE,
    COUNT(*) AS CUSTOMER_COUNT

FROM CUSTOMER_PREDICTIONS

GROUP BY PREDICTED_VALUE

ORDER BY PREDICTED_VALUE;


-- Final validation

SELECT

    COUNT(*) AS TOTAL_PREDICTIONS,

    COUNT_IF(
        PREDICTED_VALUE IS NULL
    ) AS NULL_PREDICTIONS,

    COUNT_IF(
        ACTUAL_VALUE = PREDICTED_VALUE
    ) AS CORRECT_PREDICTIONS

FROM CUSTOMER_PREDICTIONS;