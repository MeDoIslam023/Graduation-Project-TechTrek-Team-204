-- ============================================================
-- AI Credit Risk & Loan Default Intelligence Platform
-- Analytical SQL Queries (10+) — JOINs, CTEs, window functions
-- Tested against SQLite (see notebook Section 2); portable to
-- PostgreSQL/MySQL with minor syntax tweaks (e.g. NTILE, RANK).
-- ============================================================

-- 1. Overall default rate
SELECT
  COUNT(*) AS total_applicants,
  SUM(TARGET) AS defaulters,
  ROUND(100.0 * SUM(TARGET) / COUNT(*), 2) AS default_rate_pct
FROM application_train;

-- 2. Default rate by income type
SELECT NAME_INCOME_TYPE,
       COUNT(*) AS n,
       ROUND(100.0 * SUM(TARGET) / COUNT(*), 2) AS default_rate_pct
FROM application_train
GROUP BY NAME_INCOME_TYPE
ORDER BY default_rate_pct DESC;

-- 3. Default rate and average income by education level
SELECT NAME_EDUCATION_TYPE,
       COUNT(*) AS n,
       ROUND(AVG(AMT_INCOME_TOTAL), 0) AS avg_income,
       ROUND(100.0 * SUM(TARGET) / COUNT(*), 2) AS default_rate_pct
FROM application_train
GROUP BY NAME_EDUCATION_TYPE
ORDER BY default_rate_pct DESC;

-- 4. JOIN: applicants with their bureau credit history count
SELECT a.SK_ID_CURR, a.TARGET, COUNT(b.SK_ID_BUREAU) AS n_bureau_credits
FROM application_train a
LEFT JOIN bureau b ON a.SK_ID_CURR = b.SK_ID_CURR
GROUP BY a.SK_ID_CURR, a.TARGET;

-- 5. Applicants with overdue bureau credit
SELECT a.SK_ID_CURR, a.TARGET, b.CREDIT_ACTIVE, b.AMT_CREDIT_SUM_OVERDUE
FROM application_train a
JOIN bureau b ON a.SK_ID_CURR = b.SK_ID_CURR
WHERE b.AMT_CREDIT_SUM_OVERDUE > 0;

-- 6. CTE: average previous application amount per applicant
WITH prev_agg AS (
  SELECT SK_ID_CURR, AVG(AMT_APPLICATION) AS avg_prev_amt, COUNT(*) AS n_prev
  FROM previous_application
  GROUP BY SK_ID_CURR
)
SELECT a.SK_ID_CURR, a.TARGET, p.avg_prev_amt, p.n_prev
FROM application_train a
JOIN prev_agg p ON a.SK_ID_CURR = p.SK_ID_CURR
ORDER BY p.n_prev DESC;

-- 7. Window function: rank applicants by income within each family status
SELECT SK_ID_CURR, NAME_FAMILY_STATUS, AMT_INCOME_TOTAL,
       RANK() OVER (PARTITION BY NAME_FAMILY_STATUS ORDER BY AMT_INCOME_TOTAL DESC) AS income_rank
FROM application_train;

-- 8. Window function: income deciles
SELECT SK_ID_CURR, AMT_INCOME_TOTAL, TARGET,
       NTILE(10) OVER (ORDER BY AMT_INCOME_TOTAL) AS income_decile
FROM application_train;

-- 9. Late payment ratio per applicant
SELECT SK_ID_CURR,
       COUNT(*) AS n_installments,
       SUM(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END) AS n_late,
       ROUND(100.0 * SUM(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_pct
FROM installments_payments
GROUP BY SK_ID_CURR
ORDER BY late_pct DESC;

-- 10. Combined risk view: bureau overdue + late installments + target
WITH bureau_overdue AS (
  SELECT SK_ID_CURR, SUM(AMT_CREDIT_SUM_OVERDUE) AS total_overdue
  FROM bureau GROUP BY SK_ID_CURR
),
late_pay AS (
  SELECT SK_ID_CURR,
         SUM(CASE WHEN DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT THEN 1 ELSE 0 END) AS n_late
  FROM installments_payments GROUP BY SK_ID_CURR
)
SELECT a.SK_ID_CURR, a.TARGET,
       COALESCE(bo.total_overdue, 0) AS total_overdue,
       COALESCE(lp.n_late, 0) AS n_late_payments
FROM application_train a
LEFT JOIN bureau_overdue bo ON a.SK_ID_CURR = bo.SK_ID_CURR
LEFT JOIN late_pay lp ON a.SK_ID_CURR = lp.SK_ID_CURR
ORDER BY total_overdue DESC;

-- 11. Refusal rate on previous applications, joined to current target
WITH refusal_agg AS (
  SELECT SK_ID_CURR,
         COUNT(*) AS n_prev,
         SUM(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END) AS n_refused
  FROM previous_application
  GROUP BY SK_ID_CURR
)
SELECT a.SK_ID_CURR, a.TARGET, r.n_prev, r.n_refused,
       ROUND(100.0 * r.n_refused / r.n_prev, 2) AS refusal_rate_pct
FROM application_train a
JOIN refusal_agg r ON a.SK_ID_CURR = r.SK_ID_CURR
WHERE r.n_prev > 0
ORDER BY refusal_rate_pct DESC;
