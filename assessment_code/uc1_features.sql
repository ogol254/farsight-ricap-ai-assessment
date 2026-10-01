-- PostgreSQL. All joins are pre-aggregated to prevent filing x payment multiplication.
WITH params AS (SELECT DATE '2026-01-01' AS cutoff),
filing_features AS (
  SELECT f.taxpayer_id,
         COUNT(*) FILTER (WHERE f.due_date >= p.cutoff - INTERVAL '24 months') AS filings_24m,
         COUNT(*) FILTER (WHERE f.due_date >= p.cutoff - INTERVAL '24 months' AND (f.filed_date IS NULL OR f.filed_date > f.due_date)) AS late_filings_24m,
         SUM(f.tax_due) FILTER (WHERE f.due_date >= p.cutoff - INTERVAL '12 months') AS tax_due_12m
  FROM filings f CROSS JOIN params p
  WHERE f.due_date < p.cutoff GROUP BY f.taxpayer_id
),
payment_features AS (
  SELECT pay.taxpayer_id, SUM(pay.amount) AS paid_12m, MAX(pay.paid_at) AS last_paid_at
  FROM payments pay CROSS JOIN params p
  WHERE pay.status = 'SUCCESS' AND pay.paid_at < p.cutoff AND pay.paid_at >= p.cutoff - INTERVAL '12 months'
  GROUP BY pay.taxpayer_id
)
SELECT t.taxpayer_id, t.tin, t.name, t.owner_gender, t.sector, t.region, t.business_size,
       COALESCE(ff.filings_24m, 0) AS filings_24m,
       COALESCE(ff.late_filings_24m, 0) AS late_filings_24m,
       CASE WHEN COALESCE(ff.tax_due_12m, 0) = 0 THEN NULL ELSE COALESCE(pf.paid_12m, 0) / ff.tax_due_12m END AS payment_ratio_12m,
       CASE WHEN pf.last_paid_at IS NULL THEN NULL ELSE (p.cutoff - pf.last_paid_at::date) END AS days_since_last_payment
FROM taxpayers t CROSS JOIN params p
LEFT JOIN filing_features ff ON ff.taxpayer_id = t.taxpayer_id
LEFT JOIN payment_features pf ON pf.taxpayer_id = t.taxpayer_id
WHERE t.status = 'ACTIVE';

