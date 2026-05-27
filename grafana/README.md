# Grafana Dashboard — Export & Setup Guide

## Overview

`export_to_sqlite.py` reads the forecast CSV outputs and writes them into a single SQLite database at `data/grafana/export_opportunities.db`.  
Grafana reads that database via the **SQLite data source plugin**, and each panel runs a SQL query against one of the tables.

---

## Step 1 — Generate the database

Run this from the project root (after the forecast pipeline has produced its CSVs):

```bash
python grafana/export_to_sqlite.py
```

Expected output:

```
[top_forecasted_opportunities]
  Imported 50 rows → table 'top_forecasted_opportunities'

[priority_countries]
  Imported 20 rows → table 'priority_countries'

...

Database written to: data/grafana/export_opportunities.db
Tables imported: 6/6
```

Re-run any time the forecast CSVs are refreshed.

---

## Step 2 — Install Grafana

Download from https://grafana.com/grafana/download and start the server (default port 3000).

---

## Step 3 — Install the SQLite plugin

```bash
grafana-cli plugins install frser-sqlite-datasource
# then restart Grafana
```

Or install from the Grafana UI: **Plugins → Search "SQLite" → Install**.

---

## Step 4 — Add the data source

1. Open Grafana → **Configuration → Data Sources → Add data source**.
2. Search for **SQLite** and select it.
3. Set **Path** to the absolute path of the database file, e.g.:

   ```
   /absolute/path/to/ml-project/data/grafana/export_opportunities.db
   ```

4. Click **Save & Test** — you should see "Database connected".

---

## Step 5 — Create dashboard panels

Add a new dashboard and use the SQL queries below as panel queries.

---

## Example SQL Queries

### Top Forecasted Opportunities

```sql
SELECT
    country,
    product_code,
    forecasted_value,
    growth_rate_pct
FROM top_forecasted_opportunities
ORDER BY forecasted_value DESC
LIMIT 20;
```

### Priority Countries by Forecast Value

```sql
SELECT
    country,
    SUM(forecasted_value) AS total_forecasted_value,
    AVG(growth_rate_pct)  AS avg_growth_rate
FROM priority_countries
GROUP BY country
ORDER BY total_forecasted_value DESC;
```

### Priority Products

```sql
SELECT
    product_code,
    product_name,
    forecasted_value,
    priority_score
FROM priority_products
ORDER BY priority_score DESC
LIMIT 15;
```

### Forecast vs Actual (Historical Comparison)

```sql
SELECT
    period,
    product_code,
    actual_value,
    forecasted_value,
    (forecasted_value - actual_value) AS error
FROM historical_forecast_comparison
ORDER BY period ASC;
```

### Model Metrics Summary

```sql
SELECT
    model_name,
    ROUND(mae,  4) AS mae,
    ROUND(rmse, 4) AS rmse,
    ROUND(mape, 4) AS mape_pct,
    ROUND(r2,   4) AS r2
FROM forecast_model_metrics
ORDER BY rmse ASC;
```

### Final Forecasts Timeline

```sql
SELECT
    forecast_date,
    country,
    product_code,
    forecasted_value
FROM final_forecasts
ORDER BY forecast_date ASC;
```

---

## Tables Reference

| Table | Source CSV |
|---|---|
| `top_forecasted_opportunities` | `data/forecast_outputs/top_forecasted_opportunities.csv` |
| `priority_countries` | `data/forecast_outputs/priority_countries.csv` |
| `priority_products` | `data/forecast_outputs/priority_products.csv` |
| `final_forecasts` | `data/forecast_outputs/final_forecasts.csv` |
| `forecast_model_metrics` | `data/forecast_outputs/forecast_model_metrics.csv` |
| `historical_forecast_comparison` | `data/forecast_outputs/historical_forecast_comparison.csv` |

Product code columns (`k`, `product_code`, `hs_code`) are stored as TEXT to prevent leading-zero truncation.

---

## Backup Dashboard

The original Streamlit dashboard is preserved at `dashboard/app.py` and can be run independently:

```bash
streamlit run dashboard/app.py
```
