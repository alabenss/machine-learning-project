# Grafana Dashboard Setup

This folder contains the SQLite export workflow for the final professional dashboard.

Grafana is the dashboard used for the final demo because it satisfies the project PDF requirement for a professional visualization dashboard. The Streamlit dashboard is still kept as a backup/live Python dashboard in `dashboard/app.py`.

## 1. Create The SQLite Database

First generate or refresh the forecast CSV outputs:

```powershell
python forecasting_pipeline.py
```

Then convert the forecast CSV files into SQLite:

```powershell
python grafana/export_to_sqlite.py
```

The script reads the CSV files from:

```text
data/forecast_outputs/
```

It creates the SQLite database at:

```text
data/grafana/export_opportunities.db
```

Re-run `python grafana/export_to_sqlite.py` every time the forecast CSV files are refreshed.

## 2. Start Grafana With Docker

From the project root, run:

```powershell
docker run -d --name grafana-export-dashboard -p 3000:3000 -e "GF_INSTALL_PLUGINS=frser-sqlite-datasource" -v "${PWD}/data/grafana:/var/lib/grafana/export_data" grafana/grafana
```

Open Grafana at:

```text
http://localhost:3000
```

The SQLite database path inside the Docker container is:

```text
/var/lib/grafana/export_data/export_opportunities.db
```

Use this path when creating the SQLite data source in Grafana.

## 3. Main Tables

The SQLite database contains these dashboard tables:

| Table | Purpose |
|---|---|
| `top_forecasted_opportunities` | Ranked country-product-year export opportunities |
| `priority_countries` | Best international markets for Algerian exporters |
| `priority_products` | Best products for future export focus |
| `final_forecasts` | Future forecasted demand by country, product, and year |
| `forecast_model_metrics` | Forecast model accuracy metrics |
| `historical_forecast_comparison` | Actual vs predicted values on historical test years |

Product code columns such as `k` are kept as text so HS codes are not damaged.

## 4. Recommended Grafana Panels

Use these panels for a simple final-demo dashboard:

| Panel | Suggested table/query |
|---|---|
| Ranked opportunities KPI | `SELECT COUNT(*) FROM top_forecasted_opportunities;` |
| Countries analyzed KPI | `SELECT COUNT(DISTINCT country_name) FROM top_forecasted_opportunities;` |
| Products analyzed KPI | `SELECT COUNT(DISTINCT k) FROM top_forecasted_opportunities;` |
| Top opportunities by score | Bar chart from `top_forecasted_opportunities` ordered by `forecast_opportunity_score` |
| Top forecasted opportunities table | Table from `top_forecasted_opportunities` |
| Priority international markets | Bar chart or table from `priority_countries` |
| Priority products | Bar chart or table from `priority_products` |
| Forecasted demand by year | Time series or bar chart from `final_forecasts` grouped by `year` |
| Forecast model performance | Table or bar chart from `forecast_model_metrics` |
| Best forecasting model | Stat panel using the selected model from `forecast_model_metrics` |

## 5. Useful SQL Queries

### Ranked Opportunities KPI

```sql
SELECT COUNT(*) AS ranked_opportunities
FROM top_forecasted_opportunities;
```

### Countries Analyzed KPI

```sql
SELECT COUNT(DISTINCT country_name) AS countries_analyzed
FROM top_forecasted_opportunities;
```

### Products Analyzed KPI

```sql
SELECT COUNT(DISTINCT k) AS products_analyzed
FROM top_forecasted_opportunities;
```

### Top Opportunities By Score

```sql
SELECT
    country_name || ' - ' || k AS opportunity,
    ROUND(forecast_opportunity_score, 3) AS score
FROM top_forecasted_opportunities
ORDER BY forecast_opportunity_score DESC
LIMIT 15;
```

### Top Forecasted Opportunities Table

```sql
SELECT
    rank,
    country_name,
    k AS hs_code,
    description_short,
    year,
    ROUND(forecasted_value, 2) AS forecasted_value,
    ROUND(predicted_growth_pct, 2) AS predicted_growth_pct,
    ROUND(forecast_opportunity_score, 3) AS opportunity_score
FROM top_forecasted_opportunities
ORDER BY rank ASC
LIMIT 50;
```

### Priority International Markets

```sql
SELECT
    country_name,
    ROUND(total_forecasted_value, 2) AS total_forecasted_value,
    ROUND(average_opportunity_score, 3) AS average_opportunity_score,
    high_opportunity_products
FROM priority_countries
ORDER BY average_opportunity_score DESC
LIMIT 20;
```

### Priority Products

```sql
SELECT
    k AS hs_code,
    description_short,
    ROUND(total_forecasted_value, 2) AS total_forecasted_value,
    ROUND(average_opportunity_score, 3) AS average_opportunity_score,
    high_opportunity_countries
FROM priority_products
ORDER BY average_opportunity_score DESC
LIMIT 20;
```

### Forecasted Demand By Year

```sql
SELECT
    year,
    SUM(forecast_partner_import_v) AS forecasted_demand
FROM final_forecasts
GROUP BY year
ORDER BY year;
```

### Forecast Model Performance

```sql
SELECT
    model,
    ROUND(MAE, 2) AS MAE,
    ROUND(RMSE, 2) AS RMSE,
    ROUND(WMAPE, 4) AS WMAPE,
    selected_for_future_forecast
FROM forecast_model_metrics
ORDER BY WMAPE ASC;
```

### Best Forecasting Model

```sql
SELECT model AS best_model
FROM forecast_model_metrics
WHERE selected_for_future_forecast = 1
LIMIT 1;
```

### Historical Forecast Comparison

```sql
SELECT
    t AS year,
    country_name,
    k AS hs_code,
    actual,
    prediction,
    model
FROM historical_forecast_comparison
ORDER BY year ASC
LIMIT 200;
```

## 6. Backup Dashboard

If Docker, Grafana, or the SQLite plugin setup fails, the same forecast outputs can still be opened in Streamlit:

```powershell
streamlit run dashboard/app.py
```

Streamlit reads the same CSV outputs and remains useful for live Python exploration. Grafana remains the professional dashboard for the final demo.
