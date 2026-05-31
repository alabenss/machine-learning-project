# Grafana Dashboard Setup

This folder contains the SQLite export workflow for the final professional dashboard.

Grafana is the dashboard used for the final demo because it satisfies the project PDF requirement for a professional visualization dashboard. The Streamlit dashboard is still kept as a backup/live Python dashboard in `dashboard/app.py`.

## 1. Create The SQLite Database

First generate or refresh the forecast CSV outputs:

```powershell
python forecasting_pipeline.py
```

Then convert the dashboard CSV files into SQLite:

```powershell
python grafana/export_to_sqlite.py
```

The script reads the CSV files from:

```text
data/forecast_outputs/
data/clustering_outputs/
data/classification_outputs/
```

It creates the SQLite database at:

```text
data/grafana/export_opportunities.db
```

Re-run `python grafana/export_to_sqlite.py` every time the forecast, clustering, or classification CSV files are refreshed.

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
| `cluster_evaluation_summary` | Silhouette, Davies-Bouldin, and ARI scores for clustering models |
| `country_clusters` | Country cluster assignment and market-segment features |
| `product_clusters` | Product cluster assignment and demand/opportunity features |
| `sector_clusters` | Sector cluster assignment and sector-level opportunity features |
| `cluster_priority_market_ranking` | Top clustered market segments and priority countries |
| `cross_cluster_opportunity` | Country-cluster by product-cluster opportunity matrix |
| `classification_model_comparison` | Accuracy, precision, recall, and F1 for classifiers |
| `classification_predictions_2023` | Full High / Medium / Low predicted opportunity labels |
| `classification_top_export_opportunities` | Top ranked classification opportunities |
| `classification_opportunities_by_country` | High-opportunity counts by country |
| `classification_opportunities_by_product` | High-opportunity counts by product |
| `classification_feature_importance` | Consensus feature-importance ranking |

The exporter also creates these helper views for Grafana panels:

| View | Purpose |
|---|---|
| `vw_cluster_best_models` | Cluster models ordered by Silhouette score |
| `vw_country_cluster_summary` | Country counts and average metrics by cluster |
| `vw_product_cluster_summary` | Product counts and average metrics by cluster |
| `vw_sector_cluster_summary` | Sector counts and average metrics by cluster |
| `vw_classification_label_counts` | Predicted High / Medium / Low label counts |
| `vw_classification_best_model` | Best classifier by macro F1 |

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
| Cluster model quality | Table or bar chart from `vw_cluster_best_models` |
| Country cluster summary | Bar/table panel from `vw_country_cluster_summary` |
| Product cluster summary | Bar/table panel from `vw_product_cluster_summary` |
| Sector cluster summary | Bar/table panel from `vw_sector_cluster_summary` |
| Cluster priority markets | Bar/table panel from `cluster_priority_market_ranking` |
| Classification best model | Stat panel from `vw_classification_best_model` |
| Classification model comparison | Bar/table panel from `classification_model_comparison` |
| Predicted opportunity classes | Bar chart from `vw_classification_label_counts` |
| Classified top opportunities | Table from `classification_top_export_opportunities` |
| Classification feature importance | Bar chart from `classification_feature_importance` |

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

### Cluster Model Quality

```sql
SELECT
    Task,
    Model,
    ROUND(Silhouette, 4) AS silhouette,
    ROUND(davies_bouldin, 4) AS davies_bouldin,
    ARI
FROM vw_cluster_best_models;
```

### Country Cluster Summary

```sql
SELECT
    cluster_kmeans,
    countries,
    ROUND(avg_opportunity_rate, 4) AS avg_opportunity_rate,
    ROUND(avg_import_scale_log, 3) AS avg_import_scale_log,
    ROUND(avg_import_growth_slope, 4) AS avg_import_growth_slope
FROM vw_country_cluster_summary;
```

### Cluster Priority Markets

```sql
SELECT
    Rank,
    Country,
    ISO3,
    Cluster,
    ROUND("Import scale (log)", 3) AS import_scale_log,
    ROUND("Opportunity rate", 3) AS opportunity_rate,
    ROUND("Growth slope", 3) AS growth_slope,
    ROUND("Priority score", 3) AS priority_score
FROM cluster_priority_market_ranking
ORDER BY "Priority score" DESC
LIMIT 20;
```

### Classification Best Model

```sql
SELECT
    model AS best_classifier,
    ROUND(accuracy, 4) AS accuracy,
    ROUND(macro_f1, 4) AS macro_f1
FROM vw_classification_best_model;
```

### Classification Model Comparison

```sql
SELECT
    model,
    ROUND(accuracy, 4) AS accuracy,
    ROUND(macro_precision, 4) AS macro_precision,
    ROUND(macro_recall, 4) AS macro_recall,
    ROUND(macro_f1, 4) AS macro_f1
FROM classification_model_comparison
ORDER BY macro_f1 DESC;
```

### Predicted Opportunity Classes

```sql
SELECT
    predicted_label,
    rows
FROM vw_classification_label_counts;
```

### Classified Top Opportunities

```sql
SELECT
    country_name,
    description_short,
    ROUND(partner_import_v, 2) AS partner_import_v,
    ROUND(global_demand_rank, 3) AS global_demand_rank,
    ROUND(world_demand_growth, 3) AS world_demand_growth,
    ROUND(market_penetration, 6) AS market_penetration,
    predicted_label
FROM classification_top_export_opportunities
LIMIT 30;
```

### Classification Feature Importance

```sql
SELECT
    feature,
    ROUND(avg_rank, 2) AS avg_rank
FROM classification_feature_importance
ORDER BY avg_rank ASC
LIMIT 15;
```

## 6. Backup Dashboard

If Docker, Grafana, or the SQLite plugin setup fails, the same forecast, clustering, and classification outputs can still be opened in Streamlit:

```powershell
streamlit run dashboard/app.py
```

Streamlit reads the same CSV outputs and remains useful for live Python exploration. Grafana remains the professional dashboard for the final demo.
