# Forecasting And Dashboard Explanation

## 1. Overview

This part of the project forecasts future global export opportunities for Algerian exporters and presents the results in a professional dashboard.

The workflow is:

1. Build or load the prepared machine learning dataset.
2. Forecast future partner-country import demand.
3. Rank the best country-product export opportunities.
4. Save the forecast results as CSV files.
5. Convert the CSV files to SQLite for Grafana.
6. Build the final professional dashboard in Grafana.

Grafana is used for the final demo because the project PDF asks for a professional dashboard tool such as Grafana, Superset, Metabase, or Kibana. Streamlit is still kept as a backup/live Python dashboard.

## 2. Forecasting Data

The forecasting pipeline uses:

```text
data/master_df.parquet
```

This file contains country-product-year trade data and engineered indicators.

Important columns include:

- `t`: year
- `j`: partner country code
- `k`: HS6 product code
- `iso3`: partner country ISO3 code
- `country_name`: partner country name
- `description_short`: product description
- `partner_import_v`: partner country import demand
- `alg_export_v`: Algeria export value
- `world_import_v`: global product demand
- `world_demand_growth`: global demand growth
- `global_demand_rank`: global demand strength
- `market_penetration`: Algeria's current market penetration
- `rca`: revealed comparative advantage

The main forecasting target is `partner_import_v` because it represents demand in the destination market.

## 3. Non-Hydrocarbon Focus

The default forecast focuses on non-hydrocarbon diversification.

Hydrocarbon products are excluded by default using:

- HS product codes starting with `27`
- product descriptions containing terms such as petroleum, oils, gas, coal, or bituminous

This keeps the project focused on export diversification rather than only explaining Algeria's existing hydrocarbon exports.

## 4. Forecasting Models

The pipeline compares several models:

- `naive_last_value`
- `moving_average_3`
- `trend_adjusted_naive`
- `simple_exponential_smoothing`
- `arima_110_log`
- `random_forest`
- `gradient_boosting`
- `hist_gradient_boosting`

The models are evaluated with MAE, RMSE, MAPE, sMAPE, and WMAPE. Lower error values mean better forecasting performance.

In the current generated outputs, the selected model is saved in:

```text
data/forecast_outputs/forecast_model_metrics.csv
```

## 5. Opportunity Ranking

The final opportunity ranking is saved in:

```text
data/forecast_outputs/top_forecasted_opportunities.csv
```

The ranking combines:

- forecasted demand
- predicted growth
- global demand rank
- revealed comparative advantage
- low current Algerian market penetration

This helps identify opportunities that are large, growing, globally relevant, and still under-penetrated by Algerian exporters.

## 6. Forecast Output Files

The forecasting pipeline generates:

- `data/forecast_outputs/final_forecasts.csv`
- `data/forecast_outputs/forecast_model_metrics.csv`
- `data/forecast_outputs/historical_forecast_comparison.csv`
- `data/forecast_outputs/top_forecasted_opportunities.csv`
- `data/forecast_outputs/priority_countries.csv`
- `data/forecast_outputs/priority_products.csv`
- `data/forecast_outputs/plots/`

These CSV files are the shared outputs used by both Grafana and Streamlit.

## 7. Convert Forecast CSV Files To SQLite

Grafana reads the forecast results from a SQLite database.

Run this command from the project root:

```powershell
python grafana/export_to_sqlite.py
```

The SQLite database is created at:

```text
data/grafana/export_opportunities.db
```

The conversion script imports the main forecast CSV files into SQLite tables.

## 8. Main Grafana Tables

The database contains:

- `top_forecasted_opportunities`
- `priority_countries`
- `priority_products`
- `final_forecasts`
- `forecast_model_metrics`
- `historical_forecast_comparison`

These tables are enough to build the professional dashboard required by the project.

## 9. Run Grafana With Docker

Use this Docker command from the project root:

```powershell
docker run -d --name grafana-export-dashboard -p 3000:3000 -e "GF_INSTALL_PLUGINS=frser-sqlite-datasource" -v "${PWD}/data/grafana:/var/lib/grafana/export_data" grafana/grafana
```

Open Grafana at:

```text
http://localhost:3000
```

Inside Docker, the SQLite database path is:

```text
/var/lib/grafana/export_data/export_opportunities.db
```

This is the path to use when creating the SQLite data source in Grafana.

## 10. Recommended Grafana Panels

The final dashboard should include:

- Ranked opportunities KPI
- Countries analyzed KPI
- Products analyzed KPI
- Top opportunities by score
- Top forecasted opportunities table
- Priority international markets
- Priority products
- Forecasted demand by year
- Forecast model performance
- Best forecasting model

These panels cover the main business questions: how many opportunities were ranked, which countries matter most, which products matter most, how demand changes over time, and which model produced the forecast.

## 11. Streamlit Backup Dashboard

The Streamlit dashboard is kept in:

```text
dashboard/app.py
```

Run it with:

```powershell
streamlit run dashboard/app.py
```

Streamlit reads the same CSV outputs. It is useful as a backup/live Python dashboard if Docker, Grafana, or the SQLite plugin setup fails.

## 12. Final Demo Message

For the final presentation, the correct message is:

Grafana is the professional dashboard used for the final demo and satisfies the project PDF dashboard requirement. Streamlit is kept as a backup dashboard because it can open the same generated forecast outputs directly in Python.

## 13. Assumptions And Limitations

- Forecasts are decision-support estimates, not guaranteed future trade values.
- The model depends on the quality and length of historical trade data.
- Some optional EDA context depends on whether supporting files exist in `data/eda_outputs/`.
- The dashboard should be refreshed by rerunning the forecasting pipeline and then rerunning `python grafana/export_to_sqlite.py`.
