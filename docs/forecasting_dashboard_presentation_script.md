# Forecasting Dashboard Presentation Script

## 1. Introduction

My part of the project focuses on forecasting and dashboard visualization.

The goal is to forecast future market demand, rank export opportunities for Algerian exporters, and present the results in a professional dashboard.

## 2. Problem

Algeria needs to identify export opportunities beyond hydrocarbons.

Instead of only describing past exports, this part predicts where future demand may grow and which country-product pairs may be attractive for Algerian exporters.

## 3. Data

The forecasting pipeline uses the cleaned and engineered master dataset:

```text
data/master_df.parquet
```

This dataset contains country-product-year trade data, partner import demand, Algeria export values, global demand indicators, market penetration, RCA, and other useful features.

## 4. Forecasting Target

The main target is:

```text
partner_import_v
```

I forecast this value because it represents demand in the destination market. If a country is expected to import more of a product in the future, it may be a better opportunity for Algerian exporters.

## 5. Non-Hydrocarbon Focus

The default analysis focuses on non-hydrocarbon products.

Hydrocarbon products are excluded using HS codes starting with `27` and product descriptions related to petroleum, oils, gas, coal, or bituminous products.

This supports the project goal of export diversification.

## 6. Models

I compared baseline, statistical, and machine learning models:

- `naive_last_value`
- `moving_average_3`
- `trend_adjusted_naive`
- exponential smoothing
- ARIMA when possible
- random forest
- gradient boosting
- histogram gradient boosting

The models are evaluated with MAE, RMSE, MAPE, sMAPE, and WMAPE.

## 7. Opportunity Ranking

After forecasting, I rank country-product opportunities using a score based on:

- forecasted demand
- predicted growth
- global demand rank
- RCA
- low current Algerian market penetration

This ranking helps identify opportunities that are large, growing, globally relevant, and still not fully captured by Algeria.

The main ranked output is:

```text
data/forecast_outputs/top_forecasted_opportunities.csv
```

## 8. Grafana Workflow

The forecast pipeline first produces CSV outputs in:

```text
data/forecast_outputs/
```

Then I convert those CSV files into a SQLite database for Grafana:

```powershell
python grafana/export_to_sqlite.py
```

The SQLite database is saved at:

```text
data/grafana/export_opportunities.db
```

The main SQLite tables are:

- `top_forecasted_opportunities`
- `priority_countries`
- `priority_products`
- `final_forecasts`
- `forecast_model_metrics`
- `historical_forecast_comparison`

## 9. Running Grafana

I start Grafana with Docker using:

```powershell
docker run -d --name grafana-export-dashboard -p 3000:3000 -e "GF_INSTALL_PLUGINS=frser-sqlite-datasource" -v "${PWD}/data/grafana:/var/lib/grafana/export_data" grafana/grafana
```

Grafana opens at:

```text
http://localhost:3000
```

Inside Docker, the SQLite database path is:

```text
/var/lib/grafana/export_data/export_opportunities.db
```

This is the path used when configuring the SQLite data source in Grafana.

## 10. Dashboard Panels

The final Grafana dashboard includes or can include:

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

These panels make the model results easy to present to CACI experts, exporters, and policymakers.

## 11. Professional Dashboard Requirement

The project PDF asks for a professional dashboard tool.

This requirement is satisfied because the final dashboard is built in Grafana, using a SQLite database generated from the machine learning forecast outputs.

## 12. Streamlit Backup

The Streamlit dashboard is still kept as a backup/live Python dashboard:

```powershell
streamlit run dashboard/app.py
```

It uses the same forecast CSV outputs. If the Grafana or Docker setup fails during the demo, the same results can still be opened in Streamlit.

## 13. Conclusion

This part transforms machine learning forecasts into practical business insights.

Grafana is used for the final professional dashboard, while Streamlit remains available as a reliable Python backup.
