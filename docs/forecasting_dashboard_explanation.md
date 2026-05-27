# Forecasting and Dashboard Explanation

## 1. Overview of My Part

My part of the project focuses on forecasting and dashboard visualization for **Forecasting and Identifying Global Export Opportunities for Algerian Exporters**.

This part takes the cleaned and engineered master dataset, builds forecasting models for future trade demand, creates a forecast-based opportunity ranking, and presents the results in an interactive dashboard for CACI experts, Algerian exporters, and policymakers.

The main files for this part are:

- `forecasting_pipeline.py`
- `notebooks/07_forecasting.ipynb`
- `dashboard/app.py`
- `dashboard/README.md`

## 2. Dataset Used

Forecasting uses:

- `data/master_df.parquet`

This master dataset was produced earlier in the project by:

- `build_master_df.py`
- `notebooks/01_build_master_dataset.ipynb`
- `notebooks/02_data_cleaning_eda.ipynb`

The forecasting step does not rebuild raw BACI data. It starts from the already prepared master dataset.

## 3. Why `data/master_df.parquet` Is Used

`data/master_df.parquet` is used because it contains country-product-year level trade data and engineered features needed for forecasting and opportunity analysis.

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
- `global_demand_rank`: relative global demand strength
- `market_penetration`: Algeria's current penetration in the destination market
- `rca`: revealed comparative advantage
- `label_opportunity`: opportunity label, if available
- `split`: temporal train/validation/test split, if available

This makes the dataset suitable for forecasting future market demand and connecting those forecasts to export opportunity indicators.

## 4. Role of `eda_outputs`

`data/eda_outputs/` is not the main forecasting training data.

It is used only as supporting dashboard context for:

- historical export trends,
- global demand summaries,
- top export partners,
- top export products,
- country/product opportunity summaries,
- sector summaries if available,
- country-product heatmaps if available.

The main forecast and ranking results in the dashboard come from `data/forecast_outputs/`.

## 5. Forecasting Target

The main forecasting target is:

- `partner_import_v`

This target is used because it represents demand in the destination market. For Algerian exporters, a country-product pair is interesting when the partner country is expected to import a high or growing amount of that product.

`alg_export_v` is also kept, but as historical context. It shows Algeria's current export presence in a market, while `partner_import_v` shows the potential market size.

## 6. Non-Hydrocarbon Focus

The default forecasting scope focuses on non-hydrocarbon diversification.

Hydrocarbon products are excluded by default using:

- HS product codes starting with `27`,
- or descriptions containing terms such as petroleum, oils, gas, coal, or bituminous.

This is important because Algeria already depends heavily on hydrocarbons. The goal of the project is not only to explain existing hydrocarbon exports, but to identify future diversification opportunities for Algerian exporters.

The code still allows hydrocarbons to be included by changing:

- `INCLUDE_HYDROCARBONS = True`

## 7. Forecasting Level and Sample

Forecasts are created at country-product level:

- country name,
- HS6 product code,
- product description,
- year.

The default configuration is:

- `TOP_PRODUCTS = 50`
- `TOP_COUNTRIES_PER_PRODUCT = 10`
- `FORECAST_HORIZON = 3`
- `INCLUDE_HYDROCARBONS = False`

The sample is selected in a balanced way. First, the pipeline selects top products by demand. Then, for each selected product, it selects top countries. This avoids the results being dominated by one product such as petroleum oils.

## 8. Forecasting Models

The pipeline compares several forecasting approaches:

- `naive_last_value`: uses the last observed value as the next forecast.
- `moving_average_3`: uses the average of the last three years.
- `trend_adjusted_naive`: adjusts the last value using a clipped recent growth rate.
- `simple_exponential_smoothing`: smooths the time series when enough years are available.
- `arima_110_log`: ARIMA model on log-transformed values, only if `statsmodels` is installed and the series is long enough.
- `random_forest`: tree-based machine learning regression model.
- `gradient_boosting`: gradient boosting regression model.
- `hist_gradient_boosting`: histogram-based gradient boosting regression model.

Statistical models are skipped safely if the package is unavailable or if a series does not have enough historical years.

## 9. Machine Learning Features

The machine learning models use lag and contextual features, including:

- `target_lag_1`
- `target_lag_2`
- `target_lag_3`
- `rolling_mean_3`
- `growth_rate_lag_1`
- `year`
- `series_code`
- `country_code`
- `product_code`
- `global_demand_rank`, if available
- `world_demand_growth`, if available
- `market_penetration`, if available
- `rca`, if available

Lag features use only past information, which is necessary for a valid forecasting setup.

## 10. Time-Based Split

Random train/test splitting is not used because it can leak future information into training.

The pipeline uses the existing `split` column when available. In this project, the split separates earlier years for training/validation and later years for testing. If the `split` column is not available, the pipeline falls back to a time-based split.

This keeps the evaluation closer to the real forecasting problem: predicting future years from past years.

## 11. Evaluation Metrics

The models are evaluated using:

- **MAE**: mean absolute error. It measures the average size of the forecast error.
- **RMSE**: root mean squared error. It penalizes large errors more strongly.
- **MAPE**: mean absolute percentage error. It expresses error as a percentage.
- **sMAPE**: symmetric MAPE. It is more stable than regular MAPE in some cases.
- **WMAPE**: weighted MAPE. It measures total absolute error relative to total actual demand.

MAPE can become unstable when actual trade values are zero or very small. Because trade data often contains zeros and small values, sMAPE and WMAPE are also included.

## 12. Best Model Selection

The generated `data/forecast_outputs/forecast_model_metrics.csv` shows that the current best model is:

- `random_forest`

In the latest generated outputs, `random_forest` has the lowest RMSE and WMAPE among the available models.

The dashboard uses WMAPE when available to identify the best model for display, otherwise it falls back to RMSE.

## 13. Future Forecasts

Future forecasts are generated for the configured forecast horizon:

- `FORECAST_HORIZON = 3`

The output forecasts future partner import demand for future years and saves them to:

- `data/forecast_outputs/final_forecasts.csv`

If the best model is a flat baseline, the pipeline can use `trend_adjusted_naive` for future growth analysis. This avoids producing a useless flat forecast while still staying honest, because the adjustment is based on observed recent growth and clipped to avoid extreme values.

In the current run, the future forecast model is `random_forest`.

## 14. Opportunity Ranking

The project creates the final forecast-based opportunity ranking:

- `data/forecast_outputs/top_forecasted_opportunities.csv`

The ranking uses this score:

```text
forecast_opportunity_score =
0.40 * normalized_forecasted_demand
+ 0.25 * normalized_predicted_growth_rate
+ 0.20 * normalized_global_demand_rank
+ 0.10 * normalized_rca
+ 0.05 * normalized_low_market_penetration_score
```

Each component means:

- **Forecasted demand**: expected future destination-market demand.
- **Predicted growth**: expected demand increase.
- **Global demand rank**: how globally demanded the product is.
- **RCA**: Algeria's comparative advantage signal, if available.
- **Low market penetration**: higher score when Algeria is currently underrepresented in that market.

If a component is missing, the code handles it safely. Missing optional values are treated neutrally, and if a whole component is unavailable, the available weights are normalized.

## 15. Priority Countries and Products

The pipeline also creates:

- `data/forecast_outputs/priority_countries.csv`
- `data/forecast_outputs/priority_products.csv`

`priority_countries.csv` summarizes ranked opportunities by country, including average opportunity score, total forecasted value, total predicted growth, and the number of high-opportunity products.

`priority_products.csv` summarizes ranked opportunities by product, including average opportunity score, total forecasted value, total predicted growth, and the number of high-opportunity countries.

## 16. Output Files Generated

The forecasting and ranking pipeline generates:

- `data/forecast_outputs/final_forecasts.csv`: future demand forecasts by country-product-year.
- `data/forecast_outputs/forecast_model_metrics.csv`: model evaluation metrics.
- `data/forecast_outputs/historical_forecast_comparison.csv`: actual vs predicted values on historical test years.
- `data/forecast_outputs/top_forecasted_opportunities.csv`: final forecast-based opportunity ranking.
- `data/forecast_outputs/priority_countries.csv`: priority international markets.
- `data/forecast_outputs/priority_products.csv`: priority products.
- `data/forecast_outputs/plots/`: saved forecast visualizations.

## 17. Professional Dashboard Requirement

The project PDF recommends a professional visualization dashboard such as:

- Grafana,
- Apache Superset,
- Metabase,
- Kibana,
- or similar tools.

This implementation provides:

1. A working Streamlit interactive dashboard for the live project demonstration.
2. Dashboard-ready CSV outputs that can also be connected to Grafana, Apache Superset, Metabase, Kibana, or Power BI.
3. The required stakeholder functions:
   - explore export opportunities by country, sector, and product,
   - visualize global demand trends and trade flows,
   - monitor predicted export growth and market potential,
   - identify priority international markets,
   - compare historical vs forecasted trade indicators.

Streamlit is used because it works directly with the Python project and the generated CSV files, without needing extra database or server infrastructure.

## 18. Dashboard Explanation

The dashboard is implemented in:

- `dashboard/app.py`

Dashboard-specific instructions are in:

- `dashboard/README.md`

The dashboard sections are:

### Executive Summary

Shows KPI cards such as number of countries, products, ranked opportunities, average predicted growth, best opportunity country/product, best forecasting model, and forecast scope.

### Export Opportunities Explorer

Uses `top_forecasted_opportunities.csv` to display ranked country-product opportunities. Stakeholders can filter by country, product, year, forecast horizon, model, opportunity score, forecast scope, and hydrocarbon/non-hydrocarbon status.

### Priority International Markets

Uses `priority_countries.csv` to answer: which countries should Algerian exporters focus on first?

### Priority Products

Uses `priority_products.csv` to answer: which products appear most promising for future export opportunities?

### Global Demand Trends and Trade Flows

Uses forecast outputs and optional EDA files to show forecasted partner import demand, historical Algeria exports, top global demand products, and historical export partners/products.

### Predicted Export Growth and Market Potential

Shows top countries, products, and country-product pairs by predicted growth. It also shows the relationship between forecasted demand and opportunity score.

### Historical vs Forecasted Trade Indicators

Compares historical demand context with future forecasted demand, and shows actual vs predicted values on test years using the best model selected by WMAPE or RMSE.

### Forecast Model Performance

Shows model metrics such as MAE, RMSE, MAPE, sMAPE, and WMAPE, and explains that lower values indicate better forecast accuracy.

### Historical EDA Context

Uses `data/eda_outputs/` as supporting historical context only. This can include exports by year, sector summaries, opportunity summaries, and heatmaps if available.

### Interpretation for CACI, Exporters, and Policymakers

Explains how to interpret opportunity scores, low market penetration, RCA, and forecast limitations in simple business language.

## 19. How to Run Forecasting

From the project root, run:

```powershell
python forecasting_pipeline.py
```

Or run the notebook:

```text
notebooks/07_forecasting.ipynb
```

This regenerates the forecast outputs under:

- `data/forecast_outputs/`

## 20. How to Run Dashboard

From the project root, run:

```powershell
streamlit run dashboard/app.py
```

If using the project virtual environment from the parent folder:

```powershell
..\.venv\Scripts\streamlit.exe run dashboard/app.py
```

## 21. Assumptions and Limitations
- The opportunity ranking uses forecasting, demand, RCA, and market penetration instead.
- Forecasting is limited by the number of available historical years.
- Some sector-level analysis depends on whether sector columns exist in the available data.
- Forecasts are decision-support estimates, not guaranteed future values.
- Streamlit is used for the live demo, while Grafana, Superset, Metabase, Kibana, or Power BI can consume the CSV outputs later.
