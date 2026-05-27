# Forecasting and Dashboard Presentation Script

## 1. Introduction

My part of the project focuses on forecasting and dashboard visualization.

The goal of my part is to take the cleaned project data, forecast future market demand, rank possible export opportunities, and make the results easy to explore in an interactive dashboard.

## 2. Problem

Algeria needs to identify future export opportunities beyond hydrocarbons. So instead of only looking at past exports, my part tries to predict where demand may grow in the future.

This helps us move from historical analysis to forward-looking decision support.

## 3. Data

I used the cleaned and engineered master dataset produced earlier in the pipeline.

It contains country-product-year trade data, partner import demand, global demand indicators, Algeria export values, market penetration, RCA, and other engineered features.

The main dataset is:

```text
data/master_df.parquet
```

## 4. Non-Hydrocarbon Focus

By default, I focused on non-hydrocarbon products because the goal of the project is export diversification.

If we only selected the largest trade values, petroleum products would dominate the results. So I excluded obvious hydrocarbon products by default, such as HS codes starting with 27 or product descriptions containing petroleum, oils, gas, coal, or bituminous.

## 5. Forecasting Target

I mainly forecasted partner import demand because it represents the size of the potential market for Algerian exporters.

In the data, this target is called:

```text
partner_import_v
```

I also kept Algeria's export value, `alg_export_v`, as context, because it shows whether Algeria is already present in that market.

## 6. Models

I compared simple baseline models, statistical forecasting models, and machine learning models with lag features.

The models included:

- `naive_last_value`
- `moving_average_3`
- `trend_adjusted_naive`
- Exponential Smoothing when enough data was available
- ARIMA when possible
- Random Forest
- Gradient Boosting
- HistGradientBoosting

The simple models are important because a machine learning model should only be considered useful if it performs better than a basic baseline.

## 7. Evaluation

I evaluated the models using MAE, RMSE, MAPE, sMAPE, and WMAPE.

These metrics measure how far the predictions are from the real values.

MAE and RMSE measure the size of the error. WMAPE and sMAPE are useful because normal MAPE can become unstable when trade values are zero or very small.

In the current generated results, Random Forest performed best based on the saved model metrics.

## 8. Opportunity Ranking

After forecasting, I ranked the best future opportunities using a score that combines forecasted demand, predicted growth, global demand, RCA, and Algeria's current market penetration.

The idea is that a strong opportunity should usually have:

- high forecasted market demand,
- positive predicted growth,
- strong global demand,
- some comparative advantage signal if available,
- and low current Algerian market penetration, meaning there may be room to grow.

The final ranked file is:

```text
data/forecast_outputs/top_forecasted_opportunities.csv
```

## 9. Dashboard

The dashboard allows stakeholders to explore opportunities by country and product, visualize global demand trends, monitor predicted export growth, identify priority markets, and compare historical values with forecasted values.

The main dashboard file is:

```text
dashboard/app.py
```

It reads the saved CSV outputs only. It does not rerun the forecasting notebook and it does not rebuild raw data.

## 10. Professional Dashboard Requirement

The project recommends Grafana or similar professional tools.

For the live demo, I used Streamlit because it integrates directly with Python and is easy to demonstrate.

The outputs are saved as CSV files, so they can also be connected later to Grafana, Superset, Metabase, Kibana, or Power BI.

This means the current dashboard is practical for the project demo, while the data outputs are still ready for professional BI tools.

## 11. Conclusion

This part transforms the machine learning results into practical insights that can support Algerian exporters, CACI experts, and policymakers in choosing promising international markets.
