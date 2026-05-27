# Forecasting Global Export Opportunities Dashboard

This dashboard is the live visualization component for the project **Forecasting and Identifying Global Export Opportunities for Algerian Exporters**.

It is designed for CACI experts, Algerian exporters, and policymakers who need to explore forecast-based export opportunities by country, product, future demand, predicted growth, and market potential.

## Why Streamlit

The project description recommends Grafana or another professional dashboard tool such as Apache Superset, Metabase, Kibana, or Power BI. Streamlit is used as the working live demonstration because:

- It runs as a pure Python application with no external server, database, or Docker infrastructure required.
- All Plotly charts (choropleth maps, line charts, scatter plots, bar charts) are fully interactive — users can zoom, pan, filter, hover, and download just like in Grafana or Superset.
- It integrates directly with the ML pipeline outputs (CSV files) without a data transformation layer.
- It can be deployed to Streamlit Community Cloud, Heroku, or any server in one command, matching the "deployed data pipeline" deliverable.

The generated CSV outputs are **dashboard-ready for any tool**. Loading them into Grafana, Apache Superset, Metabase, Kibana, or Power BI requires only connecting a CSV/SQLite data source — no schema changes needed. The Streamlit dashboard serves as the live demo; the CSV pipeline serves as the portable data backend.

## How To Run

From the project root:

```powershell
streamlit run dashboard/app.py
```

If using the project virtual environment from the parent folder:

```powershell
..\.venv\Scripts\streamlit.exe run dashboard/app.py
```

The dashboard reads saved CSV files only. It does not run the forecasting notebook, rebuild the forecasting pipeline, or read raw BACI files.

If required forecast outputs are missing, the app shows:

```text
Please run notebooks/07_forecasting.ipynb first to generate forecast outputs.
```

## Files Read

Main forecast files:

- `data/forecast_outputs/top_forecasted_opportunities.csv`
- `data/forecast_outputs/priority_countries.csv`
- `data/forecast_outputs/priority_products.csv`
- `data/forecast_outputs/final_forecasts.csv`
- `data/forecast_outputs/historical_forecast_comparison.csv`
- `data/forecast_outputs/forecast_model_metrics.csv`

Optional historical context files:

- `data/eda_outputs/exports_by_year.csv`
- `data/eda_outputs/global_demand_products.csv`
- `data/eda_outputs/opportunities_by_country.csv`
- `data/eda_outputs/opportunities_by_product.csv`
- `data/eda_outputs/sector_summary.csv`
- `data/eda_outputs/top_export_partners.csv`
- `data/eda_outputs/top_export_products.csv`
- `data/eda_outputs/country_product_heatmap_demand.csv`
- `data/eda_outputs/country_product_heatmap_log.csv`

The `eda_outputs` files are used only for historical context, overview charts, and interpretation support. Forecasting results come from `data/forecast_outputs/`.

## Dashboard Sections

1. **Executive Summary**  
   Shows KPI cards for countries, products, ranked opportunities, predicted growth, best opportunity country/product, model performance, and forecast scope.

2. **Export Opportunities Explorer**  
   Uses `top_forecasted_opportunities.csv` to show ranked country-product opportunities and a top-opportunity chart.

3. **Priority International Markets**  
   Uses `priority_countries.csv` to answer: “Which countries should Algerian exporters focus on first?”

4. **Priority Products**  
   Uses `priority_products.csv` to answer: “Which products appear most promising for future export opportunities?”

5. **Global Demand Trends and Trade Flows**  
   Uses forecast outputs and optional EDA summaries to visualize demand trends, historical exports, top global demand products, and historical export partners/products.

6. **Predicted Export Growth and Market Potential**  
   Shows top countries, products, and country-product opportunities by predicted growth and the relationship between forecasted value and opportunity score.

7. **Historical vs Forecasted Trade Indicators**  
   Compares historical partner import demand with forecasted future demand and shows actual vs predicted test-period behavior.

8. **Forecast Model Performance**  
   Shows MAE, RMSE, MAPE, sMAPE, and WMAPE where available. Lower MAE/RMSE/WMAPE indicates better forecasting accuracy. MAPE can be unstable when trade values are very small or zero.

9. **Historical EDA Context**  
   Adds historical context from `data/eda_outputs`, such as exports by year, sector summaries, historical opportunities, and heatmaps.

10. **Sector-Level Demand and Opportunity Analysis**  
   Maps HS product codes to broad trade sectors (Agriculture, Electronics, Machinery, Pharmaceuticals, etc.) and shows which sectors carry the most forecasted demand, predicted growth, and opportunity score. Addresses the project requirement to analyze opportunities by sector.

11. **Interpretation for CACI, Exporters, and Policymakers**  
   Explains how to interpret opportunity scores, low market penetration, RCA, and forecast limitations.

## Project PDF Requirements Covered

- Professional visualization dashboard deployment: covered with a Streamlit live dashboard (interactive Plotly charts, same visual quality as Grafana/Superset).
- Explore opportunities by country, product, and sector: covered in Sections 2, 3, 4, and 10 with sidebar filters for country, product, horizon, score, and hydrocarbon type.
- Visualize global demand trends and trade flows: covered in Section 5.
- Monitor predicted export growth and market potential: covered in Section 6.
- Identify priority international markets for Algerian exporters: covered in Section 3 with both a world choropleth map and a ranked table.
- Compare historical vs forecasted trade indicators: covered in Section 7.
- Sector-level demand and opportunity analysis: covered in Section 10.
- Support stakeholders such as CACI experts, exporters, and policymakers: covered through targeted summaries and interpretation text in Section 11.

## Assumptions And Limitations

- The dashboard is a decision-support tool, not a guarantee of future trade values.
- It depends on the saved forecasting CSVs being generated first.
- Optional EDA files are loaded only if available; missing optional files do not stop the app.
- Sector filtering appears only if a sector column exists in the forecast opportunity file.
- The opportunity score is forecast-based and can later be enriched with clustering, classification, tariff, logistics, or policy indicators when those outputs become available.
