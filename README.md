# Machine Learning Project

## Forecasting And Dashboard

This part of the project forecasts future partner-market demand for Algerian export opportunities and presents the results in dashboards.

Grafana is the professional dashboard used for the final demo because it satisfies the project PDF requirement for a professional visualization dashboard. Streamlit is kept as a backup/live Python dashboard that can open the same forecast outputs if the Grafana setup fails.

## Main Files

- `forecasting_pipeline.py`
- `notebooks/07_forecasting.ipynb`
- `grafana/export_to_sqlite.py`
- `grafana/README.md`
- `dashboard/app.py`
- `dashboard/README.md`
- `docs/forecasting_dashboard_explanation.md`
- `docs/forecasting_dashboard_presentation_script.md`

## Generate Forecast Outputs

From the project root:

```powershell
python forecasting_pipeline.py
```

Or run:

```text
notebooks/07_forecasting.ipynb
```

Forecast outputs are saved in:

- `data/forecast_outputs/final_forecasts.csv`
- `data/forecast_outputs/forecast_model_metrics.csv`
- `data/forecast_outputs/historical_forecast_comparison.csv`
- `data/forecast_outputs/top_forecasted_opportunities.csv`
- `data/forecast_outputs/priority_countries.csv`
- `data/forecast_outputs/priority_products.csv`
- `data/forecast_outputs/plots/`

## Convert Forecast CSV Files To SQLite

Grafana reads a SQLite database created from the forecast CSV outputs.

Run:

```powershell
python grafana/export_to_sqlite.py
```

SQLite database location:

```text
data/grafana/export_opportunities.db
```

Main Grafana tables:

- `top_forecasted_opportunities`
- `priority_countries`
- `priority_products`
- `final_forecasts`
- `forecast_model_metrics`
- `historical_forecast_comparison`

## Run Grafana With Docker

From the project root:

```powershell
docker run -d --name grafana-export-dashboard -p 3000:3000 -e "GF_INSTALL_PLUGINS=frser-sqlite-datasource" -v "${PWD}/data/grafana:/var/lib/grafana/export_data" grafana/grafana
```

Open Grafana:

```text
http://localhost:3000
```

SQLite database path inside Docker:

```text
/var/lib/grafana/export_data/export_opportunities.db
```

Use this path when configuring the SQLite data source in Grafana.

Recommended Grafana panels:

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

Detailed Grafana setup instructions are in:

- `grafana/README.md`

## Streamlit Backup Dashboard

The Streamlit dashboard is not deleted. It remains available as a backup/live Python dashboard:

```powershell
streamlit run dashboard/app.py
```

If using the project virtual environment from the parent folder:

```powershell
..\.venv\Scripts\streamlit.exe run dashboard/app.py
```

Streamlit reads the same forecast CSV files and is useful if Docker, Grafana, or the SQLite plugin setup fails.

Dashboard-specific Streamlit instructions are in:

- `dashboard/README.md`

## Documentation

- `docs/forecasting_dashboard_explanation.md`
- `docs/forecasting_dashboard_presentation_script.md`
- `grafana/README.md`
- `dashboard/README.md`
