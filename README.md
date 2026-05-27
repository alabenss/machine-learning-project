# machine-learning-project

## Forecasting and Dashboard

This part of the project forecasts future partner-market demand for Algerian export opportunities and presents the results in an interactive Streamlit dashboard.

Main files:

- `forecasting_pipeline.py`
- `notebooks/07_forecasting.ipynb`
- `dashboard/app.py`
- `dashboard/README.md`

To generate or refresh the forecast outputs:

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

To run the dashboard:

```powershell
streamlit run dashboard/app.py
```

If using the project virtual environment from the parent folder:

```powershell
..\.venv\Scripts\streamlit.exe run dashboard/app.py
```

Detailed documentation for this part is in:

- `docs/forecasting_dashboard_explanation.md`
- `docs/forecasting_dashboard_presentation_script.md`

Dashboard-specific instructions are in `dashboard/README.md`.
