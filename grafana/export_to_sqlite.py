"""
Export forecast CSV outputs to a SQLite database for Grafana.

Run from project root:
    python grafana/export_to_sqlite.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/grafana/export_opportunities.db")

# (csv_path, table_name, text_columns)
# text_columns: column names that must be preserved as TEXT (e.g. product codes)
SOURCES = [
    (
        Path("data/forecast_outputs/top_forecasted_opportunities.csv"),
        "top_forecasted_opportunities",
        ["k", "product_code", "hs_code"],
    ),
    (
        Path("data/forecast_outputs/priority_countries.csv"),
        "priority_countries",
        ["k", "product_code", "hs_code"],
    ),
    (
        Path("data/forecast_outputs/priority_products.csv"),
        "priority_products",
        ["k", "product_code", "hs_code"],
    ),
    (
        Path("data/forecast_outputs/final_forecasts.csv"),
        "final_forecasts",
        ["k", "product_code", "hs_code"],
    ),
    (
        Path("data/forecast_outputs/forecast_model_metrics.csv"),
        "forecast_model_metrics",
        ["k", "product_code", "hs_code"],
    ),
    (
        Path("data/forecast_outputs/historical_forecast_comparison.csv"),
        "historical_forecast_comparison",
        ["k", "product_code", "hs_code"],
    ),
]


def load_csv(csv_path: Path, text_columns: list[str]) -> pd.DataFrame | None:
    """Read a CSV, forcing any present text_columns to str dtype."""
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found - skipping.")
        return None

    # Peek at column names to build dtype map only for columns that exist
    header = pd.read_csv(csv_path, nrows=0)
    dtype_map = {col: str for col in text_columns if col in header.columns}

    df = pd.read_csv(csv_path, dtype=dtype_map)
    return df


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    imported = 0

    try:
        for csv_path, table_name, text_columns in SOURCES:
            print(f"\n[{table_name}]")
            df = load_csv(csv_path, text_columns)
            if df is None:
                continue

            df.to_sql(table_name, conn, if_exists="replace", index=False)
            row_count = len(df)
            print(f"  Imported {row_count:,} rows -> table '{table_name}'")
            imported += 1

        conn.commit()
    finally:
        conn.close()

    if imported == 0:
        print("\nNo CSV files were found. Run your forecast pipeline first.")
        sys.exit(1)

    print(f"\nDatabase written to: {DB_PATH}")
    print(f"Tables imported: {imported}/{len(SOURCES)}")


if __name__ == "__main__":
    main()
