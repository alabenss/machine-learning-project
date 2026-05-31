"""
Export dashboard CSV outputs to a SQLite database for Grafana.

Run from project root:
    python grafana/export_to_sqlite.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/grafana/export_opportunities.db")

# (csv_path, table_name, text_columns, required)
# text_columns: column names that must be preserved as TEXT (e.g. product codes)
# required: missing required files fail the export; missing optional files are skipped
SOURCES = [
    (
        Path("data/forecast_outputs/top_forecasted_opportunities.csv"),
        "top_forecasted_opportunities",
        ["k", "product_code", "hs_code"],
        True,
    ),
    (
        Path("data/forecast_outputs/priority_countries.csv"),
        "priority_countries",
        ["k", "product_code", "hs_code"],
        True,
    ),
    (
        Path("data/forecast_outputs/priority_products.csv"),
        "priority_products",
        ["k", "product_code", "hs_code"],
        True,
    ),
    (
        Path("data/forecast_outputs/final_forecasts.csv"),
        "final_forecasts",
        ["k", "product_code", "hs_code"],
        True,
    ),
    (
        Path("data/forecast_outputs/forecast_model_metrics.csv"),
        "forecast_model_metrics",
        ["k", "product_code", "hs_code"],
        True,
    ),
    (
        Path("data/forecast_outputs/historical_forecast_comparison.csv"),
        "historical_forecast_comparison",
        ["k", "product_code", "hs_code"],
        True,
    ),
    (
        Path("data/clustering_outputs/cluster_evaluation_summary.csv"),
        "cluster_evaluation_summary",
        ["Task", "Model"],
        False,
    ),
    (
        Path("data/clustering_outputs/country_clusters.csv"),
        "country_clusters",
        ["j", "iso3"],
        False,
    ),
    (
        Path("data/clustering_outputs/product_clusters.csv"),
        "product_clusters",
        ["k"],
        False,
    ),
    (
        Path("data/clustering_outputs/sector_clusters.csv"),
        "sector_clusters",
        ["hs2"],
        False,
    ),
    (
        Path("data/clustering_outputs/priority_market_ranking.csv"),
        "cluster_priority_market_ranking",
        ["ISO3", "Country"],
        False,
    ),
    (
        Path("data/clustering_outputs/cross_cluster_opportunity.csv"),
        "cross_cluster_opportunity",
        ["Country Cluster"],
        False,
    ),
    (
        Path("data/classification_outputs/model_comparison.csv"),
        "classification_model_comparison",
        ["model"],
        False,
    ),
    (
        Path("data/classification_outputs/predictions_2023.csv"),
        "classification_predictions_2023",
        ["j", "k", "iso3"],
        False,
    ),
    (
        Path("data/classification_outputs/top_export_opportunities.csv"),
        "classification_top_export_opportunities",
        ["k"],
        False,
    ),
    (
        Path("data/classification_outputs/opp_by_country_predicted.csv"),
        "classification_opportunities_by_country",
        [],
        False,
    ),
    (
        Path("data/classification_outputs/opp_by_product_predicted.csv"),
        "classification_opportunities_by_product",
        ["k"],
        False,
    ),
    (
        Path("data/classification_outputs/feature_importance_consensus.csv"),
        "classification_feature_importance",
        [],
        False,
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
    return df.drop(columns=[col for col in df.columns if col.startswith("Unnamed")], errors="ignore")


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    result = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (table_name,),
    ).fetchone()
    return result is not None


def create_views(conn: sqlite3.Connection) -> None:
    """Create Grafana-friendly views when their source tables exist."""
    view_sql: list[tuple[str, str, list[str]]] = [
        (
            "vw_cluster_best_models",
            """
            CREATE VIEW vw_cluster_best_models AS
            SELECT
                Task,
                Model,
                Silhouette,
                "Davies-Bouldin" AS davies_bouldin,
                ARI
            FROM cluster_evaluation_summary
            WHERE Silhouette IS NOT NULL
            ORDER BY Silhouette DESC
            """,
            ["cluster_evaluation_summary"],
        ),
        (
            "vw_country_cluster_summary",
            """
            CREATE VIEW vw_country_cluster_summary AS
            SELECT
                cluster_kmeans,
                COUNT(*) AS countries,
                AVG(opportunity_rate) AS avg_opportunity_rate,
                AVG(log_total_partner_import) AS avg_import_scale_log,
                AVG(import_growth_slope) AS avg_import_growth_slope
            FROM country_clusters
            GROUP BY cluster_kmeans
            ORDER BY cluster_kmeans
            """,
            ["country_clusters"],
        ),
        (
            "vw_product_cluster_summary",
            """
            CREATE VIEW vw_product_cluster_summary AS
            SELECT
                cluster_kmeans,
                COUNT(*) AS products,
                AVG(opportunity_rate) AS avg_opportunity_rate,
                AVG(log_world_import) AS avg_world_import_log,
                AVG(world_import_slope) AS avg_world_import_slope
            FROM product_clusters
            GROUP BY cluster_kmeans
            ORDER BY cluster_kmeans
            """,
            ["product_clusters"],
        ),
        (
            "vw_sector_cluster_summary",
            """
            CREATE VIEW vw_sector_cluster_summary AS
            SELECT
                cluster_kmeans,
                COUNT(*) AS sectors,
                AVG(opportunity_rate) AS avg_opportunity_rate,
                AVG(log_world_import) AS avg_world_import_log,
                AVG(world_import_slope) AS avg_world_import_slope
            FROM sector_clusters
            GROUP BY cluster_kmeans
            ORDER BY cluster_kmeans
            """,
            ["sector_clusters"],
        ),
        (
            "vw_classification_label_counts",
            """
            CREATE VIEW vw_classification_label_counts AS
            SELECT
                predicted_label,
                COUNT(*) AS rows
            FROM classification_predictions_2023
            GROUP BY predicted_label
            ORDER BY rows DESC
            """,
            ["classification_predictions_2023"],
        ),
        (
            "vw_classification_best_model",
            """
            CREATE VIEW vw_classification_best_model AS
            SELECT
                model,
                accuracy,
                macro_precision,
                macro_recall,
                macro_f1
            FROM classification_model_comparison
            ORDER BY macro_f1 DESC
            LIMIT 1
            """,
            ["classification_model_comparison"],
        ),
    ]

    for view_name, sql, required_tables in view_sql:
        conn.execute(f"DROP VIEW IF EXISTS {view_name}")
        if all(table_exists(conn, table) for table in required_tables):
            conn.execute(sql)


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    imported = 0

    try:
        missing_required: list[Path] = []
        for csv_path, table_name, text_columns, required in SOURCES:
            print(f"\n[{table_name}]")
            df = load_csv(csv_path, text_columns)
            if df is None:
                if required:
                    missing_required.append(csv_path)
                continue

            df.to_sql(table_name, conn, if_exists="replace", index=False)
            row_count = len(df)
            print(f"  Imported {row_count:,} rows -> table '{table_name}'")
            imported += 1

        if missing_required:
            print("\nMissing required forecast files:")
            for path in missing_required:
                print(f"  - {path}")
            sys.exit(1)

        create_views(conn)
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
