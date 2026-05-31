from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover - Streamlit fallback for lean environments.
    px = None
    go = None
    PLOTLY_AVAILABLE = False


APP_TITLE = "Forecasting Global Export Opportunities for Algerian Exporters"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = PROJECT_ROOT / "data" / "forecast_outputs"
EDA_DIR = PROJECT_ROOT / "data" / "eda_outputs"
CLUSTERING_DIR = PROJECT_ROOT / "data" / "clustering_outputs"
CLASSIFICATION_DIR = PROJECT_ROOT / "data" / "classification_outputs"

FORECAST_FILES = {
    "opportunities": FORECAST_DIR / "top_forecasted_opportunities.csv",
    "countries": FORECAST_DIR / "priority_countries.csv",
    "products": FORECAST_DIR / "priority_products.csv",
    "forecasts": FORECAST_DIR / "final_forecasts.csv",
    "comparison": FORECAST_DIR / "historical_forecast_comparison.csv",
    "metrics": FORECAST_DIR / "forecast_model_metrics.csv",
}

EDA_FILES = {
    "exports_by_year": EDA_DIR / "exports_by_year.csv",
    "global_demand_products": EDA_DIR / "global_demand_products.csv",
    "opportunities_by_country": EDA_DIR / "opportunities_by_country.csv",
    "opportunities_by_product": EDA_DIR / "opportunities_by_product.csv",
    "sector_summary": EDA_DIR / "sector_summary.csv",
    "top_export_partners": EDA_DIR / "top_export_partners.csv",
    "top_export_products": EDA_DIR / "top_export_products.csv",
    "heatmap_demand": EDA_DIR / "country_product_heatmap_demand.csv",
    "heatmap_log": EDA_DIR / "country_product_heatmap_log.csv",
}

CLUSTERING_FILES = {
    "evaluation": CLUSTERING_DIR / "cluster_evaluation_summary.csv",
    "country_clusters": CLUSTERING_DIR / "country_clusters.csv",
    "product_clusters": CLUSTERING_DIR / "product_clusters.csv",
    "sector_clusters": CLUSTERING_DIR / "sector_clusters.csv",
    "priority_markets": CLUSTERING_DIR / "priority_market_ranking.csv",
    "cross_cluster": CLUSTERING_DIR / "cross_cluster_opportunity.csv",
}

CLASSIFICATION_FILES = {
    "model_comparison": CLASSIFICATION_DIR / "model_comparison.csv",
    "predictions": CLASSIFICATION_DIR / "predictions_2023.csv",
    "top_opportunities": CLASSIFICATION_DIR / "top_export_opportunities.csv",
    "country_opportunities": CLASSIFICATION_DIR / "opp_by_country_predicted.csv",
    "product_opportunities": CLASSIFICATION_DIR / "opp_by_product_predicted.csv",
    "feature_importance": CLASSIFICATION_DIR / "feature_importance_consensus.csv",
}

CLUSTERING_IMAGES = {
    "Country PCA": CLUSTERING_DIR / "country_kmeans_pca.png",
    "Product PCA": CLUSTERING_DIR / "product_kmeans_pca.png",
    "Sector PCA": CLUSTERING_DIR / "sector_kmeans_pca.png",
    "Priority Markets": CLUSTERING_DIR / "country_priority_ranking.png",
    "Cross-Cluster Opportunity": CLUSTERING_DIR / "cross_cluster_opportunity.png",
}

CLASSIFICATION_IMAGES = {
    "Model Comparison": CLASSIFICATION_DIR / "model_comparison_bar.png",
    "Confusion Matrices": CLASSIFICATION_DIR / "all_confusion_matrices.png",
    "Feature Importance": CLASSIFICATION_DIR / "feature_importance_consensus.png",
    "Opportunity Ranking": CLASSIFICATION_DIR / "top_opportunities_ranking.png",
}

HS_SECTOR_MAP: dict[str, str] = {
    "01": "Agriculture", "02": "Agriculture", "03": "Agriculture", "04": "Agriculture",
    "05": "Agriculture", "06": "Agriculture", "07": "Agriculture", "08": "Agriculture",
    "09": "Agriculture", "10": "Agriculture", "11": "Agriculture", "12": "Agriculture",
    "13": "Agriculture", "14": "Agriculture", "15": "Fats & Oils",
    "16": "Food Products", "17": "Food Products", "18": "Food Products",
    "19": "Food Products", "20": "Food Products", "21": "Food Products",
    "22": "Beverages", "23": "Food Products", "24": "Tobacco",
    "25": "Minerals & Stone", "26": "Ores & Metals", "27": "Energy (Hydrocarbons)",
    "28": "Chemicals", "29": "Chemicals", "30": "Pharmaceuticals",
    "31": "Chemicals", "32": "Chemicals", "33": "Chemicals", "34": "Chemicals",
    "35": "Chemicals", "36": "Chemicals", "37": "Chemicals", "38": "Chemicals",
    "39": "Plastics", "40": "Rubber",
    "41": "Leather", "42": "Leather", "43": "Leather",
    "44": "Wood", "45": "Wood", "46": "Wood",
    "47": "Paper", "48": "Paper", "49": "Paper",
    "50": "Textiles", "51": "Textiles", "52": "Textiles", "53": "Textiles",
    "54": "Textiles", "55": "Textiles", "56": "Textiles", "57": "Textiles",
    "58": "Textiles", "59": "Textiles", "60": "Textiles", "61": "Textiles",
    "62": "Textiles", "63": "Textiles",
    "64": "Footwear", "65": "Footwear", "66": "Footwear", "67": "Footwear",
    "68": "Stone & Glass", "69": "Stone & Glass", "70": "Stone & Glass",
    "71": "Precious Metals & Gems",
    "72": "Base Metals", "73": "Base Metals", "74": "Base Metals", "75": "Base Metals",
    "76": "Base Metals", "77": "Base Metals", "78": "Base Metals", "79": "Base Metals",
    "80": "Base Metals", "81": "Base Metals", "82": "Base Metals", "83": "Base Metals",
    "84": "Machinery & Equipment", "85": "Electronics",
    "86": "Transport", "87": "Transport", "88": "Transport", "89": "Transport",
    "90": "Precision Instruments", "91": "Precision Instruments", "92": "Precision Instruments",
    "93": "Weapons", "94": "Furniture", "95": "Toys & Sports", "96": "Miscellaneous",
    "97": "Art & Antiques",
}


def _add_sector_column(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "k" not in df.columns:
        return df
    chapter = df["k"].astype(str).str.zfill(6).str[:2]
    df = df.copy()
    df["sector"] = chapter.map(HS_SECTOR_MAP).fillna("Other")
    return df


EXPORT_VALUE_COLUMNS = ["alg_export_usd", "alg_export_v", "export_value", "value", "v"]
WORLD_DEMAND_COLUMNS = ["world_import_usd", "world_import_v", "partner_import_v", "demand", "value"]
COUNT_COLUMNS = ["opportunity_rows", "count", "n_opportunities", "label_opportunity", "total"]
YEAR_COLUMNS = ["t", "year"]


st.set_page_config(page_title=APP_TITLE, page_icon="DZ", layout="wide")


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
        h1, h2, h3 {letter-spacing: 0;}
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.35);
            padding: 14px 16px;
            border-radius: 8px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
        }
        div[data-testid="stMetricLabel"] p {
            font-size: 0.88rem;
            color: inherit;
            opacity: 0.78;
        }
        div[data-testid="stMetricValue"] {
            color: inherit;
        }
        .section-note {
            color: inherit;
            opacity: 0.82;
            font-size: 0.98rem;
            line-height: 1.45;
        }
        .small-muted {
            color: inherit;
            opacity: 0.74;
            font-size: 0.86rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def safe_load_csv(path: str) -> pd.DataFrame | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        st.warning(f"Could not read {csv_path.name}: {exc}")
        return None


def load_required_data() -> dict[str, pd.DataFrame] | None:
    missing = [path for path in FORECAST_FILES.values() if not path.exists()]
    if missing:
        st.error("Please run notebooks/07_forecasting.ipynb first to generate forecast outputs.")
        with st.expander("Missing required forecast files", expanded=True):
            for path in missing:
                st.write(path.relative_to(PROJECT_ROOT))
        return None
    return {name: safe_load_csv(str(path)) for name, path in FORECAST_FILES.items()}


def load_optional_eda() -> dict[str, pd.DataFrame]:
    return {
        name: df
        for name, path in EDA_FILES.items()
        if (df := safe_load_csv(str(path))) is not None
    }


def load_optional_outputs(files: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {
        name: df
        for name, path in files.items()
        if (df := safe_load_csv(str(path))) is not None
    }


def has_cols(df: pd.DataFrame | None, cols: Iterable[str]) -> bool:
    return df is not None and all(col in df.columns for col in cols)


def first_available(df: pd.DataFrame, candidates: list[str]) -> str | None:
    if df is None:
        return None
    return next((col for col in candidates if col in df.columns), None)


def best_model_from_metrics(metrics: pd.DataFrame | None) -> tuple[str | None, str | None]:
    if metrics is None or metrics.empty or "model" not in metrics.columns:
        return None, None
    metric_col = "WMAPE" if "WMAPE" in metrics.columns else "RMSE" if "RMSE" in metrics.columns else None
    if metric_col is None:
        return None, None
    valid = metrics.copy()
    valid[metric_col] = pd.to_numeric(valid[metric_col], errors="coerce")
    valid = valid.dropna(subset=[metric_col]).sort_values(metric_col)
    if valid.empty:
        return None, metric_col
    return str(valid.iloc[0]["model"]), metric_col


def fmt_number(value: float | int | None) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def format_table(df: pd.DataFrame, currency_cols: Iterable[str] = (), pct_cols: Iterable[str] = ()):
    fmt = {col: "{:,.2f}" for col in currency_cols if col in df.columns}
    fmt.update({col: "{:,.2f}%" for col in pct_cols if col in df.columns})
    try:
        return df.style.format(fmt)
    except AttributeError:
        return df


def bar_chart(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None, orientation: str = "v") -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No data available for this chart with the current filters.")
        return
    if PLOTLY_AVAILABLE:
        fig = px.bar(df, x=x, y=y, color=color if color in df.columns else None, title=title, orientation=orientation)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(df.set_index(x)[y])


def line_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str) -> None:
    if df.empty or not has_cols(df, [x, y, color]):
        st.info("No data available for this chart with the current filters.")
        return
    if PLOTLY_AVAILABLE:
        fig = px.line(df, x=x, y=y, color=color, markers=True, title=title)
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(df.pivot_table(index=x, columns=color, values=y, aggfunc="sum"))


def scatter_chart(df: pd.DataFrame, x: str, y: str, size: str | None, color: str | None, title: str) -> None:
    if df.empty or x not in df.columns or y not in df.columns:
        st.info("No data available for this chart with the current filters.")
        return
    if PLOTLY_AVAILABLE:
        plot_df = df.copy()
        size_col = None
        if size in plot_df.columns:
            numeric_size = pd.to_numeric(plot_df[size], errors="coerce").fillna(0)
            plot_df["_marker_size"] = numeric_size.clip(lower=0)
            if plot_df["_marker_size"].gt(0).any():
                size_col = "_marker_size"
        fig = px.scatter(
            plot_df,
            x=x,
            y=y,
            size=size_col,
            color=color if color in df.columns else None,
            hover_data=[col for col in ["country_name", "k", "description_short"] if col in df.columns],
            title=title,
        )
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.scatter_chart(df[[x, y]])


def show_output_images(images: dict[str, Path], key: str) -> None:
    available = {label: path for label, path in images.items() if path.exists()}
    if not available:
        return
    selected = st.selectbox("Saved diagnostic plot", list(available.keys()), key=f"{key}_image_select")
    st.image(str(available[selected]), caption=selected, use_container_width=True)


def normalize_filter_values(series: pd.Series) -> list:
    values = series.dropna().unique().tolist()
    return sorted(values, key=lambda item: str(item))


def filter_like_dashboard(df: pd.DataFrame, filtered_opps: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    filtered = df.copy()
    if "country_name" in filtered.columns and "country_name" in filtered_opps.columns:
        countries = set(filtered_opps["country_name"].dropna().astype(str))
        filtered = filtered[filtered["country_name"].astype(str).isin(countries)]
    if "k" in filtered.columns and "k" in filtered_opps.columns:
        products = set(filtered_opps["k"].dropna().astype(str))
        filtered = filtered[filtered["k"].astype(str).isin(products)]
    return filtered


def apply_sidebar_filters(opps: pd.DataFrame, forecasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.header("Filters")
    filtered_opps = opps.copy()
    filtered_forecasts = forecasts.copy()

    if "country_name" in opps.columns:
        countries = normalize_filter_values(opps["country_name"])
        selected = st.sidebar.multiselect("Country", countries, default=[])
        if selected:
            filtered_opps = filtered_opps[filtered_opps["country_name"].isin(selected)]
            if "country_name" in filtered_forecasts.columns:
                filtered_forecasts = filtered_forecasts[filtered_forecasts["country_name"].isin(selected)]

    if "description_short" in opps.columns:
        if "k" in opps.columns:
            product_lookup = (
                opps[["k", "description_short"]]
                .dropna(subset=["k"])
                .drop_duplicates()
                .assign(label=lambda df: df["k"].astype(str) + " - " + df["description_short"].astype(str))
                .sort_values("label")
            )
            product_options = product_lookup["label"].tolist()
            label_to_code = dict(zip(product_lookup["label"], product_lookup["k"]))
            selected = st.sidebar.multiselect("Product", product_options, default=[])
            selected_codes = [label_to_code[label] for label in selected]
        else:
            selected = st.sidebar.multiselect("Product", normalize_filter_values(opps["description_short"]), default=[])
            selected_codes = []
        if selected:
            if selected_codes and "k" in filtered_opps.columns:
                filtered_opps = filtered_opps[filtered_opps["k"].isin(selected_codes)]
                if "k" in filtered_forecasts.columns:
                    filtered_forecasts = filtered_forecasts[filtered_forecasts["k"].isin(selected_codes)]
            else:
                filtered_opps = filtered_opps[filtered_opps["description_short"].isin(selected)]
                if "description_short" in filtered_forecasts.columns:
                    filtered_forecasts = filtered_forecasts[filtered_forecasts["description_short"].isin(selected)]

    if "year" in opps.columns:
        years = pd.to_numeric(opps["year"], errors="coerce").dropna()
        if not years.empty:
            min_year, max_year = int(years.min()), int(years.max())
            selected_range = st.sidebar.slider("Year range", min_year, max_year, (min_year, max_year))
            filtered_opps = filtered_opps[filtered_opps["year"].between(*selected_range)]
            if "year" in filtered_forecasts.columns:
                filtered_forecasts = filtered_forecasts[filtered_forecasts["year"].between(*selected_range)]

    if "forecast_horizon" in opps.columns:
        horizons = normalize_filter_values(opps["forecast_horizon"])
        selected = st.sidebar.multiselect("Forecast horizon", horizons, default=horizons)
        if selected:
            filtered_opps = filtered_opps[filtered_opps["forecast_horizon"].isin(selected)]
            if "forecast_horizon" in filtered_forecasts.columns:
                filtered_forecasts = filtered_forecasts[filtered_forecasts["forecast_horizon"].isin(selected)]

    model_col = "model_used" if "model_used" in opps.columns else "model"
    if model_col in opps.columns:
        models = normalize_filter_values(opps[model_col])
        selected = st.sidebar.multiselect("Model used", models, default=models)
        if selected:
            filtered_opps = filtered_opps[filtered_opps[model_col].isin(selected)]
            if "model" in filtered_forecasts.columns:
                filtered_forecasts = filtered_forecasts[filtered_forecasts["model"].isin(selected)]

    if "forecast_opportunity_score" in opps.columns:
        score = pd.to_numeric(opps["forecast_opportunity_score"], errors="coerce").dropna()
        if not score.empty:
            selected_score = st.sidebar.slider(
                "Opportunity score range",
                0.0,
                1.0,
                (float(max(0, score.min())), float(min(1, score.max()))),
                0.01,
            )
            filtered_opps = filtered_opps[
                filtered_opps["forecast_opportunity_score"].between(selected_score[0], selected_score[1])
            ]

    if "forecast_scope" in opps.columns:
        scopes = normalize_filter_values(opps["forecast_scope"])
        selected = st.sidebar.multiselect("Forecast scope", scopes, default=scopes)
        if selected:
            filtered_opps = filtered_opps[filtered_opps["forecast_scope"].isin(selected)]
            if "forecast_scope" in filtered_forecasts.columns:
                filtered_forecasts = filtered_forecasts[filtered_forecasts["forecast_scope"].isin(selected)]

    if "is_hydrocarbon" in opps.columns:
        hydro_choice = st.sidebar.radio(
            "Hydrocarbon filter",
            ["All", "Non-hydrocarbon only", "Hydrocarbon only"],
            index=1 if not opps["is_hydrocarbon"].astype(str).str.lower().isin(["true", "1"]).any() else 0,
        )
        hydro = opps["is_hydrocarbon"].astype(str).str.lower().isin(["true", "1", "yes"])
        if hydro_choice == "Non-hydrocarbon only":
            filtered_opps = filtered_opps.loc[~hydro.reindex(filtered_opps.index, fill_value=False)]
            if "is_hydrocarbon" in filtered_forecasts.columns:
                fhydro = filtered_forecasts["is_hydrocarbon"].astype(str).str.lower().isin(["true", "1", "yes"])
                filtered_forecasts = filtered_forecasts.loc[~fhydro]
        elif hydro_choice == "Hydrocarbon only":
            filtered_opps = filtered_opps.loc[hydro.reindex(filtered_opps.index, fill_value=False)]
            if "is_hydrocarbon" in filtered_forecasts.columns:
                fhydro = filtered_forecasts["is_hydrocarbon"].astype(str).str.lower().isin(["true", "1", "yes"])
                filtered_forecasts = filtered_forecasts.loc[fhydro]

    if "sector" in opps.columns:
        sectors = normalize_filter_values(opps["sector"])
        selected = st.sidebar.multiselect("Sector", sectors, default=sectors)
        if selected:
            filtered_opps = filtered_opps[filtered_opps["sector"].isin(selected)]

    return filtered_opps, filtered_forecasts


def executive_summary(opps: pd.DataFrame, products: pd.DataFrame, metrics: pd.DataFrame) -> None:
    st.header("1. Executive Summary")
    if opps.empty:
        st.info("No ranked opportunities match the current filters.")
        return
    best = opps.sort_values("forecast_opportunity_score", ascending=False).iloc[0]
    best_metric = None
    if metrics is not None and not metrics.empty:
        metric_col = "WMAPE" if "WMAPE" in metrics.columns else "RMSE"
        valid_metrics = metrics.dropna(subset=[metric_col])
        if not valid_metrics.empty:
            best_metric = valid_metrics.sort_values(metric_col).iloc[0]

    sectors = opps["sector"].nunique() if "sector" in opps.columns else None
    avg_growth = pd.to_numeric(opps.get("predicted_growth_pct", pd.Series(dtype=float)), errors="coerce").mean()
    scope = ", ".join(map(str, opps["forecast_scope"].dropna().unique()[:3])) if "forecast_scope" in opps.columns else "n/a"

    cols = st.columns(4)
    cols[0].metric("Countries analyzed", f"{opps['country_name'].nunique():,}" if "country_name" in opps.columns else "n/a")
    cols[1].metric("Products analyzed", f"{opps['k'].nunique():,}" if "k" in opps.columns else "n/a")
    cols[2].metric("Ranked opportunities", f"{len(opps):,}")
    cols[3].metric("Avg predicted growth", f"{avg_growth:.2f}%" if pd.notna(avg_growth) else "n/a")

    cols = st.columns(4)
    cols[0].metric("Best country", str(best.get("country_name", "n/a")))
    cols[1].metric("Best product", str(best.get("k", "n/a")))
    cols[2].metric("Best model", str(best_metric.get("model", "n/a")) if best_metric is not None else "n/a")
    cols[3].metric("Sectors analyzed", f"{sectors:,}" if sectors is not None else "Not available")
    st.markdown(f"<p class='small-muted'>Forecast scope: {scope}</p>", unsafe_allow_html=True)


def opportunities_explorer(opps: pd.DataFrame) -> None:
    st.header("2. Export Opportunities Explorer")
    st.markdown(
        "<p class='section-note'>Explore ranked country-product opportunities by forecasted demand, predicted growth, market penetration, RCA, and opportunity score.</p>",
        unsafe_allow_html=True,
    )
    if "rca" in opps.columns:
        rca_missing_pct = opps["rca"].isna().mean() * 100
        if rca_missing_pct > 50:
            st.info(
                f"RCA values are missing for {rca_missing_pct:.0f}% of ranked rows. "
                "When RCA is missing, its weight (10%) is redistributed to the other scoring components. "
                "RCA availability depends on whether Algeria has any reported export flow for that product-country pair."
            )
    if opps.empty:
        st.info("No opportunities match the selected filters.")
        return

    chart_data = opps.sort_values("forecast_opportunity_score", ascending=False).head(20).copy()
    chart_data["label"] = chart_data["country_name"].astype(str) + " | " + chart_data["k"].astype(str)
    bar_chart(chart_data.sort_values("forecast_opportunity_score"), "forecast_opportunity_score", "label", "Top Opportunities by Score", orientation="h")

    table_cols = [
        "rank",
        "country_name",
        "k",
        "description_short",
        "forecasted_value",
        "predicted_growth_pct",
        "forecast_opportunity_score",
        "market_penetration",
        "rca",
        "model_used",
        "reason_or_interpretation",
    ]
    table = opps[[col for col in table_cols if col in opps.columns]].sort_values("forecast_opportunity_score", ascending=False)
    st.dataframe(
        format_table(table, currency_cols=["forecasted_value"], pct_cols=["predicted_growth_pct"]),
        use_container_width=True,
        height=440,
    )


def priority_markets(countries: pd.DataFrame, filtered_opps: pd.DataFrame) -> None:
    st.header("3. Priority International Markets")
    st.markdown(
        "<p class='section-note'>Which countries should Algerian exporters focus on first?</p>",
        unsafe_allow_html=True,
    )
    if countries is None or countries.empty:
        st.info("priority_countries.csv is not available.")
        return
    visible_countries = filtered_opps["country_name"].unique() if "country_name" in filtered_opps.columns else countries["country_name"].unique()
    data = countries[countries["country_name"].isin(visible_countries)].copy()
    top = data.sort_values("average_opportunity_score", ascending=False).head(15)

    if PLOTLY_AVAILABLE and "iso3" in data.columns:
        map_data = data.copy()
        map_data["iso3"] = map_data["iso3"].astype(str).str.strip().str.upper()
        fig = px.choropleth(
            map_data,
            locations="iso3",
            color="average_opportunity_score",
            hover_name="country_name",
            hover_data={"total_forecasted_value": True, "total_predicted_growth": True, "high_opportunity_products": True},
            color_continuous_scale="YlOrRd",
            title="World Map: Average Forecast Opportunity Score by Country",
            labels={"average_opportunity_score": "Opportunity Score"},
        )
        fig.update_layout(height=480, margin=dict(l=0, r=0, t=55, b=0), geo=dict(showframe=False, showcoastlines=True))
        st.plotly_chart(fig, use_container_width=True)

    bar_chart(top.sort_values("average_opportunity_score"), "average_opportunity_score", "country_name", "Top 15 Priority Countries by Opportunity Score", orientation="h")
    st.dataframe(format_table(data.head(50)), use_container_width=True, height=360)


def priority_products(products: pd.DataFrame, filtered_opps: pd.DataFrame) -> None:
    st.header("4. Priority Products")
    st.markdown(
        "<p class='section-note'>Which products appear most promising for future export opportunities?</p>",
        unsafe_allow_html=True,
    )
    if products is None or products.empty:
        st.info("priority_products.csv is not available.")
        return
    visible_products = filtered_opps["k"].unique() if "k" in filtered_opps.columns else products["k"].unique()
    data = products[products["k"].isin(visible_products)].copy()
    data["label"] = data["k"].astype(str) + " - " + data["description_short"].astype(str).str.slice(0, 46)
    top = data.sort_values("average_opportunity_score", ascending=False).head(15)
    bar_chart(top.sort_values("average_opportunity_score"), "average_opportunity_score", "label", "Top Priority Products", orientation="h")
    st.dataframe(format_table(data.drop(columns=["label"], errors="ignore").head(50)), use_container_width=True, height=360)


def demand_and_trade_flows(forecasts: pd.DataFrame, comparison: pd.DataFrame, eda: dict[str, pd.DataFrame]) -> None:
    st.header("5. Global Demand Trends and Trade Flows")
    if not forecasts.empty and has_cols(forecasts, ["year", "forecast_partner_import_v", "description_short"]):
        product_trend = (
            forecasts.groupby(["year", "description_short"], observed=True)["forecast_partner_import_v"]
            .sum()
            .reset_index()
            .sort_values("forecast_partner_import_v", ascending=False)
        )
        top_products = product_trend.groupby("description_short")["forecast_partner_import_v"].sum().nlargest(5).index
        line_chart(product_trend[product_trend["description_short"].isin(top_products)], "year", "forecast_partner_import_v", "description_short", "Forecasted Partner Import Demand by Product")

    cols = st.columns(2)
    with cols[0]:
        exports = eda.get("exports_by_year")
        year_col = first_available(exports, YEAR_COLUMNS)
        export_col = first_available(exports, EXPORT_VALUE_COLUMNS)
        if exports is not None and year_col and export_col:
            st.line_chart(exports.set_index(year_col)[export_col])
        else:
            st.info("Historical exports_by_year.csv is not available.")
    with cols[1]:
        global_products = eda.get("global_demand_products")
        demand_col = first_available(global_products, WORLD_DEMAND_COLUMNS)
        if global_products is not None and "description_short" in global_products.columns and demand_col:
            top = global_products.nlargest(12, demand_col).copy()
            top["label"] = top["description_short"].astype(str).str.slice(0, 48)
            bar_chart(top.sort_values(demand_col), demand_col, "label", "Top Global Demand Products", orientation="h")
        else:
            st.info("global_demand_products.csv is not available.")

    cols = st.columns(2)
    with cols[0]:
        partners = eda.get("top_export_partners")
        partner_export_col = first_available(partners, EXPORT_VALUE_COLUMNS)
        if partners is not None and "country_name" in partners.columns and partner_export_col:
            top = partners.nlargest(12, partner_export_col)
            bar_chart(top.sort_values(partner_export_col), partner_export_col, "country_name", "Top Historical Export Partners", orientation="h")
    with cols[1]:
        products = eda.get("top_export_products")
        product_export_col = first_available(products, EXPORT_VALUE_COLUMNS)
        if products is not None and "description_short" in products.columns and product_export_col:
            top = products.nlargest(12, product_export_col).copy()
            top["label"] = top["description_short"].astype(str).str.slice(0, 48)
            bar_chart(top.sort_values(product_export_col), product_export_col, "label", "Top Historical Export Products", orientation="h")


def growth_and_potential(opps: pd.DataFrame) -> None:
    st.header("6. Predicted Export Growth and Market Potential")
    if opps.empty:
        st.info("No growth data available with the current filters.")
        return
    cols = st.columns(2)
    with cols[0]:
        country_growth = opps.groupby("country_name", observed=True)["predicted_growth_abs"].sum().nlargest(15).reset_index()
        bar_chart(country_growth.sort_values("predicted_growth_abs"), "predicted_growth_abs", "country_name", "Top Countries by Predicted Growth", orientation="h")
    with cols[1]:
        product_growth = opps.groupby(["k", "description_short"], observed=True)["predicted_growth_abs"].sum().nlargest(15).reset_index()
        product_growth["label"] = product_growth["k"].astype(str) + " - " + product_growth["description_short"].astype(str).str.slice(0, 42)
        bar_chart(product_growth.sort_values("predicted_growth_abs"), "predicted_growth_abs", "label", "Top Products by Predicted Growth", orientation="h")
    scatter_chart(opps, "forecasted_value", "forecast_opportunity_score", "predicted_growth_abs", "country_name", "Forecasted Value vs Opportunity Score")


def historical_vs_forecasted(forecasts: pd.DataFrame, comparison: pd.DataFrame, metrics: pd.DataFrame) -> None:
    st.header("7. Historical vs Forecasted Trade Indicators")
    if forecasts.empty:
        st.info("No forecast rows match the current filters.")
        return
    selection = forecasts.copy()
    if "series_id" in selection.columns:
        top_series = selection.groupby("series_id")["forecast_partner_import_v"].sum().nlargest(1).index.tolist()
        selection = selection[selection["series_id"].isin(top_series)]

    future = selection[["year", "forecast_partner_import_v"]].copy()
    future["series"] = "Forecasted partner import demand"
    future = future.rename(columns={"forecast_partner_import_v": "value"})
    latest_hist = selection[["year", "historical_partner_import_v"]].drop_duplicates().copy()
    latest_hist["year"] = latest_hist["year"].min() - 1
    latest_hist["series"] = "Latest historical partner import demand"
    latest_hist = latest_hist.rename(columns={"historical_partner_import_v": "value"})
    combined = pd.concat([latest_hist[["year", "value", "series"]], future[["year", "value", "series"]]], ignore_index=True)
    line_chart(combined, "year", "value", "series", "Historical Baseline vs Future Forecast")

    if comparison is not None and not comparison.empty and has_cols(comparison, ["t", "actual", "prediction", "model"]):
        best_model, metric_col = best_model_from_metrics(metrics)
        if best_model is None or best_model not in set(comparison["model"].dropna().astype(str)):
            best_model = str(comparison["model"].dropna().iloc[0])
            metric_col = "available comparison data"
        comp = comparison[comparison["model"].astype(str).eq(best_model)].copy()
        comp = comp.groupby("t", observed=True)[["actual", "prediction"]].mean().reset_index()
        comp_long = comp.melt(id_vars="t", value_vars=["actual", "prediction"], var_name="series", value_name="value")
        line_chart(comp_long, "t", "value", "series", f"Actual vs Predicted on Test Years ({best_model}, best by {metric_col})")


def model_performance(metrics: pd.DataFrame) -> None:
    st.header("8. Forecast Model Performance")
    st.markdown(
        "<p class='section-note'>Lower MAE, RMSE, and WMAPE indicate better forecast accuracy. MAPE can be unstable when real trade values are very small or zero. "
        "ARIMA requires the <code>statsmodels</code> package; install it via <code>pip install statsmodels</code> to enable ARIMA comparison.</p>",
        unsafe_allow_html=True,
    )
    if metrics is None or metrics.empty:
        st.info("forecast_model_metrics.csv is not available.")
        return
    best_model, metric_col = best_model_from_metrics(metrics)
    if best_model is not None:
        st.success(f"Best selected model by {metric_col}: {best_model}")
    if metrics is not None and "notes" in metrics.columns:
        arima_notes = metrics.loc[metrics["notes"].astype(str).str.len() > 0, "notes"]
        if not arima_notes.empty:
            st.warning(f"Note: {arima_notes.iloc[0]}")
    cols = st.columns(2)
    with cols[0]:
        if "RMSE" in metrics.columns:
            bar_chart(metrics.dropna(subset=["RMSE"]).sort_values("RMSE"), "model", "RMSE", "RMSE by Model")
    with cols[1]:
        if metric_col in metrics.columns:
            bar_chart(metrics.dropna(subset=[metric_col]).sort_values(metric_col), "model", metric_col, f"{metric_col} by Model")
    st.dataframe(format_table(metrics), use_container_width=True)


def historical_eda_context(eda: dict[str, pd.DataFrame]) -> None:
    st.header("9. Historical EDA Context")
    st.markdown(
        "<p class='section-note'>These panels use data/eda_outputs only as historical support. They are not the main forecasting dataset.</p>",
        unsafe_allow_html=True,
    )
    sector = eda.get("sector_summary")
    sector_demand_col = first_available(sector, WORLD_DEMAND_COLUMNS + ["partner_demand_usd", "partner_demand"])
    if sector is not None and "sector" in sector.columns and sector_demand_col:
        bar_chart(sector.nlargest(12, sector_demand_col).sort_values(sector_demand_col), sector_demand_col, "sector", "Historical Partner Demand by Sector", orientation="h")

    cols = st.columns(2)
    with cols[0]:
        country = eda.get("opportunities_by_country")
        country_count_col = first_available(country, COUNT_COLUMNS)
        if country is not None and "country_name" in country.columns and country_count_col:
            bar_chart(country.nlargest(12, country_count_col).sort_values(country_count_col), country_count_col, "country_name", "Historical Opportunity Rows by Country", orientation="h")
    with cols[1]:
        product = eda.get("opportunities_by_product")
        product_count_col = first_available(product, COUNT_COLUMNS)
        if product is not None and "description_short" in product.columns and product_count_col:
            top = product.nlargest(12, product_count_col).copy()
            top["label"] = top["description_short"].astype(str).str.slice(0, 48)
            bar_chart(top.sort_values(product_count_col), product_count_col, "label", "Historical Opportunity Rows by Product", orientation="h")

    heatmap = eda.get("heatmap_log")
    if heatmap is None:
        heatmap = eda.get("heatmap_demand")
    if heatmap is not None and "country_name" in heatmap.columns and PLOTLY_AVAILABLE:
        matrix = heatmap.set_index("country_name")
        fig = px.imshow(matrix, aspect="auto", title="Country-Product Demand Heatmap")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)


def sector_analysis(opps: pd.DataFrame) -> None:
    st.header("10. Sector-Level Demand and Opportunity Analysis")
    st.markdown(
        "<p class='section-note'>HS product codes are grouped into broad trade sectors. This view shows which sectors drive the most forecasted demand and predicted growth.</p>",
        unsafe_allow_html=True,
    )
    if opps.empty:
        st.info("No opportunity data available for sector analysis.")
        return
    df = _add_sector_column(opps)
    sector_agg = (
        df.groupby("sector", observed=True)
        .agg(
            total_forecasted_value=("forecasted_value", "sum"),
            total_predicted_growth=("predicted_growth_abs", "sum"),
            avg_opportunity_score=("forecast_opportunity_score", "mean"),
            opportunity_count=("forecast_opportunity_score", "size"),
        )
        .reset_index()
        .sort_values("total_forecasted_value", ascending=False)
    )
    cols = st.columns(2)
    with cols[0]:
        bar_chart(sector_agg.sort_values("total_forecasted_value"), "total_forecasted_value", "sector", "Total Forecasted Demand by Sector", orientation="h")
    with cols[1]:
        bar_chart(sector_agg.sort_values("avg_opportunity_score"), "avg_opportunity_score", "sector", "Average Opportunity Score by Sector", orientation="h")
    st.dataframe(format_table(sector_agg, currency_cols=["total_forecasted_value", "total_predicted_growth"]), use_container_width=True, height=320)


def clustering_analysis(clustering: dict[str, pd.DataFrame]) -> None:
    st.header("11. Clustering-Based Market Segmentation")
    st.markdown(
        "<p class='section-note'>These panels read data/clustering_outputs to show country, product, and sector segments, cluster quality, and priority market groups.</p>",
        unsafe_allow_html=True,
    )
    if not clustering:
        st.info("No clustering outputs found in data/clustering_outputs.")
        return

    evaluation = clustering.get("evaluation")
    if evaluation is not None and not evaluation.empty:
        eval_table = evaluation.copy()
        if "Silhouette" in eval_table.columns:
            eval_table["Silhouette"] = pd.to_numeric(eval_table["Silhouette"], errors="coerce")
            best = eval_table.dropna(subset=["Silhouette"]).sort_values("Silhouette", ascending=False).head(1)
            if not best.empty:
                st.success(f"Best clustering quality by Silhouette: {best.iloc[0].get('Task')} - {best.iloc[0].get('Model')}")
        st.dataframe(format_table(eval_table), use_container_width=True, height=250)

    cols = st.columns(2)
    with cols[0]:
        priority = clustering.get("priority_markets")
        if priority is not None and not priority.empty and has_cols(priority, ["Country", "Priority score"]):
            priority = priority.copy()
            priority["Priority score"] = pd.to_numeric(priority["Priority score"], errors="coerce")
            top = priority.dropna(subset=["Priority score"]).nlargest(15, "Priority score")
            bar_chart(top.sort_values("Priority score"), "Priority score", "Country", "Cluster Priority Market Ranking", orientation="h")
        else:
            st.info("priority_market_ranking.csv is not available.")
    with cols[1]:
        country_clusters = clustering.get("country_clusters")
        if country_clusters is not None and "cluster_kmeans" in country_clusters.columns:
            summary_cols = {
                "Countries": ("country_name", "count") if "country_name" in country_clusters.columns else ("j", "count"),
            }
            if "opportunity_rate" in country_clusters.columns:
                summary_cols["Avg opportunity rate"] = ("opportunity_rate", "mean")
            if "log_total_partner_import" in country_clusters.columns:
                summary_cols["Avg import scale log"] = ("log_total_partner_import", "mean")
            country_summary = (
                country_clusters.groupby("cluster_kmeans", observed=True)
                .agg(**summary_cols)
                .reset_index()
                .sort_values("cluster_kmeans")
            )
            st.dataframe(format_table(country_summary), use_container_width=True, height=420)
        else:
            st.info("country_clusters.csv is not available.")

    cols = st.columns(2)
    with cols[0]:
        product_clusters = clustering.get("product_clusters")
        if product_clusters is not None and "cluster_kmeans" in product_clusters.columns:
            product_summary = product_clusters.groupby("cluster_kmeans", observed=True).size().reset_index(name="Products")
            bar_chart(product_summary, "cluster_kmeans", "Products", "Products per Cluster")
    with cols[1]:
        sector_clusters = clustering.get("sector_clusters")
        if sector_clusters is not None and "cluster_kmeans" in sector_clusters.columns:
            sector_summary = sector_clusters.groupby("cluster_kmeans", observed=True).size().reset_index(name="Sectors")
            bar_chart(sector_summary, "cluster_kmeans", "Sectors", "Sectors per Cluster")

    cross_cluster = clustering.get("cross_cluster")
    if cross_cluster is not None and not cross_cluster.empty and PLOTLY_AVAILABLE:
        matrix = cross_cluster.set_index(cross_cluster.columns[0])
        fig = px.imshow(matrix, aspect="auto", title="Country-Product Cluster Opportunity Matrix")
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Saved clustering diagnostic plots"):
        show_output_images(CLUSTERING_IMAGES, "clustering")


def classification_analysis(classification: dict[str, pd.DataFrame], filtered_opps: pd.DataFrame) -> None:
    st.header("12. Classification-Based Opportunity Ranking")
    st.markdown(
        "<p class='section-note'>These panels read data/classification_outputs to show High / Medium / Low opportunity predictions, classifier performance, and feature importance.</p>",
        unsafe_allow_html=True,
    )
    if not classification:
        st.info("No classification outputs found in data/classification_outputs.")
        return

    model_comparison = classification.get("model_comparison")
    if model_comparison is not None and not model_comparison.empty:
        metrics = model_comparison.copy()
        for col in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
            if col in metrics.columns:
                metrics[col] = pd.to_numeric(metrics[col], errors="coerce")
        ranking_metric = "macro_f1" if "macro_f1" in metrics.columns else "accuracy" if "accuracy" in metrics.columns else None
        if ranking_metric:
            best = metrics.dropna(subset=[ranking_metric]).sort_values(ranking_metric, ascending=False).head(1)
            if not best.empty:
                st.success(f"Best classifier by {ranking_metric}: {best.iloc[0].get('model')}")
            cols = st.columns(2)
            with cols[0]:
                bar_chart(metrics.sort_values(ranking_metric), "model", ranking_metric, f"Classification {ranking_metric} by Model")
            with cols[1]:
                if "accuracy" in metrics.columns:
                    bar_chart(metrics.sort_values("accuracy"), "model", "accuracy", "Classification Accuracy by Model")
        st.dataframe(format_table(metrics), use_container_width=True, height=230)

    predictions = filter_like_dashboard(classification.get("predictions"), filtered_opps)
    if predictions is not None and not predictions.empty and "predicted_label" in predictions.columns:
        label_counts = predictions["predicted_label"].value_counts().rename_axis("Predicted label").reset_index(name="Rows")
        bar_chart(label_counts, "Predicted label", "Rows", "Predicted Opportunity Classes")

    cols = st.columns(2)
    with cols[0]:
        by_country = classification.get("country_opportunities")
        if by_country is not None and has_cols(by_country, ["country_name", "n_high_opportunities"]):
            by_country = by_country.copy()
            by_country["n_high_opportunities"] = pd.to_numeric(by_country["n_high_opportunities"], errors="coerce")
            top = by_country.dropna(subset=["n_high_opportunities"]).nlargest(15, "n_high_opportunities")
            bar_chart(top.sort_values("n_high_opportunities"), "n_high_opportunities", "country_name", "High Opportunities by Country", orientation="h")
    with cols[1]:
        by_product = classification.get("product_opportunities")
        if by_product is not None and has_cols(by_product, ["description_short", "n_high_opportunities"]):
            by_product = by_product.copy()
            by_product["n_high_opportunities"] = pd.to_numeric(by_product["n_high_opportunities"], errors="coerce")
            top = by_product.dropna(subset=["n_high_opportunities"]).nlargest(15, "n_high_opportunities")
            top["label"] = top["description_short"].astype(str).str.slice(0, 50)
            bar_chart(top.sort_values("n_high_opportunities"), "n_high_opportunities", "label", "High Opportunities by Product", orientation="h")

    feature_importance = classification.get("feature_importance")
    if feature_importance is not None and has_cols(feature_importance, ["feature", "avg_rank"]):
        importance = feature_importance.copy()
        importance["avg_rank"] = pd.to_numeric(importance["avg_rank"], errors="coerce")
        top_features = importance.dropna(subset=["avg_rank"]).sort_values("avg_rank").head(15).sort_values("avg_rank", ascending=False)
        bar_chart(top_features, "avg_rank", "feature", "Consensus Feature Importance (Lower Rank Is Better)", orientation="h")

    top_opportunities = filter_like_dashboard(classification.get("top_opportunities"), filtered_opps)
    if top_opportunities is not None and not top_opportunities.empty:
        top_opportunities = top_opportunities.drop(columns=[col for col in top_opportunities.columns if col.startswith("Unnamed")], errors="ignore")
        table_cols = [
            "country_name",
            "description_short",
            "partner_import_v",
            "global_demand_rank",
            "world_demand_growth",
            "market_penetration",
            "rca",
            "dist_km",
            "predicted_label",
        ]
        table = top_opportunities[[col for col in table_cols if col in top_opportunities.columns]].head(50)
        st.dataframe(
            format_table(table, currency_cols=["partner_import_v"], pct_cols=["world_demand_growth"]),
            use_container_width=True,
            height=420,
        )

    with st.expander("Saved classification diagnostic plots"):
        show_output_images(CLASSIFICATION_IMAGES, "classification")


def interpretation_section(non_hydrocarbon_default: bool = False) -> None:
    st.header("13. Interpretation for CACI, Exporters, and Policymakers")
    if non_hydrocarbon_default:
        st.info("The current default view focuses on non-hydrocarbon diversification opportunities.")
    st.markdown(
        """
        This dashboard helps CACI experts, exporters, and policymakers identify international markets where future demand is expected to grow.
        A strong opportunity is usually a market with high forecasted demand, positive predicted growth, strong global demand, and low current Algerian market penetration.

        A high opportunity score means the country-product pair looks attractive based on forecasted demand and market indicators.
        Low market penetration can indicate an underexploited market.
        RCA helps show whether Algeria may have comparative advantage, when available.
        Forecasts are decision-support tools, not guaranteed future values.
        """
    )


def main() -> None:
    inject_style()
    st.title(APP_TITLE)
    st.caption("Forecast-based decision support dashboard for export diversification and international market prioritization.")

    data = load_required_data()
    if data is None:
        return
    eda = load_optional_eda()
    clustering = load_optional_outputs(CLUSTERING_FILES)
    classification = load_optional_outputs(CLASSIFICATION_FILES)

    opportunities = data["opportunities"]
    countries = data["countries"]
    products = data["products"]
    forecasts = data["forecasts"]
    comparison = data["comparison"]
    metrics = data["metrics"]

    if opportunities is None or forecasts is None:
        st.error("Please run notebooks/07_forecasting.ipynb first to generate forecast outputs.")
        return

    non_hydrocarbon_default = False
    if "forecast_scope" in opportunities.columns:
        non_hydrocarbon_default = opportunities["forecast_scope"].astype(str).str.contains("non_hydrocarbon", case=False, na=False).any()
    if "is_hydrocarbon" in opportunities.columns:
        hydro = opportunities["is_hydrocarbon"].astype(str).str.lower().isin(["true", "1", "yes"])
        non_hydrocarbon_default = non_hydrocarbon_default or not hydro.any()
    if non_hydrocarbon_default:
        st.info("The current default view focuses on non-hydrocarbon diversification opportunities.")

    filtered_opps, filtered_forecasts = apply_sidebar_filters(opportunities, forecasts)

    executive_summary(filtered_opps, products, metrics)
    st.divider()
    opportunities_explorer(filtered_opps)
    st.divider()
    priority_markets(countries, filtered_opps)
    st.divider()
    priority_products(products, filtered_opps)
    st.divider()
    demand_and_trade_flows(filtered_forecasts, comparison, eda)
    st.divider()
    growth_and_potential(filtered_opps)
    st.divider()
    historical_vs_forecasted(filtered_forecasts, comparison, metrics)
    st.divider()
    model_performance(metrics)
    st.divider()
    historical_eda_context(eda)
    st.divider()
    sector_analysis(filtered_opps)
    st.divider()
    clustering_analysis(clustering)
    st.divider()
    classification_analysis(classification, filtered_opps)
    st.divider()
    interpretation_section(non_hydrocarbon_default)


if __name__ == "__main__":
    main()
