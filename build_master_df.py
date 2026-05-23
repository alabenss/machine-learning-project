# =============================================================================
# build_master_df.py  —  CORRECTED VERSION v2
# Full pipeline: Raw BACI → Master DataFrame with all trade features
#
# GRAIN: one row per (year t, HS6 product k, partner country j)
#        for ALL combinations — including zero-export pairs.
#
# KEY DESIGN DECISIONS (see individual sections for detail):
#   • Full grid is the skeleton — df_algeria is just one input to it
#   • ISO3 merged ONCE at the top — never re-merged later
#   • RCA uses export-mirror data with explicit documentation
#   • Label is percentile-based — stable across data vintages
#   • Lags computed on log1p scale — reduces skew from large outliers
#   • Three-way temporal split: train / val / test
#   • Two feature tiers: SIMPLE (5 features) and ADVANCED (full)
#
# Sources:
#   1. BACI     → bilateral trade flows + calculated features
#   2. GeoDist  → distance, language, border, colonial ties
#   3. World Bank → GDP, population, trade openness (wbdata API)
#   4. UNCTAD   → country-level export concentration/diversification
# =============================================================================

import numpy as np
import pandas as pd
import wbdata
from pathlib import Path

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

BACI_DIR      = Path("data/baci")
BACI_VERSION  = "202601"
HS_VERSION    = "HS12"
GEODIST_PATH  = Path("data/geodist/dist_cepii.csv")
UNCTAD_PATH   = Path("data/unctad/unctad_diversification.csv")
OUTPUT_PATH   = Path("data/master_df.parquet")

ALG_CODE      = 12
ALG_ISO3      = "DZA"
YEARS         = list(range(2012, 2024))

# Grid size control — only keep products whose peak annual world import
# value (across all years) exceeds this threshold (thousand USD = $1M).
# Raising this shrinks the grid and speeds up the pipeline.
# At $1M: typically ~3,000–4,000 active HS6 products out of ~5,000.
MIN_WORLD_DEMAND_USD = 1_000

# Temporal split boundaries
TRAIN_END = 2019   # train:  2012 – 2019
VAL_END   = 2021   # val:    2020 – 2021
#                    test:   2022 – 2023

# =============================================================================
# 1. LOAD BACI
# =============================================================================

print("── Step 1: Loading BACI ──")

frames = []
for year in YEARS:
    path = BACI_DIR / f"BACI_{HS_VERSION}_Y{year}_V{BACI_VERSION}.csv"
    if path.exists():
        df = pd.read_csv(path, dtype={'k': str})
        frames.append(df)
        print(f"  Loaded {year}: {len(df):,} rows")
    else:
        print(f"  WARNING: {path} not found — skipping")

baci = pd.concat(frames, ignore_index=True)
print(f"  Total BACI rows: {len(baci):,}")

# Lookup tables
country_codes = pd.read_csv(
    BACI_DIR / f"country_codes_V{BACI_VERSION}.csv",
    dtype={'country_code': int}
)
product_codes = pd.read_csv(
    BACI_DIR / f"product_codes_{HS_VERSION}_V{BACI_VERSION}.csv",
    dtype={'code': str}
)

# Numeric code → ISO3 crosswalk (used ONCE in Step 2, then carried forward)
crosswalk = country_codes[[
    'country_code', 'iso_3digit_alpha', 'country_name_abbreviation'
]].copy()
crosswalk.columns = ['country_code', 'iso3', 'country_name']


# =============================================================================
# 2. ATTACH ISO3 TO BACI PARTNER CODES — DONE ONCE HERE, NEVER AGAIN
# =============================================================================
# FIX vs v1: the original code merged the crosswalk in Step 5 (GeoDist)
# and again implicitly in Step 10 (labels), risking duplicate columns and
# misalignment. We resolve the j → iso3 mapping a single time here and
# carry iso3 through every subsequent merge.

print("\n── Step 2: Attaching ISO3 to all partner codes ──")

# Map every j code that appears in BACI to its ISO3
j_to_iso3 = crosswalk[['country_code', 'iso3', 'country_name']].rename(
    columns={'country_code': 'j'}
)


# =============================================================================
# 3. BUILD COMPONENT DATAFRAMES FROM BACI
# =============================================================================

print("\n── Step 3: Building component dataframes ──")

# 3a. Algeria's actual bilateral exports — (t, j, k) only where flow > 0
df_algeria = (
    baci[baci['i'] == ALG_CODE]
    .groupby(['t', 'j', 'k'], as_index=False)
    .agg(alg_export_v=('v', 'sum'), alg_export_q=('q', 'sum'))
)
print(f"  df_algeria:          {len(df_algeria):,} rows")

# 3b. Each partner's total imports per product from ALL sources — (t, j, k)
df_partner_imports = (
    baci.groupby(['t', 'j', 'k'], as_index=False)
    .agg(partner_import_v=('v', 'sum'), partner_import_q=('q', 'sum'))
)
print(f"  df_partner_imports:  {len(df_partner_imports):,} rows")

# 3c. Global demand per product — (t, k)
df_world_imports = (
    baci.groupby(['t', 'k'], as_index=False)
    .agg(world_import_v=('v', 'sum'), world_import_q=('q', 'sum'))
)
print(f"  df_world_imports:    {len(df_world_imports):,} rows")

# 3d. What Algeria imports from each partner — (t, j, k)
# 'i' is the source of Algeria's imports; rename to 'j' to align on partner
df_algeria_imports = (
    baci[baci['j'] == ALG_CODE]
    .groupby(['t', 'i', 'k'], as_index=False)
    .agg(alg_import_from_i_v=('v', 'sum'), alg_import_from_i_q=('q', 'sum'))
    .rename(columns={'i': 'j'})
)
print(f"  df_algeria_imports:  {len(df_algeria_imports):,} rows")

# 3e. Algeria's total exports per product per year — (t, k) — for RCA
df_alg_total_by_product = (
    df_algeria.groupby(['t', 'k'], as_index=False)['alg_export_v'].sum()
    .rename(columns={'alg_export_v': 'alg_total_export_k'})
)

# 3f. Algeria's grand total exports per year — (t) — for RCA denominator
df_alg_grand_total = (
    df_algeria.groupby('t', as_index=False)['alg_export_v'].sum()
    .rename(columns={'alg_export_v': 'alg_grand_total'})
)

# 3g. World total exports per year — (t) — for RCA denominator
# NOTE on RCA data source (important for academic transparency):
#   Standard Balassa RCA uses world EXPORTS in the denominator.
#   BACI records bilateral trade as both an export (i) and an import (j).
#   Summing world_import_v across all j gives us the total world trade
#   value of each product, which is identical to summing world exports
#   (every export is recorded as an import by the destination).
#   This is NOT an approximation — it is exact by construction in BACI.
#   We document this explicitly to pre-empt examiner questions.
df_world_grand_total = (
    df_world_imports.groupby('t', as_index=False)['world_import_v'].sum()
    .rename(columns={'world_import_v': 'world_grand_total'})
)


# =============================================================================
# 4. BUILD THE FULL OPPORTUNITY GRID
# =============================================================================
# The CRITICAL fix from v1: the grid, not df_algeria, is the master skeleton.
#
# Why this matters:
#   A row where alg_export_v = 0 and partner_import_v is large is a MISSED
#   OPPORTUNITY — exactly what the model should learn to detect.
#   Starting from df_algeria silently discards these rows.
#
# Memory management:
#   Raw grid = ~200 partners × ~5,000 products × 14 years ≈ 14M rows.
#   After the MIN_WORLD_DEMAND_USD filter (applied below), this typically
#   falls to ~3–5M rows — manageable in 8–16 GB RAM.
#   If memory is still an issue, raise MIN_WORLD_DEMAND_USD or switch to
#   Polars / DuckDB (drop-in replacements for the groupby/merge logic).

print("\n── Step 4: Building full opportunity grid ──")

# Filter products: keep only those with meaningful world demand in at least one year
active_products = (
    df_world_imports.groupby('k')['world_import_v'].max()
)
active_products = set(active_products[active_products >= MIN_WORLD_DEMAND_USD].index)
all_partners    = baci['j'].unique()

print(f"  Products before filter: {baci['k'].nunique():,}")
print(f"  Products after  filter: {len(active_products):,}  (world demand ≥ ${MIN_WORLD_DEMAND_USD:,}k)")
print(f"  Partners:               {len(all_partners):,}")
print(f"  Years:                  {len(YEARS)}")
estimated_rows = len(all_partners) * len(active_products) * len(YEARS)
print(f"  Estimated grid rows:    {estimated_rows:,}")

full_grid = pd.MultiIndex.from_product(
    [YEARS, all_partners, sorted(active_products)],
    names=['t', 'j', 'k']
).to_frame(index=False)

# Attach ISO3 and country name ONCE — all later merges use iso3
full_grid = full_grid.merge(j_to_iso3, on='j', how='left')

# Left-join Algeria's actual exports — missing = 0 (no current flow)
master = full_grid.merge(df_algeria,         on=['t', 'j', 'k'], how='left')
master = master.merge(df_partner_imports,    on=['t', 'j', 'k'], how='left')
master = master.merge(df_world_imports,      on=['t', 'k'],       how='left')
master = master.merge(df_algeria_imports,    on=['t', 'j', 'k'],  how='left')

# Fill all value columns: 0 means "no recorded trade", not "missing data"
fill_zero_cols = [
    'alg_export_v', 'alg_export_q',
    'alg_import_from_i_v', 'alg_import_from_i_q',
    'partner_import_v', 'partner_import_q',
    'world_import_v', 'world_import_q',
]
master[fill_zero_cols] = master[fill_zero_cols].fillna(0)

print(f"  master after merges: {len(master):,} rows × {master.shape[1]} cols")
print(f"  Zero-export rows (opportunities): {(master['alg_export_v'] == 0).sum():,}")
print(f"  Positive-export rows (existing):  {(master['alg_export_v'] > 0).sum():,}")


# =============================================================================
# 5. COMPUTED FEATURES FROM BACI
# =============================================================================

print("\n── Step 5: Computing BACI features ──")

master = master.sort_values(['j', 'k', 't']).reset_index(drop=True)

# ── 5a. Export growth rate ────────────────────────────────────────────────────
# Because the full grid already has zeros for all (j, k, t) combinations,
# pct_change() correctly handles:
#   0 → 0        : NaN (no meaningful change in zero)
#   0 → positive : inf → replaced with NaN (entry events — flagged separately)
#   positive → 0 : -1.0 (exit events)
#   positive → positive : normal growth

master['alg_export_growth'] = (
    master.groupby(['j', 'k'])['alg_export_v']
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
)

# Binary flag for new market entry: was zero last year, positive this year
master['is_new_entry'] = (
    (master.groupby(['j', 'k'])['alg_export_v'].shift(1) == 0) &
    (master['alg_export_v'] > 0)
).astype(int)

# ── 5b. World demand growth ───────────────────────────────────────────────────
# FIX vs v1: fillna(0) after merge to prevent NaN propagation into the model.
# A missing world_demand_growth means the product had no prior year data —
# treat as 0 (flat demand) rather than letting the NaN silently bias tree splits.

world_growth = df_world_imports.sort_values(['k', 't']).copy()
world_growth['world_demand_growth'] = (
    world_growth.groupby('k')['world_import_v']
    .pct_change()
    .replace([np.inf, -np.inf], np.nan)
)
master = master.merge(
    world_growth[['t', 'k', 'world_demand_growth']],
    on=['t', 'k'], how='left'
)
# First year of each product has no prior year → fill with 0
master['world_demand_growth'] = master['world_demand_growth'].fillna(0)

# ── 5c. Global demand index ───────────────────────────────────────────────────
master['global_demand_log'] = np.log1p(master['world_import_v'])
master['global_demand_rank'] = (
    master.groupby('t')['world_import_v'].rank(pct=True)
)

# ── 5d. Market penetration ────────────────────────────────────────────────────
# With the full grid, zero-export pairs correctly show 0 (not NaN).
master['market_penetration'] = (
    master['alg_export_v']
    / master['partner_import_v'].replace(0, np.nan)
).fillna(0).clip(upper=1.0)

n_clipped = (
    (master['alg_export_v'] / master['partner_import_v'].replace(0, np.nan)) > 1.0
).sum()
if n_clipped > 0:
    print(f"  WARNING: market_penetration clipped on {n_clipped} rows — check BACI rounding")

# ── 5e. Bilateral trade balance ───────────────────────────────────────────────
master['trade_balance_bilateral'] = (
    master['alg_export_v'] - master['alg_import_from_i_v']
)
master['trade_coverage_ratio'] = (
    master['alg_export_v']
    / master['alg_import_from_i_v'].replace(0, np.nan)
)

# ── 5f. Export destination count ─────────────────────────────────────────────
n_dest = (
    df_algeria[df_algeria['alg_export_v'] > 0]
    .groupby(['t', 'k'])['j'].nunique()
    .reset_index(name='n_export_destinations')
)
master = master.merge(n_dest, on=['t', 'k'], how='left')
master['n_export_destinations'] = master['n_export_destinations'].fillna(0)

# ── 5g. RCA — Revealed Comparative Advantage (Balassa Index) ─────────────────
# Formula: RCA_k = (X_alg_k / X_alg_total) / (X_world_k / X_world_total)
#
# Data source note (important for academic integrity):
#   Both Algeria's numerator and the world denominator use EXPORT values.
#   In BACI, the total of world_import_v for product k equals world exports
#   of product k by construction (every export is someone else's import).
#   So world_import_v IS world_export_v here — NOT an approximation.
#   This satisfies the standard Balassa formulation exactly.
#
# Interpretation:
#   RCA > 1 → Algeria has comparative advantage in product k
#   RCA < 1 → Algeria is relatively weak in product k
#   RCA is (t, k) level — not partner-specific

master = master.merge(df_alg_total_by_product, on=['t', 'k'], how='left')
master = master.merge(df_alg_grand_total,       on='t',        how='left')
master = master.merge(df_world_grand_total,     on='t',        how='left')

master['rca'] = (
    (master['alg_total_export_k']  / master['alg_grand_total'].replace(0, np.nan))
    /
    (master['world_import_v']      / master['world_grand_total'].replace(0, np.nan))
).replace([np.inf, -np.inf], np.nan)

# Drop intermediate columns used only for RCA calculation
master.drop(
    columns=['alg_total_export_k', 'alg_grand_total', 'world_grand_total'],
    inplace=True
)

print(f"  Features computed: alg_export_growth, is_new_entry, world_demand_growth,")
print(f"    global_demand_log, global_demand_rank, market_penetration,")
print(f"    trade_balance_bilateral, trade_coverage_ratio,")
print(f"    n_export_destinations, rca")


# =============================================================================
# 6. LAG FEATURES (computed on log scale)
# =============================================================================
# FIX vs v1: lags are computed on log1p-transformed values, not raw values.
#
# Why log1p:
#   Raw export values span many orders of magnitude ($1k to $1B+).
#   A lag of raw alg_export_v gives a very skewed feature — a few huge
#   values dominate gradients in tree-based models and distance metrics.
#   log1p compresses the scale and makes the growth signal more symmetric.
#   This is equivalent to how economists model trade in gravity models.
#
# Note: zero stays zero under log1p (log1p(0) = 0), so sustained
# non-exporters are still represented as a coherent flat signal.
# The entry event (0 → positive) becomes a jump from 0 to log1p(value),
# which is informative and correctly scaled.

print("\n── Step 6: Computing lag features (log1p scale) ──")

LAG_COLS = ['alg_export_v', 'market_penetration', 'world_import_v', 'rca']

# First, create log1p versions of the columns to lag
for col in LAG_COLS:
    master[f'{col}_log'] = np.log1p(master[col].fillna(0))

# Compute t-1 and t-2 lags within each (partner, product) group
for col in LAG_COLS:
    log_col = f'{col}_log'
    for lag in [1, 2]:
        master[f'{col}_lag{lag}'] = (
            master.groupby(['j', 'k'])[log_col].shift(lag)
        )

# Drop the intermediate log columns — the lags are what we keep
master.drop(columns=[f'{col}_log' for col in LAG_COLS], inplace=True)

print(f"  Lag features: t-1 and t-2 for {LAG_COLS} (on log1p scale)")


# =============================================================================
# 7. DEFINE THE TARGET LABEL (percentile-based)
# =============================================================================
# FIX vs v1: replaced hard absolute thresholds with percentile-based thresholds.
#
# Why percentile-based is better:
#   Absolute thresholds ($500k, $10M) are fragile:
#     • They depend on the unit of measurement (BACI uses thousand USD)
#     • They become miscalibrated as trade volumes change over years
#     • They produce very different class balances for different datasets
#
#   Percentile thresholds are self-calibrating:
#     • Top 20% partner demand = large importers of that product, this year
#     • Bottom 20% Algeria presence = Algeria barely penetrated this market
#   The intersection identifies structurally attractive, underexploited markets.
#
# Two conditions define HIGH OPPORTUNITY:
#   1. Algeria's market penetration is in the bottom LOW_PENETRATION_PCT
#      (Algeria barely present → room to grow)
#   2. Partner imports of this product are in the top HIGH_DEMAND_PCT
#      (real, large demand exists)
#
# Optional third condition (commented out): RCA >= 1
#   Adding RCA as a hard filter reduces false positives but also misses
#   products Algeria could develop. Leave it as a feature, not a filter.

print("\n── Step 7: Defining target label (percentile-based) ──")

LOW_PENETRATION_PCT  = 0.20   # bottom 20% of market_penetration within (t, k)
HIGH_DEMAND_PCT      = 0.80   # top 20% of partner_import_v within (t, k)

# Compute per (t, k) percentile thresholds — year and product aware
penetration_thresh = (
    master.groupby(['t', 'k'])['market_penetration']
    .transform(lambda x: x.quantile(LOW_PENETRATION_PCT))
)
demand_thresh = (
    master.groupby(['t', 'k'])['partner_import_v']
    .transform(lambda x: x.quantile(HIGH_DEMAND_PCT))
)

master['label_opportunity'] = (
    (master['market_penetration'] <= penetration_thresh) &
    (master['partner_import_v']   >= demand_thresh)
    # Optional: & (master['rca'] >= 1)
).astype(int)

class_dist = master['label_opportunity'].value_counts(normalize=True)
print(f"  Label distribution:")
print(f"    Opportunity (1):    {class_dist.get(1, 0)*100:.1f}%")
print(f"    No opportunity (0): {class_dist.get(0, 0)*100:.1f}%")

# Target: roughly 15–25% positives for a usable classifier.
# If outside this range, adjust LOW_PENETRATION_PCT / HIGH_DEMAND_PCT.
pos_rate = class_dist.get(1, 0)
if pos_rate < 0.10:
    print(f"  WARNING: Only {pos_rate*100:.1f}% positives — consider raising thresholds")
elif pos_rate > 0.40:
    print(f"  WARNING: {pos_rate*100:.1f}% positives — consider tightening thresholds")


# =============================================================================
# 8. LOAD & MERGE GEODIST (CEPII)
# =============================================================================
# iso3 is already on master from Step 4 — no second crosswalk merge needed.

print("\n── Step 8: Merging GeoDist ──")

if GEODIST_PATH.exists():
    geodist = pd.read_csv(GEODIST_PATH, encoding='latin-1')
    geodist_alg = geodist[geodist['iso_o'] == ALG_ISO3][[
        'iso_d', 'dist', 'distw', 'contig',
        'comlang_off', 'comlang_ethno', 'colony', 'comcol', 'smctry'
    ]].copy()
    geodist_alg.columns = [
        'iso3', 'dist_km', 'distw_km', 'contig',
        'comlang_off', 'comlang_ethno', 'colony', 'comcol', 'smctry'
    ]

    # iso3 already exists on master — merge directly, no crosswalk needed
    master = master.merge(geodist_alg, on='iso3', how='left')

    coverage = master['dist_km'].notna().mean() * 100
    print(f"  GeoDist merged — coverage: {coverage:.1f}%")

    # Flag unmatched iso3 codes if coverage is low
    if coverage < 80:
        unmatched = master[master['dist_km'].isna()]['iso3'].value_counts().head(10)
        print(f"  Top unmatched iso3 codes:\n{unmatched.to_string()}")
else:
    print(f"  WARNING: {GEODIST_PATH} not found — skipping GeoDist")


# =============================================================================
# 9. LOAD & MERGE WORLD BANK (via wbdata API)
# =============================================================================
# FIX vs v1: explicit rename() replaces positional column assignment.
# iso3 already on master — merge on ['t', 'iso3'] directly.

print("\n── Step 9: Fetching World Bank indicators ──")

WB_INDICATORS = {
    'NY.GDP.MKTP.CD': 'gdp_usd',
    'SP.POP.TOTL':    'population',
    'NE.TRD.GNFS.ZS': 'trade_pct_gdp',
    'NY.GDP.PCAP.CD': 'gdp_per_capita',
}

try:
    wb_raw = wbdata.get_dataframe(WB_INDICATORS, convert_date=False)
    wb_raw = wb_raw.reset_index()

    # FIX: explicit rename — never assume column position from reset_index()
    wb_raw = wb_raw.rename(columns={'country': 'iso3', 'date': 't'})
    wb_raw = wb_raw.rename(columns=WB_INDICATORS)

    wb_raw['t'] = wb_raw['t'].astype(int)
    wb_raw = wb_raw[wb_raw['t'].isin(YEARS)]

    master = master.merge(
        wb_raw[['t', 'iso3', 'gdp_usd', 'population', 'trade_pct_gdp', 'gdp_per_capita']],
        on=['t', 'iso3'], how='left'
    )

    master['market_size_usd'] = (
        master['gdp_usd'] * (master['trade_pct_gdp'] / 100)
    )

    pct_missing_gdp = master['gdp_usd'].isna().mean() * 100
    print(f"  World Bank merged — missing GDP: {pct_missing_gdp:.1f}%")
    if pct_missing_gdp > 20:
        unmatched_wb = master[master['gdp_usd'].isna()]['iso3'].value_counts().head(10)
        print(f"  Top unmatched iso3:\n{unmatched_wb.to_string()}")

except Exception as e:
    print(f"  WARNING: World Bank fetch failed ({e}) — skipping")


# =============================================================================
# 10. LOAD & MERGE UNCTAD DIVERSIFICATION INDEX
# =============================================================================
# These are Algeria-level, year-level features (HHI, diversification index).
# Broadcasting them to all rows for year t is correct ML practice —
# identical to how GDP is used. Document this clearly.

print("\n── Step 10: Merging UNCTAD diversification ──")

if UNCTAD_PATH.exists():
    unctad = pd.read_csv(UNCTAD_PATH)

    col_map = {
        'Economy': 'economy',
        'Year':    't',
        'Herfindahl-Hirschman Index, exports, normalized': 'hhi_export',
        'Herfindahl-Hirschman Index, imports, normalized': 'hhi_import',
        'Diversification index, exports': 'diversification_index',
    }
    unctad = unctad.rename(columns={k: v for k, v in col_map.items() if k in unctad.columns})

    if 'economy' in unctad.columns:
        unctad_alg = unctad[unctad['economy'].str.upper().str.contains('ALGERIA', na=False)]
    else:
        unctad_alg = unctad[unctad['iso3'] == ALG_ISO3]

    # FIX vs v1: hard validation — fail loudly instead of silently producing NaN
    expected_cols = {'t', 'hhi_export', 'diversification_index'}
    missing_cols  = expected_cols - set(unctad_alg.columns)
    if missing_cols:
        raise ValueError(
            f"UNCTAD file missing columns after rename: {missing_cols}\n"
            f"Actual columns: {list(unctad_alg.columns)}\n"
            f"Update col_map keys to match your downloaded file's headers."
        )

    unctad_alg = unctad_alg[['t', 'hhi_export', 'diversification_index']].copy()
    unctad_alg['t'] = unctad_alg['t'].astype(int)

    master = master.merge(unctad_alg, on='t', how='left')
    print(f"  UNCTAD merged — HHI coverage: {master['hhi_export'].notna().mean()*100:.1f}%")
    print(f"  (Algeria-level macro features, broadcast to all rows per year — correct by design)")
else:
    print(f"  WARNING: {UNCTAD_PATH} not found — skipping UNCTAD")


# =============================================================================
# 11. ADD PRODUCT DESCRIPTIONS
# =============================================================================

print("\n── Step 11: Adding product descriptions ──")

product_codes['description_short'] = product_codes['description'].str[:60]
master = master.merge(
    product_codes[['code', 'description_short']].rename(columns={'code': 'k'}),
    on='k', how='left'
)


# =============================================================================
# 12. TEMPORAL SPLIT — THREE-WAY
# =============================================================================
# FIX vs v1: added validation set between train and test.
#
# Why three-way split matters:
#   • Training set  → fit model parameters
#   • Validation set → tune hyperparameters, select features, choose thresholds
#   • Test set      → final unbiased evaluation (touch ONCE at the end)
#
#   Using test set for tuning inflates performance estimates — a common and
#   penalised mistake in academic ML projects.
#
# NEVER use random train_test_split on panel/time-series data.
# Rows share time periods — random split allows training on 2022 and testing
# on 2020, which is information leakage from the future.

print("\n── Step 12: Temporal split ──")

def assign_split(t):
    if t <= TRAIN_END:
        return 'train'
    elif t <= VAL_END:
        return 'val'
    else:
        return 'test'

master['split'] = master['t'].map(assign_split)

for s in ['train', 'val', 'test']:
    n = (master['split'] == s).sum()
    yrs = sorted(master.loc[master['split'] == s, 't'].unique())
    print(f"  {s:5s} ({yrs[0]}–{yrs[-1]}): {n:,} rows")


# =============================================================================
# 13. TWO FEATURE TIERS
# =============================================================================
# Simple model: 5 core features — interpretable, fast, good baseline
# Advanced model: full feature set — production quality

SIMPLE_FEATURES = [
    'market_penetration',    # how present is Algeria already?
    'global_demand_log',     # how large is world demand for this product?
    'rca',                   # does Algeria have comparative advantage?
    'dist_km',               # gravity: how far is the partner?
    'gdp_per_capita',        # proxy for partner's purchasing power / market type
]

ADVANCED_FEATURES = [
    # BACI features
    'alg_export_growth', 'world_demand_growth', 'global_demand_log',
    'global_demand_rank', 'market_penetration', 'trade_balance_bilateral',
    'trade_coverage_ratio', 'n_export_destinations', 'rca', 'is_new_entry',
    # Lag features (log1p scale)
    'alg_export_v_lag1', 'alg_export_v_lag2',
    'market_penetration_lag1', 'market_penetration_lag2',
    'world_import_v_lag1', 'world_import_v_lag2',
    'rca_lag1', 'rca_lag2',
    # GeoDist
    'dist_km', 'distw_km', 'contig', 'comlang_off',
    'comlang_ethno', 'colony', 'comcol', 'smctry',
    # World Bank
    'gdp_usd', 'population', 'trade_pct_gdp', 'gdp_per_capita', 'market_size_usd',
    # UNCTAD
    'hhi_export', 'diversification_index',
]

# Filter to only columns that actually exist (optional sources may be skipped)
ADVANCED_FEATURES = [f for f in ADVANCED_FEATURES if f in master.columns]
SIMPLE_FEATURES   = [f for f in SIMPLE_FEATURES   if f in master.columns]

print(f"\n── Step 13: Feature tiers ──")
print(f"  Simple   model features ({len(SIMPLE_FEATURES)}):   {SIMPLE_FEATURES}")
print(f"  Advanced model features ({len(ADVANCED_FEATURES)}): listed above")


# =============================================================================
# 14. FINAL COLUMN ORDERING
# =============================================================================

print("\n── Step 14: Final column ordering ──")

col_order = [
    # Identifiers
    't', 'j', 'k', 'iso3', 'country_name', 'description_short', 'split',
    # Target
    'label_opportunity',
    # Entry flag
    'is_new_entry',
    # Raw values
    'alg_export_v', 'alg_export_q',
    'alg_import_from_i_v', 'alg_import_from_i_q',
    'partner_import_v', 'partner_import_q',
    'world_import_v', 'world_import_q',
    # All features (simple + advanced, deduplicated)
    *dict.fromkeys(ADVANCED_FEATURES),
]

master = master[[c for c in col_order if c in master.columns]]

print(f"  Final master_df: {len(master):,} rows × {master.shape[1]} columns")
print(f"  Years:           {sorted(master['t'].unique())}")
print(f"  Unique partners: {master['j'].nunique()}")
print(f"  Unique products: {master['k'].nunique()}")


# =============================================================================
# 15. SANITY CHECKS
# =============================================================================

print("\n── Step 15: Sanity checks ──")

checks = {
    "market_penetration in [0, 1]":
        master['market_penetration'].dropna().between(0, 1).all(),

    "no inf in alg_export_growth":
        not master['alg_export_growth'].isin([np.inf, -np.inf]).any(),

    "no inf in rca":
        not master['rca'].isin([np.inf, -np.inf]).any(),

    "global_demand_rank in (0, 1]":
        master['global_demand_rank'].dropna().between(0, 1, inclusive='right').all(),

    "world_import_v >= partner_import_v":
        (master['world_import_v'] >= master['partner_import_v']).all(),

    "label_opportunity is binary":
        set(master['label_opportunity'].unique()).issubset({0, 1}),

    "split column is complete (train/val/test only)":
        set(master['split'].unique()).issubset({'train', 'val', 'test'}),

    "zero-export rows preserved":
        (master['alg_export_v'] == 0).any(),

    "lag1 NaN for earliest year":
        master.loc[master['t'] == min(YEARS), 'alg_export_v_lag1'].isna().all(),

    "world_demand_growth has no NaN (filled)":
        master['world_demand_growth'].isna().sum() == 0,
}

all_passed = True
for check, result in checks.items():
    status = "✓" if result else "✗ FAILED"
    print(f"  {status}  {check}")
    if not result:
        all_passed = False

print("\n  All checks passed." if all_passed else "\n  Some checks FAILED — review before modelling.")


# =============================================================================
# 16. SAVE
# =============================================================================

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
master.to_parquet(OUTPUT_PATH, index=False)
print(f"\n── Saved to {OUTPUT_PATH} ──")
print(master.dtypes.to_string())

# Opportunity landscape summary
print("\n── Opportunity summary (label = 1) ──")
opp = master[master['label_opportunity'] == 1]
print(f"  Total opportunity rows:          {len(opp):,}")
print(f"  Unique opportunity partners:     {opp['j'].nunique()}")
print(f"  Unique opportunity products:     {opp['k'].nunique()}")
top_partners = opp.groupby('country_name')['k'].nunique().sort_values(ascending=False).head(10)
print(f"  Top 10 partners by # products:\n{top_partners.to_string()}")