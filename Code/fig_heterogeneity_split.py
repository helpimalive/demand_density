
import pandas as pd
import polars as pl
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from preprocess import load_data
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ardl import ardl_select_order
def run_heterogeneity_split_annual():
    df = (
        load_data(100, cached=False).select(
            "year",
            "met_name",
            "real_rent_growth",
            "real_rent_growth_next_year",
            "density_rented",
            "total_bedroom_density",
            "supply_growth",
        )
        .to_pandas()
        .set_index(["met_name", "year"])
        .sort_index()
    )

    df["delta_rdi"]    = df.groupby(level="met_name")["density_rented"].diff()
    df["delta_supply"] = df.groupby(level="met_name")["supply_growth"].diff()
    df["rdi_lag1"]     = df.groupby(level="met_name")["density_rented"].shift(1)
    df["supply_lag1"]  = df.groupby(level="met_name")["supply_growth"].shift(1)

    ardl_vars = [
        "rdi_lag1",
        "delta_rdi",
        "supply_lag1",
        "delta_supply",
        "real_rent_growth",
        "total_bedroom_density",
    ]

    df_clean = df[["real_rent_growth_next_year"] + ardl_vars].dropna()

    # ── Annual median RDI split ────────────────────────────────────────────────
    # compute median rdi_lag1 across metros within each year
    baseline_rdi = (
        df_clean
        .reset_index()
        .sort_values("year")
        .groupby("met_name")["total_bedroom_density"]
        .first()
    )
    median_rdi = baseline_rdi.median()
    print(f"Median baseline RDI: {median_rdi:.4f}")

    high_density_metros = baseline_rdi[baseline_rdi >= median_rdi].index
    low_density_metros  = baseline_rdi[baseline_rdi <  median_rdi].index

    df_high = df_clean[df_clean.index.get_level_values("met_name").isin(high_density_metros)]
    df_low  = df_clean[df_clean.index.get_level_values("met_name").isin(low_density_metros)]

    print(f"High-density obs: {len(df_high)}")
    print(f"Low-density obs:  {len(df_low)}")

    for label, df_sub in [("High RDI (above annual median)", df_high),
                           ("Low RDI (below annual median)",  df_low)]:
        model = PanelOLS(
            df_sub["real_rent_growth_next_year"],
            sm.add_constant(df_sub[ardl_vars]),
            entity_effects=True,
            time_effects=True,
        ).fit(cov_type="clustered", cluster_entity=True)
        print(f"\n── {label} ──")
        print(model.summary)


run_heterogeneity_split_annual()