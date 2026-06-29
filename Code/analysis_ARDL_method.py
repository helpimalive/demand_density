import pandas as pd
import polars as pl
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from preprocess import load_data
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ardl import ardl_select_order



def panel_unit_root_test(series_by_entity, name):
    """Im-Pesaran-Shin-style panel unit root: average individual ADF t-stats."""
    from statsmodels.tsa.stattools import adfuller
    tstats = []
    for _, grp in series_by_entity:
        s = grp.dropna()
        if len(s) >= 5:
            try:
                result = adfuller(s, maxlag=1, autolag=None, regression='c')
                tstats.append(result[0])
            except Exception:
                pass
    avg_t = np.mean(tstats)
    print(f"\nPanel Unit Root (avg ADF t-stat) for {name}: {avg_t:.4f}")
    print(f"  N series={len(tstats)}. More negative => more evidence of stationarity.")
    print(f"  Individual ADF 5% critical value ≈ -2.98; avg well below => stationary.")
    return avg_t

def model_ardl():
    df = (
        load_data(100, cached=False).select(
            "year",
            "met_name",
            # "occupancy",
            # "real_relative_rent_growth_this_year",
            "real_rent_growth",
            "real_rent_growth_next_year",
            "density_rented",
            # "density_owned",
            # "total_bedroom_density",
            # "total_density_hh",
            "bedroom_density_rented",
            # "bedroom_density_owned",
            # "age_under_18_share",
            # "age_18_to_24_share",
            # "age_25_to_34_share",
            # "age_34_to_49_share",
            # "age_50_and_over_share",
            "real_relative_rent_growth_next_year",
            "supply_growth",
            # "own_percent",
            # "pct_owner_hhs_with_minor",
            # "pct_renter_hhs_with_minor",
            # "median_renter_hh_income",
            # "median_owner_hh_income",
            # "owner_to_renter_income_ratio",
            # "sfr_share",
        )
        .to_pandas()
    )

    df = df.set_index(["met_name", "year"])

    # ── Panel unit root tests ──────────────────────────────────────────────────
    grouped_density = df.groupby(level="met_name")["density_rented"]
    grouped_rent    = df.groupby(level="met_name")["real_rent_growth_next_year"]
    panel_unit_root_test(grouped_density, "density_rented")
    panel_unit_root_test(grouped_rent,    "real_rent_growth_next_year")

    # ── Single-stage panel ARDL (reviewer's requested specification) ───────────
    # Δ R_{m,t+1} = α_m + γ_t
    #               + λ  · RDI_{m,t}          (lagged level  → ECM term)
    #               + δ  · Δ RDI_{m,t}         (short-run density change)
    #               + φ  · S_{m,t}             (lagged level of supply)
    #               + ψ  · Δ S_{m,t}           (short-run supply change)
    #               + ρ  · Δ R_{m,t}           (lagged rent growth)
    #               + ε

    df_ardl = df.copy()
    df_ardl = df_ardl.sort_index()

    # short-run changes (within entity)
    df_ardl["delta_rdi"]    = df_ardl.groupby(level="met_name")["density_rented"].diff()
    df_ardl["delta_supply"] = df_ardl.groupby(level="met_name")["supply_growth"].diff()

    # lagged levels (within entity)
    df_ardl["rdi_lag1"]    = df_ardl.groupby(level="met_name")["density_rented"].shift(1)
    df_ardl["supply_lag1"] = df_ardl.groupby(level="met_name")["supply_growth"].shift(1)

    ardl_vars = [
        "rdi_lag1",          # lagged level — this is the ECM coefficient
        "delta_rdi",         # short-run density change
        "supply_lag1",       # lagged level of supply
        "delta_supply",      # short-run supply change
        "real_rent_growth",  # lagged rent growth (Δ R_t)
        "bedroom_density_rented", # reviewer also wanted this control
    ]

    df_ardl_clean = df_ardl[["real_rent_growth_next_year"] + ardl_vars].dropna()
    # demean within entity and time first
    df_demeaned = df_ardl_clean - df_ardl_clean.groupby(level="met_name").transform("mean")
    df_demeaned = df_demeaned - df_demeaned.groupby(level="year").transform("mean")

    sel = ardl_select_order(
        df_demeaned["real_rent_growth_next_year"],
        maxorder=10,
        maxlag=5,
        exog=df_demeaned[["rdi_lag1", "supply_lag1"]],
        ic="bic",
        trend="n"
    )
    print(sel.model.ardl_order)
    # assert False

    ardl_model = PanelOLS(
        df_ardl_clean["real_rent_growth_next_year"],
        sm.add_constant(df_ardl_clean[ardl_vars]),
        entity_effects=True,
        time_effects=True,
    )
    ardl_results = ardl_model.fit(cov_type="clustered", cluster_entity=True)
    print("\n── Single-Stage Panel ARDL (reviewer specification) ──")
    print(ardl_results.summary)

    autocorr_ardl = acorr_ljungbox(ardl_results.resids, lags=4, return_df=True)
    print("\nAutocorrelation Test (Ljung-Box) - ARDL residuals:")
    print(autocorr_ardl)

    # economic magnitudes
    params = ardl_results.params
    print(f"\nECM coefficient on rdi_lag1: {params['rdi_lag1']:.4f}")
    print(f"  1-SD RDI level shock => "
          f"{df_ardl_clean['rdi_lag1'].std() * params['rdi_lag1'] * 10000:.1f} bps next-year rent")
    print(f"  1-SD Δ RDI shock     => "
          f"{df_ardl_clean['delta_rdi'].std() * params['delta_rdi'] * 10000:.1f} bps next-year rent")

    df_ardl_clean = df_ardl_clean.copy()
    df_ardl_clean["forecast_rent_ardl"] = ardl_results.predict()
    df_ardl_clean.reset_index().to_csv(
        r"Exhibits\ardl_rent_forecast.csv", index=False
    )

    # also save the original excess-crowding frame for descriptive exhibits
    df.reset_index().to_csv(
        r"Exhibits\excess_crowding_vs_rent_growth.csv", index=False
    )
model_ardl()