import pandas as pd
import polars as pl
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import re
from pathlib import Path

df = pd.read_csv(r"Exhibits\excess_crowding_vs_rent_growth.csv")
df = df.set_index(["met_name", "year"])
# -----------------------------
# Prepare panel
# -----------------------------
df = df.sort_index()

df["rent_next"] = df["real_relative_rent_growth_next_year"]
df["crowding_next"] = df.groupby(level=0)["excess_crowding"].shift(-1)
df["rent_lag"] = df.groupby(level=0)["real_relative_rent_growth_next_year"].shift(1)
print(df[["rent_lag", "real_relative_rent_growth_next_year"]])
df["crowding_lag"] = df["excess_crowding"]

# -----------------------------
# Final Granger-style dataframe
# -----------------------------
df_gc = df.dropna(
    subset=[
        "rent_next",  # ΔR_{t+1}
        "rent_lag",  # ΔR_t
        "crowding_lag",  # EC_t
        "crowding_next",  # EC_{t+1}
    ]
)
# Next-period crowding
df["crowding_next"] = df.groupby(level=0)["excess_crowding"].shift(-1)
df_gc = df.dropna(
    subset=[
        "real_relative_rent_growth_next_year",
        "excess_crowding",
        "rent_lag",
    ]
)

y_rent = df_gc["real_relative_rent_growth_next_year"]

X_rent = sm.add_constant(df_gc[["excess_crowding", "rent_lag"]])

rent_gc_model = PanelOLS(y_rent, X_rent, entity_effects=True, time_effects=True)

rent_gc_results = rent_gc_model.fit(cov_type="clustered", cluster_entity=True)

print("\nDynamic Ordering Test: Excess Crowding → Rent Growth")
print(rent_gc_results.summary)
df_gc2 = df.dropna(
    subset=[
        "crowding_next",
        "excess_crowding",
        "rent_lag",
    ]
)

y_crowd = df_gc2["crowding_next"]

X_crowd = sm.add_constant(df_gc2[["rent_lag", "excess_crowding"]])

crowd_gc_model = PanelOLS(y_crowd, X_crowd, entity_effects=True, time_effects=True)

crowd_gc_results = crowd_gc_model.fit(cov_type="clustered", cluster_entity=True)

print("\nDynamic Ordering Test: Rent Growth → Excess Crowding")
print(crowd_gc_results.summary)
