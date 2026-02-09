import matplotlib.pyplot as plt
import polars as pl
from preprocess import load_data
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import pandas as pd


df = (
    load_data()
    .select(
        "year",
        "met_name",
        "inventory_units",
        "supply_growth",
        "rent_growth",
        "real_relative_rent_growth_next_year",
        "population_owned",
        "population_rented",
        "households_owned",
        "households_rented",
        "density_owned",
        "density_rented",
        "density_rented_change",
        "bedroom_density_rented",
        "bedroom_density_owned",
        "age_under_18_share",
        "age_18_to_24_share",
        "age_25_to_34_share",
        "age_34_to_49_share",
        "age_50_and_over_share",
    )
    .to_pandas()
)
print(
    df[df["year"] == 2024]["population_rented"].sum(),
    df[df["year"] == 2024]["households_rented"].sum(),
)
print(
    df[df["density_rented"] == df["density_rented"].min()],
    df[df["density_rented"] == df["density_rented"].max()],
)
# create summary statistics table for numeric columns and save as LaTeX
numeric = df.select_dtypes(include=[np.number])

desc = numeric.describe().transpose()
desc = desc.rename(columns={"count": "N", "25%": "Q1", "50%": "Median", "75%": "Q3"})
desc = desc[["N", "mean", "std", "min", "Median", "max"]]
desc["N"] = desc["N"].astype(int)

out_path = os.path.join(r"Figs", "summary_stats.tex")
desc.index = [str(name).replace("_", r"\_") for name in desc.index]


def _fmt(x):
    if pd.isna(x):
        return ""
    try:
        v = float(x)
    except Exception:
        return str(x)
    # use magnitude (>=1000) for comma/no-decimal rule
    if abs(v) >= 1000:
        return f"{int(round(v)):,}"
    else:
        return f"{v:.2f}"


formatted = desc.copy()
formatted = formatted.applymap(_fmt)

latex = formatted.to_latex(
    escape=False, caption="Summary statistics", label="tab:summary_stats"
)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(latex)
