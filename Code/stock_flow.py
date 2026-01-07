import pandas as pd
import polars as pl
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from preprocess import load_data

df = (
    load_data()
    .select(
        "year",
        "met_name",
        "inventory_units",
        "rent_growth",
        "density_rented",
        "bedroom_density_rented",
        "age_under_18_share",
        "age_18_to_24_share",
        "age_25_to_34_share",
        "age_34_to_49_share",
        "age_50_and_over_share",
        "rent_growth_next",
        "supply_growth",
        "density_rented_change",
    )
    .to_pandas()
    .dropna()
)


## ORIGINAL REGRESSION
df = df.set_index(["met_name", "year"])
y = df["rent_growth_next"]
X = df[
    [
        "density_rented",
        "density_rented_change",
        "bedroom_density_rented",
        "age_25_to_34_share",
        "age_34_to_49_share",
        "age_50_and_over_share",
        "supply_growth",
    ]
]

model = PanelOLS(y, sm.add_constant(X), entity_effects=True, time_effects=True)
results = model.fit(cov_type="clustered", cluster_entity=True)
print(results.summary)
regression_original_summary = results.summary.as_latex()
with open(r"Exhibits/regression_original_summary.tex", "w") as f:
    f.write(regression_original_summary)


## RESIDUALIZED REGRESSION
# Dependent variable: observed renter household density
y_rdi = df["density_rented"]

# Explanatory variables: composition + unit size
X_rdi = df[
    [
        "bedroom_density_rented",
        "age_25_to_34_share",
        "age_34_to_49_share",
        "age_50_and_over_share",
    ]
]

# Estimate expected RDI
rdi_model = PanelOLS(
    y_rdi,
    sm.add_constant(X_rdi),
    entity_effects=True,  # MSA fixed effects
    time_effects=True,  # year fixed effects
)

rdi_results = rdi_model.fit(cov_type="clustered", cluster_entity=True)
df["excess_crowding"] = rdi_results.resids
print(df["excess_crowding"].mean())
print(df[["excess_crowding", "density_rented"]].corr())
rent_model = PanelOLS(
    df["rent_growth_next"],
    sm.add_constant(df[["excess_crowding", "density_rented_change", "supply_growth"]]),
    entity_effects=True,
    time_effects=True,
)
ex = (
    df[["excess_crowding", "rent_growth_next"]]
    .reset_index()
    .to_csv(r"Exhibits/excess_crowding_vs_rent_growth.csv")
)
rent_results = rent_model.fit(cov_type="clustered", cluster_entity=True)
print(rent_results.summary)
regression_residualized_summary = rent_results.summary.as_latex()
with open(r"Exhibits/regression_residual_summary.tex", "w") as f:
    f.write(regression_residualized_summary)
