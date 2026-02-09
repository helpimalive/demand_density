import pandas as pd
import polars as pl
import numpy as np
import statsmodels.api as sm
import polars as pl
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from preprocess import load_data
import re
from pathlib import Path

print(load_data().columns)


def ols_fe():
    df = (
        load_data(100)
        .select(
            "year",
            "met_name",
            "occupancy",
            "inventory_units",
            "rent_growth",
            "density_rented",
            "bedroom_density_rented",
            "bedroom_density_owned",
            "age_under_18_share",
            "age_18_to_24_share",
            "age_25_to_34_share",
            "age_34_to_49_share",
            "age_50_and_over_share",
            "real_relative_rent_growth_next_year",
            "supply_growth",
            "density_rented_change",
            "real_rent_growth_next_year",
            "own_percent",
            "density_owned",
            "total_population",
            "population_growth",
            "sfh_share",
        )
        .with_columns(
            (
                pl.col("supply_growth") - pl.col("supply_growth").mean().over("year")
            ).alias("excess_supply_growth")
        )
        # .filter(pl.col("year") != 2020)
        # .filter(pl.col("year") != 2008)
        # .filter(~pl.col("met_name").str.contains("Orleans"))
        .to_pandas()
    )

    ## ORIGINAL REGRESSION
    df = df.set_index(["met_name", "year"])
    y = df["density_rented"]
    X = df[
        [
            "bedroom_density_rented",
            # "sfh_share",
            "age_under_18_share",
            "age_25_to_34_share",
            "age_34_to_49_share",
        ]
    ]

    model = PanelOLS(y, sm.add_constant(X), entity_effects=True, time_effects=True)
    results = model.fit(cov_type="clustered", cluster_entity=True)
    print(results.summary)
    ## RESIDUALIZED REGRESSION
    df["excess_crowding"] = results.resids
    # COMENT OUT FOR HETOGENEITY TEST
    # split_var = "bedroom_density_rented"
    # df = df[(df[split_var] > df[split_var].median())]
    rent_model = PanelOLS(
        df["real_relative_rent_growth_next_year"],
        sm.add_constant(df[["excess_crowding", "supply_growth"]]),
        entity_effects=True,
        time_effects=True,
    )
    rent_results = rent_model.fit(cov_type="clustered", cluster_entity=True)
    print(rent_results.summary)
    df["forecast_rent"] = rent_results.predict()
    df.reset_index().to_csv(r"Exhibits\excess_crowding_vs_rent_growth.csv", index=False)


ols_fe()
