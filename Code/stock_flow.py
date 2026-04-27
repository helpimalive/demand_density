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


def ols_fe():
    df = (
        load_data(100, cached=False).select(
            "year",
            "met_name",
            "occupancy",
            "real_relative_rent_growth_this_year",
            "density_rented",
            "density_owned",
            "total_bedroom_density",
            "total_density_hh",
            "bedroom_density_rented",
            "bedroom_density_owned",
            "age_under_18_share",
            "age_18_to_24_share",
            "age_25_to_34_share",
            "age_34_to_49_share",
            "age_50_and_over_share",
            "real_relative_rent_growth_next_year",
            "supply_growth",
            "own_percent",
            "pct_owner_hhs_with_minor",
            "pct_renter_hhs_with_minor",
            "median_renter_hh_income",
            "median_owner_hh_income",
            "owner_to_renter_income_ratio",
            "sfr_share",
        )
        # .with_columns(
        #     pl.col("real_relative_rent_growth_next_year") ** (1 / 3),
        #     pl.col("real_relative_rent_growth_this_year") ** (1 / 3),
        # )
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
            "bedroom_density_owned",
            "own_percent",
            "owner_to_renter_income_ratio",
            "pct_renter_hhs_with_minor",
            "pct_owner_hhs_with_minor",
            "age_under_18_share",
            "age_18_to_24_share",
            "age_25_to_34_share",
        ]
    ]

    model = PanelOLS(y, sm.add_constant(X), entity_effects=True, time_effects=True)
    results = model.fit(cov_type="clustered", cluster_entity=True)
    print(results.summary)

    ## RESIDUALIZED REGRESSION
    df["excess_crowding"] = results.resids

    rent_model = PanelOLS(
        df["real_relative_rent_growth_next_year"],
        sm.add_constant(
            df[
                [
                    "excess_crowding",
                    "real_relative_rent_growth_this_year",
                    "supply_growth",
                ]
            ]
        ),
        entity_effects=True,
        time_effects=True,
    )
    rent_results = rent_model.fit(cov_type="clustered", cluster_entity=True)
    print(rent_results.summary)

    print(df["supply_growth"].std() * 0.13)
    print(df["excess_crowding"].std() * 0.024)

    df["forecast_rent"] = rent_results.predict()
    df.reset_index().to_csv(r"Exhibits\excess_crowding_vs_rent_growth.csv", index=False)


ols_fe()
