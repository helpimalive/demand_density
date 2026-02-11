import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from preprocess import load_data
from matplotlib.ticker import MaxNLocator
from statsmodels.stats.weightstats import ttost_ind


def graph_spread_between_excess_crowding_and_rent_growth():
    df = load_data().select(
        "year",
        "met_name",
        "density_rented_change",
        "density_rented",
        "bedroom_density_rented",
        "real_rent_growth_next_year",
        "real_relative_rent_growth_next_year",
        "real_relative_rent_growth_this_year",
    )
    yvar = "real_relative_rent_growth_next_year"
    df = (
        df.with_columns(
            (
                pl.col("density_rented").median()
                # .over('year')
            ).alias("density_rented_median")
        )
        .with_columns(
            pl.when(pl.col("density_rented") >= pl.col("density_rented_median"))
            .then(pl.lit("High Density"))
            .otherwise(pl.lit("Low Density"))
            .alias("density_rented_group")
        )
        .with_columns(
            pl.col("density_rented_change").median()
            # .over("year")
            .alias("density_rented_change_median")
        )
        .with_columns(
            pl.when(
                pl.col("density_rented_change") < pl.col("density_rented_change_median")
            )
            .then(pl.lit("Densifying"))
            .otherwise(pl.lit("De-densifying"))
            .alias("density_rented_change_group"),
        )
    )
    print(
        df.group_by(["density_rented_group", "density_rented_change_group"])
        .agg(
            pl.col(yvar).mean().alias("rent_var") * 10000,
            pl.col(yvar).count(),
            pl.col(yvar).std().alias("std_dev") * 10000,
        )
        .sort("density_rented_group")
    )


graph_spread_between_excess_crowding_and_rent_growth()
