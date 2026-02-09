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


def placebo_check():

    # lead placebo: predict past rent growth with current RDI
    df = pl.read_csv(r"Exhibits\excess_crowding_vs_rent_growth.csv")
    df_lead = (
        df.sort(["met_name", "year"], descending=False)
        .with_columns(
            pl.col("rent_growth")
            .shift(-1)
            .over("met_name", order_by="year")
            .alias("lead_placebo")
        )
        .drop_nulls(["lead_placebo", "excess_crowding"])
    )

    # fixed effects via demeaning (within estimator)
    df_dm = df_lead.with_columns(
        (
            pl.col("lead_placebo")
            - pl.col("lead_placebo").mean().over("met_name")
            - pl.col("lead_placebo").mean().over("year")
            + pl.col("lead_placebo").mean()
        ).alias("y_dm"),
        (
            pl.col("excess_crowding")
            - pl.col("excess_crowding").mean().over("met_name")
            - pl.col("excess_crowding").mean().over("year")
            + pl.col("excess_crowding").mean()
        ).alias("x_dm"),
    )
    y = df_dm["y_dm"].to_numpy()
    X = df_dm["x_dm"].to_numpy().reshape(-1, 1)

    model = sm.OLS(y, X)
    results = model.fit(
        cov_type="cluster", cov_kwds={"groups": df_dm["met_name"].to_numpy()}
    )

    print(results.summary())
    placebo_results = results.summary().as_latex()
    lines = placebo_results.splitlines()
    placebo_results = "\n".join(lines[:-4])
    # with open(r"Figs/placebo_regression_summary.tex", "w") as f:
    #     f.write(placebo_results)


def placebo_check_random():
    df = pl.read_csv(r"Exhibits\excess_crowding_vs_rent_growth.csv")
    coefs = []
    for _ in range(1000):
        df_rand = df.with_columns(
            pl.col("excess_crowding")
            .shuffle()
            .over("year")
            .alias("excess_crowding_rand")
        ).drop_nulls(["real_relative_rent_growth_next_year", "excess_crowding_rand"])

        df_dm = df_rand.with_columns(
            (
                pl.col("real_relative_rent_growth_next_year")
                - pl.col("real_relative_rent_growth_next_year").mean().over("met_name")
                - pl.col("real_relative_rent_growth_next_year").mean().over("year")
                + pl.col("real_relative_rent_growth_next_year").mean()
            ).alias("y_dm"),
            (
                pl.col("excess_crowding_rand")
                - pl.col("excess_crowding_rand").mean().over("met_name")
                - pl.col("excess_crowding_rand").mean().over("year")
                + pl.col("excess_crowding_rand").mean()
            ).alias("x_dm"),
        )

        y = df_dm["y_dm"].to_numpy()
        X = df_dm["x_dm"].to_numpy().reshape(-1, 1)

        model = sm.OLS(y, X)
        res = model.fit(
            cov_type="cluster", cov_kwds={"groups": df_dm["met_name"].to_numpy()}
        )

        coefs.append(res.params[0])

    coefs = np.array(coefs)
    print(res.summary())

    print("Mean placebo coef:", coefs.mean())
    print("Std placebo coef:", coefs.std())
    print("2.5 / 97.5 percentiles:", np.percentile(coefs, [2.5, 97.5]))


def placebo_check_linear_trend(exclude_covid=False):
    # load data
    df = pl.read_csv(r"Exhibits\excess_crowding_vs_rent_growth.csv")
    if exclude_covid:
        df = df.filter(~pl.col("year").is_in([2020]))

    # function to residualize a variable on a metro-specific linear time trend
    def residualize_on_trend(df, yvar):
        out = []

        for met, g in df.group_by("met_name"):
            t = g["year"].to_numpy()
            X = np.column_stack([np.ones(len(t)), t])
            y = g[yvar].to_numpy()

            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta

            out.append(g.with_columns(pl.Series(f"{yvar}_resid", resid)))

        return pl.concat(out)

    # residualize outcome and regressor on metro-specific trends
    df_tr = residualize_on_trend(df, "real_relative_rent_growth_next_year")
    df_tr = residualize_on_trend(df_tr, "excess_crowding")

    # two-way demeaning (MSA + year FE)
    df_dm = df_tr.with_columns(
        (
            pl.col("real_relative_rent_growth_next_year_resid")
            - pl.col("real_relative_rent_growth_next_year_resid")
            .mean()
            .over("met_name")
            - pl.col("real_relative_rent_growth_next_year_resid").mean().over("year")
            + pl.col("real_relative_rent_growth_next_year_resid").mean()
        ).alias("y_dm"),
        (
            pl.col("excess_crowding_resid")
            - pl.col("excess_crowding_resid").mean().over("met_name")
            - pl.col("excess_crowding_resid").mean().over("year")
            + pl.col("excess_crowding_resid").mean()
        ).alias("x_dm"),
    ).drop_nulls(["y_dm", "x_dm"])

    y = df_dm["y_dm"].to_numpy()
    X = df_dm["x_dm"].to_numpy().reshape(-1, 1)

    model = sm.OLS(y, X)
    results = model.fit(
        cov_type="cluster", cov_kwds={"groups": df_dm["met_name"].to_numpy()}
    )
    print(results.summary())
    lines = results.summary().as_latex().splitlines()
    results = "\n".join(lines[:-4])
    if exclude_covid:
        with open(r"Figs/placebo_check_linear_trend_no_covid.tex", "w") as f:
            f.write(results)
    else:
        with open(r"Figs/placebo_check_linear_trend.tex", "w") as f:
            f.write(results)


# placebo_check()
# placebo_check_random()
placebo_check_linear_trend(exclude_covid=False)
# placebo_check_linear_trend(exclude_covid=True)
