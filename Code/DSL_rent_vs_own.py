import pandas as pd
import polars as pl
from pathlib import Path
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import seaborn as sns

### Theory Anchoring ###
""" 
Rent growth is a function of demand for rental houses
Demographics tend to drive demad for rental housing & owner housing
One measure of demographics is # residents/household 
    at overall level (HDI)
    rental (RDI)
    and owner (ODI) levels
"""
### Variable Map ###
"""
Role    	Candidate Variables
Demand  	RDI, ODI, HDI; ratio of ODI to RDI
Outcomes	Δ rent
"""
### High Level ###
"""
Questions:
+ How does ODI/RDI relate to rent-growth and home price change in an MSA?
+ How does the RDI and ODI relate to ADI (in MSA or globally) in the long-run?
    Is HDI mean reverting?
    Does the ADI dictate where the RDI and ODI will go in the future?
    Do MSAs' ADI and RDI trend to the national means?
"""


def load_data():
    df = (
        pl.read_csv(
            Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
        )
        .with_columns(
            (pl.col("POPULATION_OWNED") / pl.col("HOUSEHOLDS_OWNED")).alias("ODI"),
            (
                (pl.col("POPULATION_OWNED") + pl.col("POPULATION_RENTED"))
                / (pl.col("HOUSEHOLDS_OWNED") + pl.col("HOUSEHOLDS_RENTED"))
            ).alias("ADI"),
        )
        .with_columns(
            (pl.col("POPULATION_OWNED") + pl.col("POPULATION_RENTED")).alias(
                "TOTAL_POP"
            )
        )
    )

    # Find top 100 MSAs by total population (using latest year in data)
    latest_year = df.select(pl.col("year")).max().item()
    top_100_msas = (
        df.filter(pl.col("year") == latest_year)
        .group_by("msa")
        .agg(pl.col("TOTAL_POP").max().alias("TOTAL_POP"))
        .sort("TOTAL_POP", descending=True)
        .head(200)
        .select("msa")
        .to_series()
        .to_list()
    )

    df = (
        df.filter(pl.col("msa").is_in(top_100_msas)).select(
            "year",
            "msa",
            "RDI",
            "ODI",
            "ADI",
            pl.col("rrg_5yr_fwd").alias("rrg"),
            pl.col("rrrg_5yr_fwd").alias("rrrg"),
        )
        # .drop_nulls()
    )
    return df


def converge():
    df = load_data()
    pivot_df = (
        df.filter(pl.col("year").is_in([2005, 2023]))
        .with_columns(
            [
                pl.when(pl.col("year") == 2005)
                .then(pl.col("RDI"))
                .otherwise(None)
                .alias("RDI_2005"),
                pl.when(pl.col("year") == 2005)
                .then(pl.col("ODI"))
                .otherwise(None)
                .alias("ODI_2005"),
                pl.when(pl.col("year") == 2005)
                .then(pl.col("ADI"))
                .otherwise(None)
                .alias("ADI_2005"),
                pl.when(pl.col("year") == 2023)
                .then(pl.col("ADI"))
                .otherwise(None)
                .alias("ADI_2023"),
                pl.when(pl.col("year") == 2023)
                .then(pl.col("RDI"))
                .otherwise(None)
                .alias("RDI_2023"),
                pl.when(pl.col("year") == 2023)
                .then(pl.col("ODI"))
                .otherwise(None)
                .alias("ODI_2023"),
            ]
        )
        .group_by("msa")
        .agg(
            [
                pl.col("RDI_2005").max(),
                pl.col("ODI_2005").max(),
                pl.col("ADI_2005").max(),
                pl.col("ADI_2023").max(),
                pl.col("RDI_2023").max(),
                pl.col("ODI_2023").max(),
            ]
        )
    )
    print(pivot_df.mean())
    pivot_df = (
        pivot_df.with_columns(
            abs(pl.col("RDI_2005") - pl.col("ODI_2005")).alias("RDI_2005_diff"),
            abs(pl.col("RDI_2023") - pl.col("ODI_2023")).alias("RDI_2023_diff"),
        )
        .with_columns(
            (pl.col("RDI_2023_diff") < pl.col("RDI_2005_diff")).alias("RDI_converge"),
            (pl.col("ADI_2023") < pl.col("ADI_2005")).alias("ADI_up"),
        )
        .select(
            # "msa",
            # "RDI_2005",
            # "ODI_2005",
            # "RDI_2023",
            "RDI_converge",
            "ADI_up",
        )
        .group_by(pl.col("ADI_up"))
        .agg(
            pl.col("RDI_converge").sum().alias("RDI_converge"),
            pl.col("RDI_converge").len().alias("RDI_converge_count"),
        )
        .with_columns(
            (pl.col("RDI_converge") / pl.col("RDI_converge_count")).alias(
                "pct_of_msas_that_converged"
            )
        )
    )
    print(pivot_df)


def median_converge():
    df = load_data()
    pivot_df = (
        df.filter(pl.col("year").is_in([2005, 2023]))
        .with_columns(
            [
                pl.when(pl.col("year") == 2005)
                .then(pl.col("RDI"))
                .otherwise(None)
                .alias("RDI_2005"),
                pl.when(pl.col("year") == 2005)
                .then(pl.col("ODI"))
                .otherwise(None)
                .alias("ODI_2005"),
                pl.when(pl.col("year") == 2005)
                .then(pl.col("ADI"))
                .otherwise(None)
                .alias("ADI_2005"),
                pl.when(pl.col("year") == 2023)
                .then(pl.col("ADI"))
                .otherwise(None)
                .alias("ADI_2023"),
                pl.when(pl.col("year") == 2023)
                .then(pl.col("RDI"))
                .otherwise(None)
                .alias("RDI_2023"),
                pl.when(pl.col("year") == 2023)
                .then(pl.col("ODI"))
                .otherwise(None)
                .alias("ODI_2023"),
            ]
        )
        .group_by("msa")
        .agg(
            [
                pl.col("RDI_2005").max(),
                pl.col("RDI_2023").max(),
                pl.col("ADI_2005").max(),
                pl.col("ADI_2023").max(),
                pl.col("ODI_2005").max(),
                pl.col("ODI_2023").max(),
            ]
        )
    )
    pivot_df = (
        pivot_df.with_columns(
            (pl.col("RDI_2023") - pl.col("RDI_2005")).alias("RDI_change"),
            (pl.col("ODI_2023") - pl.col("ODI_2005")).alias("ODI_change"),
            (pl.col("ADI_2023") - pl.col("ADI_2005")).alias("ADI_change"),
        )
        .group_by(
            (pl.col("ADI_2005") > pl.col("ADI_2005").median()).alias("ADI_gt_median")
        )
        .agg(
            pl.col("RDI_change").mean().alias("mean_RDI_change"),
            pl.col("ODI_change").mean().alias("mean_ODI_change"),
            pl.col("ADI_change").mean().alias("mean_ADI_change"),
        )
    ).sort("ADI_gt_median", descending=True)
    print(pivot_df)


def rent_growth_regression():
    df = load_data()
    df = df.with_columns(
        (pl.col("ODI") / pl.col("RDI")).alias("ODI_RDI_ratio"),
    )
    model = ols("rrg ~ ODI_RDI_ratio", data=df.to_pandas()).fit()
    print(model.summary())
    # Scatterplot with line of best fit and R^2

    df_pd = df.to_pandas()
    x = df_pd["ODI_RDI_ratio"]
    y = df_pd["rrg"]

    plt.figure(figsize=(8, 6))
    sns.regplot(x=x, y=y, ci=None, line_kws={"color": "red"})
    plt.xlabel("ODI / RDI Ratio")
    plt.ylabel("5yr Forward Real Rent Growth (rrg)")
    plt.title(f"Rent Growth vs ODI/RDI Ratio\n$R^2$ = {model.rsquared:.3f}")
    plt.tight_layout()
    plt.show()


rent_growth_regression()


# Does the ADI dictate where the RDI and ODI will go in the future?
# converge()
#   no
#   but RDI and ODI converge in the long run; 81 out of 100 MSAs show convergence;
#   the effect is more pronounced in larger msas; less pronounced in smaller
# Do MSAs' ADI and RDI trend to the national means?
#   no; all densities decrease
# Do MSAs with ADI above the national mean converge?
# median_converge()
#   they all go down, but those that were gt median go down faster
# Does the quotient of ODI/RDI explain rent growth?
# rent_growth_regression()
#   yes; it explains 11% of the variance in real rent growth (5yr fwd)
#   the lower the ODI/RDI ratio, the higher the rent growth
# Does ODI median vs RDI median explain rent growth?
