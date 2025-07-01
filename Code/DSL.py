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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

### Theory Anchoring ###
""" 
You rent growth is a function of excess demand
We can measure excess demand as a function of occupancy-adjusted renters per rental, adjusted for 
    supply growth?
    rent burden?
    in-migration?
"""
### Variable Map ###
"""
Role    	Candidate Variables
Demand  	Δ renter pop, Δ occupancy, Δ renters per rental, renters per rental level
Supply	    Δ new units, Δ completions, Δ rental stock
Frictions	Rent-to-income, supply, rent burden
Outcomes	Δ rent
"""


### High Level ###
""" Relationship between RDI and rent growth:
    |_____  higher real rent growth (abs and relative) in markets with higher excess demand, measured as RDI
            rent_vs_rdi()
    |_____  higher real rent growth (abs and relative) in markets with higher excess demand, measured as negative RDI growth
            note RDI_growth (positive) means more ppl per household; negative RDI_growth means fewer ppl per household
            rent_vs_rdi_growth()
    |_____  using them together
            high density with de-densification (high RDI and low RDI_growth) has highest rent growth
            high density with densification (high RDI and high RDI_growth) has high rent growth
            low density with de-densification (low RDI and low RDI_growth) has low rent growth
            low density with densification (low RDI and high RDI_growth) has lowest rent growth
            but there is a negligible difference in splitting by RDI_growth in the low RDI group
            rent_vs_rdi_and_growth()
"""

""" Relationship between RDI and rent growth with supply:
    |_____ High density with de-densification with low starts (high RDI;low RDI_growth;low starts) has highest rent growth 
           High density with densification with high starts (high RDI;high RDI_growth;high starts) has high lowest rent growth
           rent_vs_rdi_starts()
"""
""" Does this tell us more than occupancy + supply does?:
    |_____  yes
    baseline()
"""
""" Does adding prior year rent growth tell us more than rent_vs_rdi_starts()
    |_____  yes
    rent_vs_rdi_starts_priorrg()
"""


def rent_vs_rdi():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = df.select("year", "RDI", "rrg_5yr_fwd", "rrrg_5yr_fwd").drop_nulls()
    df = df.with_columns(
        (pl.col("RDI") > pl.col("RDI").median()).over("year").alias("RDI_high")
    )
    print(
        df.group_by("RDI_high")
        .agg(
            [
                pl.col("RDI").median().alias("med_RDI"),
                pl.col("rrg_5yr_fwd").mean().alias("mean_rrg_5yr_fwd"),
                pl.col("rrrg_5yr_fwd").mean().alias("mean_rrrg_5yr_fwd"),
            ]
        )
        .sort("RDI_high", descending=True)
    )


def rent_vs_rdi_growth():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = (
        df.select("year", "RDI_growth", "rrg_5yr_fwd", "rrrg_5yr_fwd").drop_nulls()
        # .filter(pl.col("year").is_in([2005, 2010, 2015]))
    )
    df = df.with_columns(
        (pl.col("RDI_growth") > pl.col("RDI_growth").median())
        .over("year")
        .alias("RDI_growth_high")
    )
    print(
        df.group_by("RDI_growth_high")
        .agg(
            [
                pl.col("RDI_growth").median().alias("med_RDI_growth"),
                pl.col("rrg_5yr_fwd").mean().alias("mean_rrg_5yr_fwd"),
                pl.col("rrrg_5yr_fwd").mean().alias("mean_rrrg_5yr_fwd"),
            ]
        )
        .sort("RDI_growth_high", descending=True)
    )


def rent_vs_rdi_and_growth():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = df.select(
        "year", "RDI_growth", "RDI", "rrg_5yr_fwd", "rrrg_5yr_fwd"
    ).drop_nulls()
    df = df.with_columns(
        (pl.col("RDI_growth") > pl.col("RDI_growth").median())
        .over("year")
        .alias("RDI_growth_high"),
        (pl.col("RDI") > pl.col("RDI").median()).over("year").alias("RDI_high"),
    )
    print(
        df.group_by(["RDI_growth_high", "RDI_high"])
        .agg(
            [
                pl.col("RDI").median().alias("med_RDI"),
                pl.col("RDI_growth").median().alias("med_RDI_growth"),
                pl.col("rrg_5yr_fwd").mean().alias("mean_rrg_5yr_fwd"),
                pl.col("rrrg_5yr_fwd").mean().alias("mean_rrrg_5yr_fwd"),
            ]
        )
        .sort(["RDI_high", "RDI_growth_high"], descending=True)
    )


def rent_vs_rdi_starts_priorrg():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = df.select(
        "year",
        "RDI",
        "RDI_growth",
        pl.col("real_relative_rent_growth").alias("prior_rrrg"),
        "starts_pct",
        "rrg_5yr_fwd",
        "rrrg_5yr_fwd",
    ).drop_nulls()
    df = df.with_columns(
        (pl.col("RDI") > pl.col("RDI").median()).over("year").alias("RDI_high"),
        (pl.col("prior_rrrg") > pl.col("prior_rrrg").median())
        .over("year")
        .alias("rrrg_high"),
        (pl.col("RDI_growth") > pl.col("RDI_growth").median())
        .over("year")
        .alias("RDI_growth_high"),
        (pl.col("starts_pct") > pl.col("starts_pct").median())
        .over("year")
        .alias("starts_high"),
    )
    print(
        df.group_by("RDI_high", "RDI_growth_high", "starts_high", "rrrg_high")
        .agg(
            [
                pl.col("RDI").median().alias("med_RDI"),
                pl.col("prior_rrrg").median().alias("med_prior_rrrg"),
                pl.col("RDI_growth").median().alias("med_RDI_growth"),
                pl.col("starts_pct").median().alias("med_starts_pct"),
                pl.col("rrg_5yr_fwd").mean().alias("mean_rrg_5yr_fwd"),
                pl.col("rrrg_5yr_fwd").mean().alias("mean_rrrg_5yr_fwd"),
            ]
        )
        .sort("mean_rrg_5yr_fwd", descending=True)
    )


def rent_vs_rdi_starts():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = df.select(
        "year", "RDI", "RDI_growth", "starts_pct", "rrg_5yr_fwd", "rrrg_5yr_fwd"
    ).drop_nulls()
    df = df.with_columns(
        (pl.col("RDI") > pl.col("RDI").median()).over("year").alias("RDI_high"),
        (pl.col("RDI_growth") > pl.col("RDI_growth").median())
        .over("year")
        .alias("RDI_growth_high"),
        (pl.col("starts_pct") > pl.col("starts_pct").median())
        .over("year")
        .alias("starts_high"),
    )
    print(
        df.group_by("RDI_high", "RDI_growth_high", "starts_high")
        .agg(
            [
                pl.col("RDI").median().alias("med_RDI"),
                pl.col("RDI_growth").median().alias("med_RDI_growth"),
                pl.col("starts_pct").median().alias("med_starts_pct"),
                pl.col("rrg_5yr_fwd").mean().alias("mean_rrg_5yr_fwd"),
                pl.col("rrrg_5yr_fwd").mean().alias("mean_rrrg_5yr_fwd"),
            ]
        )
        .sort("mean_rrg_5yr_fwd", descending=True)
    )


def baseline():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = df.select(
        "year",
        "occ",
        "occupancy_delta",
        "starts_pct",
        "rrg_5yr_fwd",
        "rrrg_5yr_fwd",
    ).drop_nulls()
    df = df.with_columns(
        (pl.col("occ") > pl.col("occ").median()).over("year").alias("occ_high"),
        (
            pl.col("occupancy_delta") > pl.col("occupancy_delta").median().over("year")
        ).alias("od"),
        (pl.col("starts_pct") > pl.col("starts_pct").median())
        .over("year")
        .alias("starts_high"),
    )
    print(
        df.group_by("occ_high", "od", "starts_high")
        .agg(
            [
                pl.col("occ").median().alias("med_occ"),
                pl.col("od").median().alias("med_occ_delta"),
                pl.col("starts_pct").median().alias("med_starts_pct"),
                pl.col("rrg_5yr_fwd").mean().alias("mean_rrg_5yr_fwd"),
                pl.col("rrrg_5yr_fwd").mean().alias("mean_rrrg_5yr_fwd"),
            ]
        )
        .sort("mean_rrg_5yr_fwd", descending=True)
    )


# Hypothesis Journal
"""
 H1: The means between the groups defined by RDI, RDI_growth, starts_pct, and prior_rrrg are significantly different.
    |____ The means are all significantly different except RDI_growth
    |____ Try using the interaction of RDI and RDI_growth: 
        |____ interaction term not significant
        |____ try trailing RDI_growth: no
        |____ prior period RDI: yes
        test_means()
H2: the variables RDI, RDI_last, starts_pct, and prior_rrrg are significant predictors of rrg_5yr_fwd 
    |____ only RDI_trailing and prior_rrrg
    |____ try combining into interaction terms
        "RDI_trailing", "prior_rrrg", "starts_pct", and the interaction of the three are significant predictors both of RRR and RRG_5yr_fwd 
        test_preds()
H3: going from one group to another (e.g. RDI_group == 0 to RDI_group == 1) results in a significant change in rrg_5yr_fwd
    |____ At the top decile of increase over trailing 4 years there's a spike in rrrg
        group_shift()
    |____ Integer shifting shows taht 3.0 is the sweet spot:
        ┌──────────────┬──────┬──────┬──────┬──────┐
        │ RDI_trailing ┆ 2.0  ┆ 2.5  ┆ 3.0  ┆ 3.5  │
        │ ---          ┆ ---  ┆ ---  ┆ ---  ┆ ---  │
        │ f64          ┆ i64  ┆ i64  ┆ i64  ┆ i64  │
        ╞══════════════╪══════╪══════╪══════╪══════╡
        │ 2.0          ┆ -118 ┆ -95  ┆ null ┆ null │
        │ 2.5          ┆ -40  ┆ 48   ┆ -42  ┆ null │
        │ 3.0          ┆ null ┆ 78   ┆ 161  ┆ 312  │
        │ 3.5          ┆ null ┆ null ┆ 534  ┆ 223  │
        │ 4.0          ┆ null ┆ null ┆ null ┆ 28   │
        └──────────────┴──────┴──────┴──────┴──────┘
        int_group_shift()
H4: given the above, there should be a good nonlinear function that forecasts RG based on trailing and next RDI
    |____  no, a polynomial regression with degree 2 on RDI_trailing and prior_rrrg is significant

H5: RDI alone is a significant predictor of rent-growth:
    int_group_shift()

H6: what about an OLS using dummy vars with the shift in groups?
    also not significatn

H7: does the number of positive RDI shifts in the last 5 years predict rent growth?
    |____  yes, the fewer the positive RDI_growth periods, the higher the rent growth
    pos_shift_count()
    
"""


def test_means():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = df.sort(["msa", "year"], descending=[True, True]).with_columns(
        pl.col("RDI").shift(-1).over("msa").alias("RDI_trailing"),
    )

    df = df.select(
        "year",
        "RDI",
        "RDI_growth",
        pl.col("real_relative_rent_growth").alias("prior_rrrg"),
        "starts_pct",
        "rrg_5yr_fwd",
        "rrrg_5yr_fwd",
        "RDI_trailing",
    ).drop_nulls()
    df = df.with_columns(
        (pl.col("RDI") > pl.col("RDI").median().over("year")).alias("RDI_group"),
        (pl.col("prior_rrrg") > pl.col("prior_rrrg").median().over("year")).alias(
            "prior_rrrg_group"
        ),
        (pl.col("RDI_trailing") > pl.col("RDI_trailing").median())
        .over("year")
        .alias("RDI_trailing_group"),
        (pl.col("starts_pct") > pl.col("starts_pct").median())
        .over("year")
        .alias("starts_group"),
    )
    import statsmodels.api as sm

    pdf = df.to_pandas()
    # Test means for each group independently
    for col in [
        "RDI_group",
        "RDI_trailing_group",
        "starts_group",
        "prior_rrrg_group",
    ]:
        group0 = pdf[pdf[col] == 0]["rrg_5yr_fwd"]
        group1 = pdf[pdf[col] == 1]["rrg_5yr_fwd"]
        mean_diff = group1.mean() - group0.mean()
        t_stat, p_val = ttest_ind(group0, group1, nan_policy="omit")
        print(
            f"{col}: t-stat={t_stat:.3f}, p-value={p_val:.3g}, (mean diff={mean_diff:.3f})"
        )


def test_preds():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = df.sort(["msa", "year"], descending=[True, True]).with_columns(
        pl.col("RDI").shift(-5).over("msa").alias("RDI_trailing"),
    )
    df = df.with_columns(plus_rdi=(pl.col("RDI_growth") > 0)).with_columns(
        pl.col("plus_rdi")
        .rolling_sum(window_size=5, min_samples=5)
        .over("msa")
        .alias("plus_rdi_5yr")
    )

    df = (
        df.select(
            "year",
            "RDI",
            "RDI_trailing",
            "plus_rdi_5yr",
            pl.col("real_relative_rent_growth").alias("prior_rrrg"),
            "starts_pct",
            "rrg_5yr_fwd",
            "rrrg_5yr_fwd",
        )
        .drop_nulls()
        .with_columns(
            (pl.col("prior_rrrg") * pl.col("starts_pct")).alias("interaction_1")
        )
        # .filter(pl.col("year").is_in([2006, 2012, 2018]))
    )
    import matplotlib.pyplot as plt

    # Fit model as before
    pdf = df.to_pandas()
    X = pdf[["RDI", "plus_rdi_5yr", "prior_rrrg", "interaction_1"]]
    X = sm.add_constant(X)
    y = pdf["rrg_5yr_fwd"]
    model = sm.OLS(y, X).fit()
    y_pred = model.predict(X)

    # Plot predictions vs actuals
    plt.figure(figsize=(7, 7))
    plt.scatter(y, y_pred, alpha=0.6, label="Predictions")
    # Line of best fit
    fit = np.polyfit(y, y_pred, 1)
    plt.plot(y, np.polyval(fit, y), color="red", label="Best fit")
    # 45-degree line
    plt.plot(
        [y.min(), y.max()],
        [y.min(), y.max()],
        color="gray",
        linestyle="--",
        label="y=x",
    )
    # R^2
    r2 = r2_score(y, y_pred)
    plt.title(f"Predicted vs Actual (R²={r2:.3f})")
    plt.xlabel("Actual rrg_5yr_fwd")
    plt.ylabel("Predicted rrg_5yr_fwd")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print(model.summary())


def group_shift():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = df.sort(["msa", "year"], descending=[True, True]).with_columns(
        pl.col("RDI").shift(-4).over("msa").alias("RDI_trailing"),
    )

    df = (
        df.select(
            "year",
            "RDI",
            "RDI_trailing",
            "rrg_5yr_fwd",
            "rrrg_5yr_fwd",
        )
        .drop_nulls()
        .with_columns((pl.col("RDI") - pl.col("RDI_trailing")).alias("RDI_move"))
        .with_columns(pl.col("RDI_move").qcut(5).alias("RDI_move_group"))
    )
    print(df)

    print(
        df.group_by("RDI_move_group")
        .agg(
            pl.col("RDI_move").mean().alias("mean_RDI_move").round(3),
            ((10000 * pl.col("rrrg_5yr_fwd")).mean())
            .cast(pl.Int64)
            .alias("mean_rrrg_5yr_fwd_bps"),
            ((1000 * pl.col("rrg_5yr_fwd")).mean())
            .cast(pl.Int64)
            .alias("mean_rrg_5yr_fwd_bps"),
        )
        .sort("RDI_move_group", descending=True)
    )


def int_group_shift():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = df.sort(["msa", "year"], descending=[True, True]).with_columns(
        ((pl.col("RDI").shift(-1).over("msa") * 2).round() / 2).alias("RDI_trailing"),
        ((pl.col("RDI") * 2).round() / 2).alias("RDI"),
    )
    df = (
        df.select(
            "year",
            "RDI",
            "RDI_trailing",
            "rrg_5yr_fwd",
            "rrrg_5yr_fwd",
        )
        .drop_nulls()
        .with_columns((pl.col("RDI") - pl.col("RDI_trailing")).alias("RDI_move"))
    )
    with pl.Config(set_tbl_rows=-1):
        print("MEAN")
        print(
            df.pivot(
                index="RDI_trailing",
                on="RDI",
                values="rrrg_5yr_fwd",
                aggregate_function="mean",
                sort_columns=True,
            )
            .with_columns((pl.all().exclude("RDI_trailing") * 10000).cast(pl.Int64))
            .sort("RDI_trailing")
        )
        print("COUNT")
        print(
            df.pivot(
                index="RDI_trailing",
                on="RDI",
                values="rrrg_5yr_fwd",
                aggregate_function="len",
                sort_columns=True,
            ).sort("RDI_trailing")
        )
        print("RDI_GROUP_MEAN")
        print(
            df.select(pl.col("RDI"), "rrrg_5yr_fwd", "rrg_5yr_fwd")
            .group_by("RDI")
            .agg(
                (10000 * pl.col("rrrg_5yr_fwd"))
                .mean()
                .cast(pl.Int64)
                .alias("rrrg_bps"),
                (10000 * pl.col("rrg_5yr_fwd")).mean().cast(pl.Int64).alias("rr_bps"),
                pl.col("rrrg_5yr_fwd").count().alias("count"),
            )
            .sort("RDI")
        )


def categorical_groupshift_model():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = df.sort(["msa", "year"], descending=[True, True]).with_columns(
        ((pl.col("RDI").shift(-1).over("msa") * 2).round() / 2).alias("RDI_trailing"),
        ((pl.col("RDI") * 2).round() / 2).alias("RDI"),
    )
    df = (
        df.select(
            "year",
            "msa",
            "RDI",
            "RDI_trailing",
            "real_relative_rent_growth",
            "rrg_5yr_fwd",
            "rrrg_5yr_fwd",
        )
        .drop_nulls()
        .with_columns((pl.col("RDI") - pl.col("RDI_trailing")).alias("RDI_move"))
    ).write_csv(
        Path(__file__).resolve().parent.parent
        / "data"
        / "categorical_groupshift_model.csv"
    )
    import statsmodels.api as sm

    # Convert to pandas DataFrame for easier dummy variable creation
    pdf = df.to_pandas()

    # Create categorical/dummy variables for RDI and RDI_move
    pdf["RDI_cat"] = pdf["RDI"].astype("category")
    pdf["RDI_move_cat"] = pdf["RDI_move"].astype("category")

    # Create dummy variables (drop_first avoids multicollinearity)
    X = pd.get_dummies(pdf[["RDI_cat", "RDI_move_cat"]], drop_first=True)
    X = sm.add_constant(X)
    X = X.astype(float)  # Ensure all columns are float dtype
    print(X)
    y = pdf["rrrg_5yr_fwd"]

    # Fit OLS regression
    model = sm.OLS(y, X).fit()
    # Print MAE
    y_pred = model.predict(X)
    print("MAE:", mean_absolute_error(y, y_pred))
    print(model.summary())


def pos_shift_count():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = df.sort(["msa", "year"])
    df = df.with_columns(plus_rdi=(pl.col("RDI_growth") > 0)).with_columns(
        pl.col("plus_rdi")
        .rolling_sum(window_size=5, min_samples=5)
        .over("msa")
        .alias("plus_rdi_5yr")
    )
    print(
        df.filter(pl.col("plus_rdi_5yr") == 5).select(
            "year", "msa", "plus_rdi_5yr", "rrrg_5yr_fwd"
        )
    )
    print(
        df.group_by("plus_rdi_5yr")
        .agg((10000 * pl.col("rrrg_5yr_fwd").mean()).alias("mean_rrrg_5yr_fwd"))
        .sort("plus_rdi_5yr", descending=True)
    )


### Findings ###
"""
RDI is a significant predictor of rent growth, but RDI_growth as a continuous variable is not.
"""
test_means()
"""
RDI_growth as a categorical variable (positive/negative) is a significant predictor of rent growth
With the records with 5 periods of positive RDI growth over the last 5 years having the lowest rent growth
┌──────────────┬───────────────────┐
│ plus_rdi_5yr ┆ mean_rrrg_5yr_fwd │
│ ---          ┆ ---               │
│ u32          ┆ f64               │
╞══════════════╪═══════════════════╡
│ null         ┆ -1.70203          │
│ 5            ┆ -252.691469       │
│ 4            ┆ -5.17122          │
│ 3            ┆ 49.644165         │
│ 2            ┆ 65.316567         │
│ 1            ┆ 208.835262        │
│ 0            ┆ 368.924413        │
└──────────────┴───────────────────┘
"""
pos_shift_count()
"""
For a linear model, RDI, the number of positive RDI years in the last 5, prior_rrrg, starts_pct, and their interaction create a valid model but low Rsquare.
This is the only one that seems to work well; other atteempts at categorical failed
"""
test_preds()
"""
This is likely because there's a nonlinear effect in that
Moving from one group to another (e.g. RDI_group == 0 to RDI_group == 1) results in a significant change in rrg_5yr_fwd
which is different from moving between other groups (e.g. RDI_group == 1 to RDI_group == 2).
But even just being in one group is a significant predictor of rrg_5yr_fwd.
"""
int_group_shift()
