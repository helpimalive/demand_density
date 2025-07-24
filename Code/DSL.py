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
        "POPULATION_RENTED",
        "msa",
        "year",
        "RDI",
        "RDI_growth",
        pl.col("real_relative_rent_growth").alias("prior_rrrg"),
        "starts_pct",
        "rrg_5yr_fwd",
        "rrrg_5yr_fwd",
        "RDI_trailing",
    ).drop_nulls()
    # Filter to MSAs with POPULATION_RENTED in the top 100 as of 2018
    top_2018 = (
        df.filter((pl.col("year") == 2018))
        .sort("POPULATION_RENTED", descending=True)
        .select("msa")
        .head(50)
        .to_series()
        .to_list()
    )

    df = df.filter(pl.col("msa").is_in(top_2018))
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
        (pl.col("rrg_5yr_fwd") > pl.col("rrg_5yr_fwd").median().over("year")).alias(
            "rrg_group"
        ),
    )
    # Create a pivot table: index=year, columns=RDI_group, values=count of rrg_group==True
    var = "RDI_group"  # Change this to 'RDI_group', 'prior_rrrg_group', or 'RDI_trailing_group' as needed
    pivot = (
        # df.group_by(["year", "RDI_group"])
        df.group_by(["RDI_group", "starts_group"])
        .agg(pl.col("rrg_group").sum().alias("rrg_group_true_count"))
        .pivot(values="rrg_group_true_count", index="starts_group", columns="RDI_group")
    )
    with pl.Config(set_tbl_rows=-1):
        print(pivot)

    # df = df.with_columns(BOTH=(pl.col("RDI_group") & pl.col("prior_rrrg_group")))
    # pivot = (
    #     df.group_by(["year", "BOTH"])
    #     .agg(pl.col("rrg_5yr_fwd").mean().alias("mean_rrg_5yr_fwd"))
    #     .pivot(values="mean_rrg_5yr_fwd", index="year", columns="BOTH")
    #     .sort("year")
    # )
    # with pl.Config(set_tbl_rows=-1):
    #     print(
    #         pivot.with_columns(diff=(pl.col("true") - pl.col("false")))
    #         .filter(pl.col("year") > 2012)
    #         .mean()
    #     )

    # Write the processed DataFrame to CSV for further analysis
    # df.write_csv(
    #     Path(__file__).resolve().parent.parent / "data" / "test_means_groups.csv"
    # )
    import statsmodels.api as sm

    pdf = df.to_pandas()
    # Test means for each group independently
    for col in [
        "RDI_group",
        "RDI_trailing_group",
        "starts_group",
        "prior_rrrg_group",
    ]:

        group0 = pdf[pdf[col] == 0]["rrrg_5yr_fwd"]
        group1 = pdf[pdf[col] == 1]["rrrg_5yr_fwd"]
        mean_diff = group1.mean() - group0.mean()
        t_stat, p_val = ttest_ind(group0, group1, nan_policy="omit")
        print(
            f"{col}: t-stat={t_stat:.3f}, p-value={p_val:.3g}, (mean diff={mean_diff:.3f} in rrrg_5yr_fwd)"
        )


def test_preds():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = df.sort(["msa", "year"], descending=[True, True]).with_columns(
        pl.col("RDI").shift(-1).over("msa").alias("RDI_trailing"),
        pl.col("RDI").shift(-2).over("msa").alias("RDI_trailing_2"),
    )
    df = (
        df.with_columns(plus_rdi=(pl.col("RDI_growth") > 0))
        .with_columns(
            pl.col("plus_rdi")
            .rolling_sum(window_size=5, min_samples=5)
            .over("msa")
            .alias("plus_rdi_5yr")
        )
        .with_columns(
            INTERACTION=pl.col("RDI") * pl.col("RDI_trailing"),
            TRDI=(pl.col("POPULATION_RENTED") + pl.col("POPULATION_OWNED"))
            / pl.col("HOUSEHOLDS_RENTED"),
        )
        .drop_nulls()
    )

    # Fit model as before
    pdf = df.to_pandas()
    X = pdf[["RDI_trailing", "INTERACTION", "TRDI"]]
    X = sm.add_constant(X)
    y = pdf["rrrg_5yr_fwd"]
    # y = pdf["real_relative_rg_next_year"]
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
            "real_relative_rg_next_year",
            "rrrg_5yr_fwd",
        )
        .drop_nulls()
        .with_columns((pl.col("RDI") - pl.col("RDI_trailing")).alias("RDI_move"))
    )
    # Define RDI bins and labels
    bins = [-float("inf"), 2, 3, float("inf")]
    labels = ["RDI ≤ 2", "2 < RDI < 3", "RDI ≥ 3"]

    # Convert to pandas for easier binning and plotting
    pdf = df.to_pandas()
    pdf["RDI_group"] = pd.cut(pdf["RDI"], bins=bins, labels=labels, right=False)

    # Group by RDI_group and calculate mean rrg_5yr_fwd in bps
    grouped = (
        pdf.groupby("RDI_group")["rrg_5yr_fwd"]
        .mean()
        .mul(10000)
        .astype(int)
        .reset_index(name="rrg_5_yr_fwd_bps")
    )

    # Plot bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(grouped["RDI_group"], grouped["rrg_5_yr_fwd_bps"], color="skyblue")
    plt.xlabel("# of Residents per Rental Dwelling (RDI)")
    plt.ylabel("Mean 5yr Rent Growth (bps)")
    plt.title("Crowded Markets lead to Higher Rent Growth")
    plt.tight_layout()
    plt.show()
    with pl.Config(set_tbl_rows=-1):
        # print("MEAN")
        # print(
        #     df.pivot(
        #         index="RDI_trailing",
        #         on="RDI",
        #         values="rrrg_5yr_fwd",
        #         aggregate_function="mean",
        #         sort_columns=True,
        #     )
        #     .with_columns((pl.all().exclude("RDI_trailing") * 10000).cast(pl.Int64))
        #     .sort("RDI_trailing")
        # )
        # print("COUNT")
        # print(
        #     df.pivot(
        #         index="RDI_trailing",
        #         on="RDI",
        #         values="rrrg_5yr_fwd",
        #         aggregate_function="len",
        #         sort_columns=True,
        #     ).sort("RDI_trailing")
        # )
        print("RDI_GROUP_MEAN")
        print(
            df.select(
                pl.col("RDI"),
                "real_relative_rg_next_year",
                "rrrg_5yr_fwd",
                "rrg_5yr_fwd",
            )
            .group_by("RDI")
            .agg(
                (10000 * pl.col("rrrg_5yr_fwd"))
                .mean()
                .cast(pl.Int64)
                .alias("rrrg_5yr_fwd_bps"),
                (10000 * pl.col("rrg_5yr_fwd"))
                .mean()
                .cast(pl.Int64)
                .alias("rrg_5_yr_fwd_bps"),
                pl.col("rrrg_5yr_fwd").count().alias("count"),
            )
            .sort("RDI")
        )
        print("RDI_SHIFT_MEAN")
        print(
            df.select(
                pl.col("RDI_move"),
                "real_relative_rg_next_year",
                "rrrg_5yr_fwd",
                "rrg_5yr_fwd",
            )
            .group_by("RDI_move")
            .agg(
                (10000 * pl.col("real_relative_rg_next_year"))
                .mean()
                .cast(pl.Int64)
                .alias("rrrg_next_year_bps"),
                (10000 * pl.col("rrrg_5yr_fwd"))
                .mean()
                .cast(pl.Int64)
                .alias("rrrg5_bps"),
                (10000 * pl.col("rrg_5yr_fwd")).mean().cast(pl.Int64).alias("rr_bps"),
                pl.col("rrrg_5yr_fwd").count().alias("count"),
            )
            .sort("RDI_move")
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
        df.group_by("plus_rdi_5yr")
        .agg(
            (10000 * pl.col("rrrg_5yr_fwd").mean()).alias("mean_rrrg_5yr_fwd"),
            pl.col("rrrg_5yr_fwd").count().alias("count"),
        )
        .sort("plus_rdi_5yr", descending=True)
    )


### Findings ###
"""
Having RDI > annual RDI median is a significant predictor of rent growth
with higher RDI having higher rent growth in the 5 year
"""
print("TEST MEANS_____________________")
test_means()
"""
RDI_growth count as a categorical variable (positive/negative) is not a significant predictor of rent growth
"""
print("RDI_SHIFT _____________________")
pos_shift_count()
"""
For a linear model, RDI_trailing, RDI*RDI_trailing, and the total RDI (total pop / households rented) are significant predictors of rrg_5yr_fwd
"""
print("TEST PREDICTIONS _______________")
test_preds()
"""
RDI level and RDI shift are significant predictor of rent growth
"""
print("GROUP SHIFT ____________________")
int_group_shift()
