import pandas as pd
import polars as pl
from scipy.stats import linregress
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from statsmodels.formula.api import ols
from statsmodels.api import OLS, add_constant

import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import skew, kurtosis


def get_data():
    df = pl.read_csv(Path(__file__).resolve().parent.parent / "data" / "msa_data.csv")
    df.columns = [
        "year",
        "msa",
        "rentpsf",
        "occ",
        "inventory",
        "delivered",
        "pop",
        "rent_growth",
        "demand_units",
        "absorption",
    ]
    df = df.with_columns(pl.col("year").cast(pl.Int16))

    # Filter to top 100 MSAs by 2001 inventory
    top_100_msas = (
        df.filter(pl.col("year") == 2001)
        .sort("inventory", descending=True)
        .head(100)
        .select("msa")
    )
    df = df.filter(pl.col("msa").is_in(top_100_msas))
    # Load CPI and calculate cumulative inflation adjustment
    cpi = pl.read_csv(Path(__file__).resolve().parent.parent / "data" / "cpi.csv")
    cpi = (
        cpi.with_columns(pl.col("year").cast(pl.Int16))
        .filter(pl.col("year") > 1999)
        .select(pl.col("year"), pl.col("cpi_pct"))
        .sort("year")
        .with_columns(cpi_cum=(pl.col("cpi_pct") + 1).cum_prod())
    )
    # Join CPI to dataset
    df = df.join(cpi, on="year", how="left")

    # Convert to real values
    df = df.with_columns(
        [
            (pl.col("rent_growth") - pl.col("cpi_pct")).alias("real_rent_growth"),
            (pl.col("rentpsf") / pl.col("cpi_cum")).alias("real_rentpsf"),
        ]
    )

    df = df.sort("msa", "year")

    # Calculate dependent variables in stages
    df = df.with_columns(
        [
            (pl.col("pop") / pl.col("inventory")).alias("RDI"),
        ]
    )

    df = df.with_columns(
        [
            (pl.col("delivered") / pl.col("inventory").shift(1).over("msa")).alias(
                "supply_growth"
            ),
            pl.col("real_rent_growth")
            .shift(-1)
            .over("msa")
            .alias("real_rent_growth_next_year"),
            pl.col("real_rent_growth")
            .shift(1)
            .over("msa")
            .alias("rent_growth_prior_year"),
            (pl.col("occ") - pl.col("occ").shift(1).over("msa")).alias("occ_growth"),
            (pl.col("demand_units").pct_change().over("msa")).alias(
                "demand_units_growth"
            ),
            (pl.col("absorption") - pl.col("absorption").shift(1).over("msa")).alias(
                "absorption_growth"
            ),
            ((pl.col("demand_units") / pl.col("inventory")).alias("demand_pct")),
        ]
    )

    df = df.with_columns(
        [
            pl.col("RDI").pct_change().over("msa").alias("RDI_growth"),
            # pl.col("RDI").pct_change().over("msa").alias("implied_demand"),
        ]
    )

    df = df.with_columns(
        (
            pl.col("real_rent_growth_next_year")
            - pl.col("real_rent_growth_next_year").median().over("year")
        ).alias("real_relative_rg_next_year")
    )
    df = df.with_columns(
        (
            pl.col("real_rent_growth")
            - pl.col("real_rent_growth").median().over("year")
        ).alias("real_relative_rent_growth")
    )
    # # Remove any rows with missing inflation adjustment
    # df = df.filter(pl.col("cpi_cum").is_not_null() & (pl.col("cpi_cum") != 0))
    df = df.select(
        [
            "year",
            "msa",
            "rent_growth",
            "real_rentpsf",
            "real_rent_growth",
            "RDI",
            "real_rent_growth_next_year",
            "real_relative_rent_growth",
            "RDI_growth",
            "supply_growth",
            "real_relative_rg_next_year",
        ]
    )
    df.write_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    return df


def supply_demand_curves(df, show=True):
    var_y = "real_rentpsf"
    df = df.dropna()
    # df = df[df["msa"] == "New York - NY"]
    # df = df[df["year"] == 2023]
    var_x = "RDI_growth"
    slope, intercept, r_value, p_value, std_err = linregress(df[var_x], df[var_y])
    var_x = "supply_growth"
    slope2, intercept2, r_value, p_value, std_err = linregress(df[var_x], df[var_y])
    # Intersection point
    intersection_x = (intercept2 - intercept) / (slope - slope2)
    intersection_y = intercept + slope * intersection_x
    if show:
        fig, ax = plt.subplots()

        # Demand curve
        ax.scatter(df[var_x], df[var_y], label="Implied Demand", color="green")

        # Generate x values for the line plot
        x_vals_demand = np.linspace(df[var_x].min(), df[var_x].max(), 100)
        ax.plot(x_vals_demand, intercept + slope * x_vals_demand, color="green")

        # Supply growth curve
        ax.scatter(df[var_x], df[var_y], label="Supply Growth", color="black")

        # Generate x values for the line plot
        x_vals_supply = np.linspace(df[var_x].min(), df[var_x].max(), 100)
        ax.plot(x_vals_supply, intercept2 + slope2 * x_vals_supply, color="black")

        ax.plot(intersection_x, intersection_y, "ro", label="Intersection")
        ax.vlines(
            x=intersection_x,
            ymin=0,
            ymax=intersection_y,
            linestyle="--",
            color="red",
        )
        ax.hlines(
            y=intersection_y,
            xmin=-0.05,
            xmax=intersection_x,
            linestyle="--",
            color="red",
        )

        # Adding labels and title
        ax.set_xlabel("Quantity of Units (as a % of existing)")
        ax.set_ylabel("Rent per Square Foot")
        ax.set_title("Supply and Demand Curves")
        ax.legend()
        # ax.set_xlim(-0.05, 0.05)
        # ax.set_ylim(2, 5)

        # Display the plot
        plt.show()
    return intersection_x, intersection_y


def predict_future(df):
    df["interaction"] = df["RDI_growth"] * df["supply_growth"] * df["real_rent_growth"]
    vars = [
        # "supply_growth",
        # "RDI_growth",
        # # "rent_growth",
        "interaction",
    ]
    year = 2022
    df_train = df[df["year"] <= year].dropna()
    X = df_train[vars]
    y = df_train["real_rent_growth_next_year"]
    X = sm.add_constant(X)  # Add constant term to the predictor variable
    model = sm.OLS(y, X).fit()
    df_test = df[df["year"] == 2023]
    X_test = df_test[vars]
    X_test = sm.add_constant(X_test)
    y_hat = model.predict(X_test)
    forecast = pd.concat(
        [df_test[["msa", *vars]], pd.DataFrame(y_hat, columns=["y_hat"])], axis=1
    ).sort_values("y_hat")
    # print(forecast.head(20))
    # print(forecast.tail(20))
    # forecast.to_csv(r"C:\users\mlarriva\desktop\prediction.csv")
    sns.boxplot(x=pd.qcut(df["RDI_growth"], q=4), y=df["real_rent_growth_next_year"])
    plt.xlabel("RDI Growth Quartiles")
    plt.ylabel("Rent Growth Next Year")
    plt.title("Rent Growth Next Year by RDI Growth Quartile")
    # plt.show()

    forecast = forecast.merge(
        df_test[["msa", "year", "real_rent_growth_next_year"]],
        on="msa",
        how="left",
        suffixes=("_forecast", "_actual"),
    )
    rsquare = (
        np.corrcoef(forecast["y_hat"], forecast["real_rent_growth_next_year"])[0, 1]
        ** 2
    )
    print(f"R-squared between y_hat and real_rent_growth_next_year: {rsquare:.4f}")
    forecast_top_20 = forecast.nlargest(20, "y_hat")
    forecast_bottom_20 = forecast.nsmallest(20, "y_hat")
    difference = (
        forecast_top_20["real_rent_growth_next_year"].mean()
        - forecast_bottom_20["real_rent_growth_next_year"].mean()
    )
    print(
        f"Difference in real_rent_growth_next_year {year} between top 20 and bottom 20: {difference:.2%}"
    )


def supply_demand_annual(df, past_current="past"):
    if past_current == "each":
        # df = df.dropna()
        msas = df["msa"].unique()
        years = df["year"].unique()
        holder = []
        y_m = df.loc[df["year"] > 2010, ["year", "msa"]].drop_duplicates()
        for x in y_m.itertuples():
            df_t = df[
                (df["msa"] == x.msa)
                & (df["year"] < x.year)
                & (df["year"] >= x.year - 10)
            ]
            df_current = df[(df["msa"] == x.msa) & (df["year"] == x.year)]
            try:
                intersection_x, intersection_y = supply_demand_curves(df_t, show=False)
            except:
                intersection_x = np.nan
                intersection_y = np.nan
            holder.append(
                [
                    x.msa,
                    x.year,
                    intersection_x,
                    intersection_y,
                    df_current.real_rentpsf.values[0],
                    df_current.RDI_growth.values[0],
                    df_current.supply_growth.values[0],
                    df_current.real_rent_growth_next_year.values[0],
                    df_current.real_relative_rg_next_year.values[0],
                    df_current.real_rent_growth.values[0],
                    df_current.real_relative_rent_growth.values[0],
                    df_current.rent_growth.values[0],
                ]
            )
        holder = pd.DataFrame(
            holder,
            columns=[
                "msa",
                "year",
                "quantity",
                "price",
                "real_rentpsf",
                "RDI_growth",
                "supply_growth",
                "real_rent_growth_next_year",
                "real_relative_rg_next_year",
                "real_rent_growth",
                "real_relative_rent_growth",
                "rent_growth",
            ],
        )
    holder["overpriced"] = holder["price"] < holder["real_rentpsf"]
    holder["oversupplied"] = holder["quantity"] < holder["supply_growth"]
    holder["group"] = (
        holder["oversupplied"].astype(str) + "-" + holder["overpriced"].astype(str)
    )
    return holder


def simplify_anova(df):
    print(
        df.groupby(["overpriced", "oversupplied"])[
            ["real_relative_rg_next_year", "real_rent_growth_next_year"]
        ]
        .agg("mean")
        .map(lambda x: int(x * 10000))
        .reset_index()
    )
    model = ols(
        "real_relative_rg_next_year ~ C(overpriced) + C(oversupplied)",
        data=df,
    ).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    tukey = pairwise_tukeyhsd(
        endog=df.dropna()["real_relative_rg_next_year"],
        groups=df.dropna()["group"],
        alpha=0.05,
    )
    print(tukey)


# source = get_data().to_pandas()
# df = supply_demand_annual(source, past_current="each")
# df.to_csv(Path(__file__).resolve().parent.parent / "data" / "supply_demand_annual.csv")
# simplify_anova(df)
# predict_future(df)
# TODO: a rolling OOS prediction to show the value of the model
df = pl.read_csv(
    Path(__file__).resolve().parent.parent / "data" / "supply_demand_annual.csv"
)


def event_study(original):
    holder = []

    for event in ["True-False", "False-True", "True-True", "False-False", "same"]:
        df = original.with_columns(
            pl.when(pl.col("group") == pl.col("group").shift(1).over("msa"))
            .then(pl.lit("same"))
            .otherwise(pl.lit("switch"))
            .alias("transition"),
            pl.col("group").shift(1).over("msa").alias("prev_group"),
        ).drop_nulls(subset="prev_group")
        if event != "same":
            event_years = (
                # df.filter(pl.col("transition").shift(-1).over("msa") == "same")
                # .filter(pl.col("transition").shift(-2).over("msa") == "same")
                # df.filter(pl.col("transition").shift(1).over("msa") == "same")
                # .filter(pl.col("transition").shift(2).over("msa") == "same")
                df.filter(pl.col("transition") == "switch")
                .filter(pl.col("group") == event)
                .select(pl.col("msa"), pl.col("year").alias("event_year"))
            )
        else:
            event_years = (
                df.filter(pl.col("transition").shift(-1).over("msa") == "same")
                .filter(pl.col("transition").shift(-2).over("msa") == "same")
                .filter(pl.col("transition").shift(1).over("msa") == "same")
                .filter(pl.col("transition").shift(2).over("msa") == "same")
                .filter(pl.col("transition") == "same")
                .select(pl.col("msa"), pl.col("year").alias("event_year"))
            )
        df_with_event = (
            df.join(event_years, on="msa", how="inner")
            .with_columns((pl.col("year") - pl.col("event_year")).alias("event_time"))
            .filter(pl.col("event_time").is_between(-2, 2))
        )
        pivoted = (
            df_with_event.select(
                ["msa", "transition", "event_time", "real_relative_rent_growth"]
            )
            .pivot(
                values="real_relative_rent_growth",
                index=["msa", "event_time"],
                on="event_time",
                aggregate_function="mean",  # no aggregation needed
            )
            .rename(
                {
                    "-2": "rr_rent_growth_m2",
                    "-1": "rr_rent_growth_m1",
                    "0": "rr_rent_growth_0",
                    "1": "rr_rent_growth_p1",
                    "2": "rr_rent_growth_p2",
                }
            )
            .with_columns(pl.lit(event).alias("event"))
        )
        holder = holder + pivoted.to_dicts()
    holder = pd.DataFrame(holder)
    holder = holder[holder["event"].isin(["same", "True-False", "False-True"])]
    averaged_holder = holder.groupby("event").mean(numeric_only=True).reset_index()
    std_errors = holder.groupby("event").sem(numeric_only=True).reset_index()
    plt.figure(figsize=(10, 6))
    for event in averaged_holder["event"]:
        event_data = averaged_holder[averaged_holder["event"] == event]
        error_data = std_errors[std_errors["event"] == event]
        plt.errorbar(
            ["-2", "-1", "0", "1", "2"],
            event_data[
                [
                    "rr_rent_growth_m2",
                    "rr_rent_growth_m1",
                    "rr_rent_growth_0",
                    "rr_rent_growth_p1",
                    "rr_rent_growth_p2",
                ]
            ].values.flatten(),
            yerr=error_data[
                [
                    "rr_rent_growth_m2",
                    "rr_rent_growth_m1",
                    "rr_rent_growth_0",
                    "rr_rent_growth_p1",
                    "rr_rent_growth_p2",
                ]
            ].values.flatten(),
            label=event,
            capsize=5,
        )

    plt.xlabel("Years before and after segment change")
    plt.xticks(["-2", "-1", "0", "1", "2"])
    plt.ylabel("Real Rent Growth Relative to Median")
    plt.title("Real Rent Growth before and after a segment change")
    plt.legend(title="Segment switched into")
    # plt.show()
    plt.savefig(Path(__file__).resolve().parent.parent / "Figs" / "event_study.png")


event_study(df)
