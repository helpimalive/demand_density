import pandas as pd
import polars as pl
from scipy.stats import linregress
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_ind
import plotly.express as px
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_1samp
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib as mpl
import numpy as np
from scipy.stats import sem

mpl.rcParams.update(
    {
        # Use a serif font throughout
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "font.size": 10,  # 9 pt for axis labels/text
        "axes.titlesize": 11,  # 10 pt for subplot titles
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # Line widths and marker sizes
        "lines.linewidth": 1.0,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        # Ticks: inward, only bottom/left
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        # No fancy whitegrid—just light grey if you need
        "axes.grid": False,
        "grid.color": "0.85",
        # Tight figure margins
        "figure.autolayout": True,
    }
)


def get_data(filter="top_100"):
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "msa_data.csv",
        schema_overrides={"FIPS": pl.Utf8},
    )

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
        "CBSA",
        "FIPS",
    ]
    df = df.with_columns(pl.col("year").cast(pl.Int16))

    if filter == "top_100":
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
    df = df.select(
        [
            "year",
            "msa",
            "pop",
            "rent_growth",
            "real_rentpsf",
            "real_rent_growth",
            "inventory",
            "RDI",
            "real_rent_growth_next_year",
            "real_relative_rent_growth",
            "RDI_growth",
            "supply_growth",
            "real_relative_rg_next_year",
            "FIPS",
        ]
    )
    if filter == "top_100":
        df.write_csv(
            Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
        )
    return df


def supply_demand_curves(df, show=False):
    var_y = "real_rentpsf"
    df = df.dropna()
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
        var_x = "RDI_growth"
        ax.scatter(df[var_x], df[var_y], label="Demand (RDI Growth)", color="green")

        # Generate x values for the line plot
        x_vals_demand = np.linspace(df[var_x].min(), df[var_x].max(), 100)
        ax.plot(x_vals_demand, intercept + slope * x_vals_demand, color="green")

        # Supply growth curve
        var_x = "supply_growth"
        ax.scatter(
            df[var_x], df[var_y], label="Supply (Inventory Growth)", color="black"
        )

        # Generate x values for the line plot
        x_vals_supply = np.linspace(df[var_x].min(), df[var_x].max(), 100)
        ax.plot(x_vals_supply, intercept2 + slope2 * x_vals_supply, color="black")

        ax.plot(intersection_x, intersection_y, "ro", label="Derived Equilibrium")
        ax.vlines(
            x=intersection_x,
            ymin=0.5,
            ymax=intersection_y + 0.5,
            linestyle="--",
            color="red",
        )
        ax.hlines(
            y=intersection_y,
            xmin=-0.05,
            xmax=intersection_x + 0.05,
            linestyle="--",
            color="red",
        )
        cy = df[df["year"] == df["year"].max()]
        ax.plot(
            cy["supply_growth"], cy["real_rentpsf"], "bo", label="Observed Equilibrium"
        )
        # Adding labels and title
        ax.set_xlabel("Quantity: RDI Growth and Supply Growth")
        ax.set_ylabel("Rent per Square Foot")
        # ax.set_title("Supply and Demand Curves")
        ax.legend()

        # Display the plot
        plt.show()
    return intersection_x, intersection_y


def predict_future(how):
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    if how == "naive":
        holder = []
        for year in range(2011, 2023):

            df_train = df[df["year"] == year]
            df_train["y_hat_group"] = pd.qcut(
                df_train["real_relative_rent_growth"],
                q=3,
                labels=["Bottom", "Middle", "Top"],
            )
            mean_values = df_train.groupby("y_hat_group", observed=False)[
                "real_relative_rg_next_year"
            ].mean()
            mean_values = mean_values.reset_index()
            holder.append(
                {
                    "year": year + 1,
                    "mean_bottom": mean_values[mean_values["y_hat_group"] == "Bottom"][
                        "real_relative_rg_next_year"
                    ].values[0],
                    "mean_top": mean_values[mean_values["y_hat_group"] == "Top"][
                        "real_relative_rg_next_year"
                    ].values[0],
                }
            )
        holder = pd.DataFrame(holder)
        holder["spread"] = holder["mean_top"] - holder["mean_bottom"]
        print(holder)
        print(holder.mean())
        holder.to_csv(
            Path(__file__).resolve().parent.parent / "data" / "naive_summary.csv",
            index=False,
        )
        return holder
    if how == "ARIMA":
        summary = pd.DataFrame()
        for year in range(2011, 2023):
            holder = []
            for msa in df["msa"].unique():
                df_train = df[
                    (df["year"] < year) & (df["year"] >= year - 10) & (df["msa"] == msa)
                ].sort_values("year")
                y = df_train["real_relative_rent_growth"].to_numpy()
                model = ARIMA(y, order=(1, 0, 0)).fit()
                y_hat = model.forecast(steps=1)
                holder.append(
                    {
                        "year": year + 1,
                        "msa": msa,
                        "y_hat": y_hat[0],
                        "real_relative_rg_next_year": df_train[
                            df_train["year"] == df_train["year"].max()
                        ]["real_relative_rg_next_year"].values[0],
                    }
                )
            holder = pd.DataFrame(holder)
            holder["y_hat_group"] = pd.qcut(
                holder["y_hat"], q=3, labels=["Bottom", "Middle", "Top"]
            )
            holder = (
                holder.groupby("y_hat_group")
                .agg(
                    actual_mean=("real_relative_rg_next_year", "mean"),
                )
                .reset_index()
            )
            holder["spread"] = (
                holder["actual_mean"].iloc[2] - holder["actual_mean"].iloc[0]
            )
            holder["year"] = year + 1
            holder = holder.pivot(
                index="year", columns="y_hat_group", values="actual_mean"
            ).reset_index()
            holder["spread"] = holder["Top"] - holder["Bottom"]
            holder = holder.rename(
                columns={"Bottom": "bottom", "Middle": "middle", "Top": "top"}
            )
            summary = pd.concat([summary, holder], ignore_index=True)
            summary.to_csv(
                Path(__file__).resolve().parent.parent / "data" / "arima_summary.csv",
                index=False,
            )

        return summary


def compare_predictions():
    ar = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "arima_summary.csv"
    )
    ar = ar[["year", "spread"]]
    ar.columns = ["year", "arima_spread"]

    ol = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "naive_summary.csv"
    )
    ol = ol[["year", "spread"]]
    ol.columns = ["year", "naive_spread"]

    rd = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "supply_demand_annual.csv"
    )
    rd = rd[["year", "group", "real_relative_rg_next_year"]]
    rd = rd.groupby(["year", "group"]).mean().reset_index()
    rd = rd.pivot(
        index="year", columns="group", values="real_relative_rg_next_year"
    ).reset_index()
    rd["spread"] = rd["False-True"] - rd["True-False"]
    rd = rd[["year", "spread"]]
    # rd["year"] = rd["year"] + 1
    rd.columns = ["year", "rdi_spread"]
    rd = rd.merge(ar, on="year", how="left")
    rd = rd.merge(ol, on="year", how="left")
    print(rd)
    print(rd.mean())
    # Plot the spreads over time
    plt.figure(figsize=(10, 6))
    plt.plot(rd["year"], rd["rdi_spread"], label="RDI Spread", marker="o")
    plt.plot(rd["year"], rd["arima_spread"], label="ARIMA Spread", marker="s")
    plt.plot(rd["year"], rd["naive_spread"], label="Naive Spread", marker="^")

    # Add labels, title, and legend
    plt.xlabel("Year")
    plt.ylabel("Spread")
    plt.title("Comparison of Spreads Over Time")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        Path(__file__).resolve().parent.parent
        / "Figs"
        / "spread_comparison_over_time.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    # Remove min and max from each column and calculate the mean
    plt.show()


def supply_demand_annual(df):
    msas = df["msa"].unique()
    years = df["year"].unique()
    holder = []
    y_m = df.loc[df["year"] > 2010, ["year", "msa"]].drop_duplicates()
    for x in y_m.itertuples():
        df_t = df[
            (df["msa"] == x.msa) & (df["year"] < x.year) & (df["year"] >= x.year - 10)
        ]
        df_current = df[(df["msa"] == x.msa) & (df["year"] == x.year)]
        try:
            intersection_x, intersection_y = supply_demand_curves(df_t)
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

    holder.to_csv(
        Path(__file__).resolve().parent.parent / "data" / "supply_demand_annual.csv",
        index=False,
    )
    return holder


def simplify_anova():
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "supply_demand_annual.csv"
    ).dropna(subset="real_relative_rg_next_year")
    # Replace Boolean codes with narrative labels
    df["group"] = df.apply(
        lambda row: (
            "Landlord-Favorable"
            if row["overpriced"] and not row["oversupplied"]
            else (
                "Renter-Favorable"
                if not row["overpriced"] and row["oversupplied"]
                else (
                    "Neutral-Early"
                    if not row["overpriced"] and not row["oversupplied"]
                    else "Neutral-Late"
                )
            )
        ),
        axis=1,
    )

    # Calculate mean next-year real-rent growth
    summary = (
        df.groupby("group")
        .agg(
            mean_next_year_relative_real_rent_growth=(
                "real_relative_rg_next_year",
                "mean",
            ),
            n_obs=("real_relative_rg_next_year", "size"),
        )
        .reset_index()
    )
    # Calculate means and 95% confidence intervals for each group
    group_means = df.groupby("group")["real_relative_rg_next_year"].mean()
    group_se = df.groupby("group")["real_relative_rg_next_year"].sem()
    ci_lower = group_means - 1.96 * group_se
    ci_upper = group_means + 1.96 * group_se
    # Calculate p-values for each group mean

    # Perform one-sample t-tests for each group mean
    p_values = {}
    for group in df["group"].unique():
        group_data = df[df["group"] == group]["real_relative_rg_next_year"]
        t_stat, p_value = ttest_1samp(group_data, 0, nan_policy="omit")
        p_values[group] = p_value
        # Add n_obs to the summary DataFrame
    summary["n_obs"] = df.groupby("group")["real_relative_rg_next_year"].size().values
    # Add p-values to the summary DataFrame
    summary["p-value"] = summary["group"].map(p_values)
    # Combine results into a DataFrame
    ci_summary = pd.DataFrame(
        {
            "mean (bps)": [int(x * 10000) for x in group_means],
            "95% CI Lower (bps)": [int(x * 10000) for x in ci_lower],
            "95% CI Upper (bps)": [int(x * 10000) for x in ci_upper],
            "p-value": summary["p-value"].values,
            "n_obs": summary["n_obs"].values,
        }
    ).reset_index(drop=True)

    print(ci_summary)
    assert False

    # Clustered standard errors by MSA
    model = smf.ols("real_relative_rg_next_year ~ C(group)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["msa"]}
    )

    # Prepare a DataFrame with the same structure for prediction
    group_means_df = df.groupby("group").mean(numeric_only=True).reset_index()
    group_means_df["group"] = group_means_df["group"].astype("category")

    # Predict group means and extract standard errors
    group_means = model.predict(group_means_df)
    group_se = model.get_robustcov_results(cov_type="cluster", groups=df["msa"]).bse
    # Calculate 95% confidence intervals for the group means
    ci_lower = group_means - 1.96 * group_se[: len(summary)]
    ci_upper = group_means + 1.96 * group_se[: len(summary)]
    summary["95% CI"] = list(zip(ci_lower.round(4) * 10000, ci_upper.round(4) * 10000))
    # Map standard errors to the corresponding groups
    summary["mean (bps)"] = (
        summary["mean_next_year_relative_real_rent_growth"] * 10000
    ).astype(int)
    summary["se (bps)"] = (group_se[: len(summary)] * 10000).astype(
        int
    )  # Match SEs to groups

    # Two-way ANOVA
    anova_model = smf.ols(
        "real_relative_rg_next_year ~ C(overpriced) + C(oversupplied)", data=df
    ).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)

    # Tukey HSD post-hoc test
    tukey = pairwise_tukeyhsd(
        endog=df["real_relative_rg_next_year"], groups=df["group"], alpha=0.05
    )

    # Format results into a table block
    print("\nMean Next-Year Real-Rent Growth and Standard Errors:")
    print(
        summary[["group", "mean (bps)", "se (bps)", "95% CI", "n_obs"]]
        .style.format({"p-value": "{:.4f}"})
        .to_string()
    )

    print("\nTwo-Way ANOVA:")
    print(anova_table)

    print("\nTukey HSD Post-Hoc Test:")
    print(tukey)

    # Add footnotes for SE and significance stars
    print("\nNote: *** p < 0.01, ** p < 0.05, * p < 0.10.")


def event_study():
    original = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "supply_demand_annual.csv"
    )
    holder = []
    original = original.with_columns(
        pl.when(pl.col("group") == "False-False")
        .then(pl.lit("UnderpricedUndersupplied"))
        .when(pl.col("group") == "True-False")
        .then(pl.lit("Landlord-Favorable"))
        .when(pl.col("group") == "False-True")
        .then(pl.lit("Renter-Favorable"))
        .when(pl.col("group") == "True-True")
        .then(pl.lit("OverpricedOversupplied"))
        .otherwise(pl.lit("Unknown"))
        .alias("group")
    )

    for event in [
        "Landlord-Favorable",
        "Renter-Favorable",
        # "OverpricedOversupplied",
        # "UnderpricedUndersupplied",
        "same",
    ]:
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
                # .filter(pl.col("transition").shift(1).over("msa") == "same")
                # .filter(pl.col("transition").shift(2).over("msa") == "same")
                df.filter(pl.col("transition") == "switch")
                .filter(pl.col("group") == event)
                .select(pl.col("msa"), pl.col("year").alias("event_year"))
            )
            event_years = (
                event_years.sort("event_year")
                .with_columns(
                    (
                        pl.col("event_year") - pl.col("event_year").shift(1).over("msa")
                    ).alias("year_diff")
                )
                .filter((pl.col("year_diff").is_null()) | (pl.col("year_diff") > 2))
                .drop("year_diff")
            )
            print(event)
            print(event_years)
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
                aggregate_function="mean",
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
    holder = holder[
        holder["event"].isin(["same", "Landlord-Favorable", "Renter-Favorable"])
    ]
    averaged_holder = holder.groupby("event").mean(numeric_only=True).reset_index()
    std_errors = holder.groupby("event").sem(numeric_only=True).reset_index()

    plt.figure(figsize=(10, 6))
    for event in averaged_holder["event"]:
        event_data = averaged_holder[averaged_holder["event"] == event]
        error_data = std_errors[std_errors["event"] == event]
        x_vals = np.array([-2, -1, 0, 1, 2])
        y_vals = event_data[
            [
                "rr_rent_growth_m2",
                "rr_rent_growth_m1",
                "rr_rent_growth_0",
                "rr_rent_growth_p1",
                "rr_rent_growth_p2",
            ]
        ].values.flatten()
        y_err = error_data[
            [
                "rr_rent_growth_m2",
                "rr_rent_growth_m1",
                "rr_rent_growth_0",
                "rr_rent_growth_p1",
                "rr_rent_growth_p2",
            ]
        ].values.flatten()
        if event == "Landlord-Favorable":
            color = "orange"
        elif event == "Renter-Favorable":
            color = "blue"
        else:
            color = "green"
        plt.plot(x_vals, y_vals, label=event, color=color)
        plt.fill_between(
            x_vals,
            y_vals - y_err,
            y_vals + y_err,
            alpha=0.1,
            label=f"{event} (95% CI)",
            color=color,
        )

    plt.xlabel("Years before and after segment change")
    plt.xticks(x_vals)
    plt.ylabel("Real Rent Growth Relative to Median")
    plt.title("Real Relative Rent Growth before and after a segment change")
    plt.legend(title="Segment switched into")
    plt.savefig(
        Path(__file__).resolve().parent.parent / "Figs" / "event_study.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.show()

    # Calculate differences in means and perform t-tests
    results = []
    for event in ["Landlord-Favorable", "Renter-Favorable", "same"]:
        event_data = holder[holder["event"] == event]

        # Calculate means for two years before and after the event
        before_means = event_data[["rr_rent_growth_m1", "rr_rent_growth_m2"]].mean()
        after_means = event_data[["rr_rent_growth_p1", "rr_rent_growth_p2"]].mean()

        # Perform t-tests for differences in means
        t_stat, p_value = ttest_ind(
            event_data[["rr_rent_growth_m1", "rr_rent_growth_m2"]].values.flatten(),
            event_data[["rr_rent_growth_p1", "rr_rent_growth_p2"]].values.flatten(),
            nan_policy="omit",
        )

        results.append(
            {
                "Event": event,
                "Mean Before": before_means.mean(),
                "Mean After": after_means.mean(),
                "Difference": after_means.mean() - before_means.mean(),
                "p-value": p_value,
            }
        )

    # Create a DataFrame for the results
    results_df = pd.DataFrame(results)

    # Display the table
    print("\nDifferences in Means and p-values:")
    print(results_df.to_string(index=False))


def choropleth_rdi_by_msa():
    # Load the supply_demand_annual.csv file
    df = get_data(filter=None).to_pandas()[["msa", "year", "RDI", "FIPS"]].dropna()
    df["RDI"] = df["RDI"].round(1)
    df_2019 = df[df["year"] == 2019]
    from urllib.request import urlopen
    import json

    with urlopen(
        "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    ) as response:
        counties = json.load(response)
    import plotly.express as px

    fig = px.choropleth_map(
        df,
        geojson=counties,
        locations="FIPS",
        color="RDI",
        color_continuous_scale="Spectral",
        range_color=(7, 100),
        map_style="carto-positron",
        zoom=3,
        center={"lat": 37.0902, "lon": -95.7129},
        opacity=0.5,
        labels={"unemp": "unemployment rate"},
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def plot_national_averages():
    # Load preprocessed data
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    ).dropna()
    print(df["year"].min(), df["year"].max())
    # Plot RDI vs Real Rent PSF for all MSAs
    # ds = df[df["year"] == 2023]
    ds = df.groupby(["msa"]).agg(
        {
            "RDI_growth": "mean",
            "real_rent_growth_next_year": "mean",
        }
    )
    plt.figure(figsize=(12, 8))
    plt.scatter(
        ds["RDI_growth"],
        ds["real_rent_growth_next_year"],
    )
    # Add labels for min and max x and y values
    min_x_msa = ds.loc[ds["RDI_growth"].idxmin()]
    max_x_msa = ds.loc[ds["RDI_growth"].idxmax()]
    min_y_msa = ds.loc[ds["real_rent_growth_next_year"].idxmin()]
    max_y_msa = ds.loc[ds["real_rent_growth_next_year"].idxmax()]

    plt.text(
        min_x_msa["RDI_growth"],
        min_x_msa["real_rent_growth_next_year"],
        min_x_msa.name,
        fontsize=9,
        ha="left",
    )
    plt.text(
        max_x_msa["RDI_growth"],
        max_x_msa["real_rent_growth_next_year"],
        max_x_msa.name,
        fontsize=9,
        ha="right",
    )
    plt.text(
        min_y_msa["RDI_growth"],
        min_y_msa["real_rent_growth_next_year"],
        min_y_msa.name,
        fontsize=9,
        ha="right",
    )
    plt.text(
        max_y_msa["RDI_growth"],
        max_y_msa["real_rent_growth_next_year"],
        max_y_msa.name,
        fontsize=9,
        ha="right",
    )
    plt.xlabel("Δ RDI (2011-2023)")
    plt.ylabel("Real Rent Growth (2012-2024)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    plt.grid()
    plt.tight_layout()
    slope, intercept, r_value, p_value, std_err = linregress(
        ds["RDI_growth"], ds["real_rent_growth_next_year"]
    )
    # Add text box with slope, R², and p-value
    textstr = f"β = {slope:.2f}\nR² = {r_value**2:.2f}\np = {p_value:.2e}"
    plt.gca().text(
        0.05,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )
    x_vals = np.linspace(ds["RDI_growth"].min(), ds["RDI_growth"].max(), 100)
    y_vals = slope * x_vals + intercept
    plt.plot(
        x_vals, y_vals, color="black", label=f"Line of Best Fit (R²={r_value**2:.2f})"
    )
    # Calculate 95% confidence interval
    y_pred = slope * ds["RDI_growth"] + intercept
    residuals = ds["real_rent_growth_next_year"] - y_pred
    std_error = np.std(residuals)
    ci = 1.96 * std_error  # 95% confidence interval

    plt.fill_between(
        x_vals,
        y_vals - ci,
        y_vals + ci,
        color="gray",
        alpha=0.2,
        label="95% Confidence Interval",
    )
    plt.legend()
    plt.title("Mean Change in RDI vs. Mean Real Rent Growth by MSA")
    plt.savefig(
        Path(__file__).resolve().parent.parent / "Figs" / "rdi_rent_growth_2024.pdf",
        format="pdf",
        bbox_inches="tight",  # crop extra white
        pad_inches=0.02,
    )
    plt.show()
    assert False

    # Plot a histogram of the change in RDI
    plt.figure(figsize=(10, 6))
    plt.hist(
        df["RDI_growth"].dropna(), bins=30, color="blue", alpha=0.7, edgecolor="black"
    )
    plt.xlabel("Change in RDI (Δ RDI)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Change in RDI: All MSAs (2012-2023)")
    plt.grid(axis="y", alpha=0.75)
    plt.tight_layout()
    plt.savefig(
        Path(__file__).resolve().parent.parent / "Figs" / "rdi_growth_histogram.pdf",
        format="pdf",
        bbox_inches="tight",  # crop extra white
        pad_inches=0.02,
    )
    plt.show()

    # Filter data for the year 2001
    df_2001 = df[df["year"] == 2001]

    # Identify MSAs with the least, middle, and greatest RDI values
    least_rdi_msa = df_2001.loc[df_2001["RDI"].idxmin(), "msa"]
    greatest_rdi_msa = df_2001.loc[df_2001["RDI"].idxmax(), "msa"]
    middle_rdi_msa = df_2001.iloc[
        (df_2001["RDI"].sort_values().reset_index(drop=True).index[len(df_2001) // 2])
    ]["msa"]
    # Calculate average RDI by year
    avg_rdi_by_year = df.groupby("year")["RDI"].mean().reset_index()

    # Plot the average RDI by year
    # Filter data for these MSAs
    selected_msas = [least_rdi_msa, middle_rdi_msa, greatest_rdi_msa]
    df_selected = df[df["msa"].isin(selected_msas)]

    # Plot RDI values over years for the selected MSAs
    plt.figure(figsize=(10, 6))
    for msa in selected_msas:
        msa_data = df_selected[df_selected["msa"] == msa]
        plt.plot(msa_data["year"], msa_data["RDI"], label=msa)

    plt.plot(
        avg_rdi_by_year["year"],
        avg_rdi_by_year["RDI"],
        label="Average RDI",
        linestyle="--",
        color="red",
    )
    plt.xlabel("Year")
    plt.ylabel("RDI")
    plt.title("RDI Values Over Years for Selected MSAs and Average")
    plt.legend(title="MSA")
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        Path(__file__).resolve().parent.parent
        / "Figs"
        / "rdi_trends_selected_msas.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.show()


def plot_austin_supply_demand():
    mpl.rcParams.update(
        {
            # Use a serif font throughout
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times"],
            "font.size": 10,  # 9 pt for axis labels/text
            "axes.titlesize": 11,  # 10 pt for subplot titles
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            # Line widths and marker sizes
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            # Ticks: inward, only bottom/left
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
            # No fancy whitegrid—just light grey if you need
            "axes.grid": False,
            "grid.color": "0.85",
            # Tight figure margins
            "figure.autolayout": True,
        }
    )

    df = get_data(filter="top_100").to_pandas()
    dx = df[(df["msa"] == "Austin - TX") & (df["year"] >= 2022)]
    df = df[(df["msa"] == "Austin - TX") & (df["year"] < 2022) & (df["year"] >= 2012)]
    var_y = "real_rentpsf"
    df = df.dropna()
    var_x = "RDI_growth"
    slope, intercept, r_value, p_value, std_err = linregress(df[var_x], df[var_y])
    var_x = "supply_growth"
    slope2, intercept2, r_value, p_value, std_err = linregress(df[var_x], df[var_y])
    # Intersection point
    intersection_x = (intercept2 - intercept) / (slope - slope2)
    intersection_y = intercept + slope * intersection_x
    print(intersection_x, intersection_y)
    fig, ax = plt.subplots()

    # Demand curve
    var_x = "RDI_growth"
    ax.scatter(df[var_x], df[var_y], label="Demand (RDI Growth) 2012-2021", color="red")

    # Generate x values for the line plot
    x_vals_demand = np.linspace(df[var_x].min(), df[var_x].max() + 0.01, 100)
    ax.plot(x_vals_demand, intercept + slope * x_vals_demand, color="red")

    # Supply growth curve
    var_x = "supply_growth"
    ax.scatter(
        df[var_x], df[var_y], label="Supply (Inventory Growth) 2012-2021", color="blue"
    )

    # Generate x values for the line plot
    x_vals_supply = np.linspace(df[var_x].min() - 0.02, df[var_x].max(), 100)
    ax.plot(x_vals_supply, intercept2 + slope2 * x_vals_supply, color="blue")

    ax.plot(
        intersection_x,
        intersection_y,
        marker="*",
        color="black",
        label="Derived Equilibrium 2021",
        markersize=10,
    )
    # ax.vlines(
    #     x=intersection_x,
    #     ymin=0.5,
    #     ymax=intersection_y + 0.5,
    #     linestyle="--",
    #     color="red",
    # )
    # ax.hlines(
    #     y=intersection_y,
    #     xmin=-0.05,
    #     xmax=intersection_x + 0.05,
    #     linestyle="--",
    #     color="red",
    # )
    cy = df[df["year"] == df["year"].max()]
    ax.plot(
        cy["supply_growth"],
        cy["real_rentpsf"],
        "v",
        color="gray",
        label="Price Shock 2021",
        markersize=10,  # Increase the marker size
    )

    ax.plot(
        dx["supply_growth"],
        dx["real_rentpsf"],
        "ro",
        color="purple",
        label="Equilibrium Recovery",
    )
    for i, row in dx.iterrows():
        ax.text(
            row["supply_growth"],
            row["real_rentpsf"],
            f"{str(row['year'])}",
            fontsize=9,
            ha="right",
        )
    # Adding labels and title
    ax.set_xlabel("Quantity: RDI Growth and Supply Growth")
    ax.set_ylabel("Rent per Square Foot ($)")
    ax.set_title("Supply and Demand Curves for Austin - TX (2012-2021)")
    # ax.set_title("Supply and Demand Curves")
    ax.legend()

    # Display the plot

    plt.savefig(
        Path(__file__).resolve().parent.parent / "Figs" / "austin_example.pdf",
        format="pdf",
        bbox_inches="tight",  # crop extra white
        pad_inches=0.02,
    )
    plt.show()


def plot_phoenix_supply_demand():
    mpl.rcParams.update(
        {
            # Use a serif font throughout
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times"],
            "font.size": 10,  # 9 pt for axis labels/text
            "axes.titlesize": 11,  # 10 pt for subplot titles
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            # Line widths and marker sizes
            "lines.linewidth": 1.0,
            "lines.markersize": 4,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.5,
            # Ticks: inward, only bottom/left
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
            # No fancy whitegrid—just light grey if you need
            "axes.grid": False,
            "grid.color": "0.85",
            # Tight figure margins
            "figure.autolayout": True,
        }
    )

    df = get_data(filter="top_100").to_pandas()
    dx = df[(df["msa"] == "Phoenix - AZ") & (df["year"] >= 2022)]
    df = df[(df["msa"] == "Phoenix - AZ") & (df["year"] < 2022) & (df["year"] >= 2012)]
    var_y = "real_rentpsf"
    df = df.dropna()
    var_x = "RDI_growth"
    slope, intercept, r_value, p_value, std_err = linregress(df[var_x], df[var_y])
    var_x = "supply_growth"
    slope2, intercept2, r_value, p_value, std_err = linregress(df[var_x], df[var_y])
    # Intersection point
    intersection_x = (intercept2 - intercept) / (slope - slope2)
    intersection_y = intercept + slope * intersection_x
    print(intersection_x, intersection_y)
    fig, ax = plt.subplots()

    # Demand curve
    var_x = "RDI_growth"
    ax.scatter(df[var_x], df[var_y], label="Demand (RDI Growth) 2012-2021", color="red")

    # Generate x values for the line plot
    x_vals_demand = np.linspace(df[var_x].min(), df[var_x].max(), 100)
    ax.plot(x_vals_demand, intercept + slope * x_vals_demand, color="red")

    # Supply growth curve
    var_x = "supply_growth"
    ax.scatter(
        df[var_x], df[var_y], label="Supply (Inventory Growth) 2012-2021", color="blue"
    )

    # Generate x values for the line plot
    x_vals_supply = np.linspace(df[var_x].min(), df[var_x].max(), 100)
    ax.plot(x_vals_supply, intercept2 + slope2 * x_vals_supply, color="blue")

    ax.plot(
        intersection_x,
        intersection_y,
        marker="*",
        color="black",
        label="Derived Equilibrium 2021",
        markersize=10,
    )
    # ax.vlines(
    #     x=intersection_x,
    #     ymin=0.5,
    #     ymax=intersection_y + 0.5,
    #     linestyle="--",
    #     color="red",
    # )
    # ax.hlines(
    #     y=intersection_y,
    #     xmin=-0.05,
    #     xmax=intersection_x + 0.05,
    #     linestyle="--",
    #     color="red",
    # )
    cy = df[df["year"] == df["year"].max()]
    ax.plot(
        cy["supply_growth"],
        cy["real_rentpsf"],
        "v",
        color="gray",
        label="Supply Shock 2021",
        markersize=10,  # Increase the marker size
    )

    ax.plot(
        dx["supply_growth"],
        dx["real_rentpsf"],
        "ro",
        color="purple",
        label="Equilibrium Recovery",
    )
    for i, row in dx.iterrows():
        ax.text(
            row["supply_growth"],
            row["real_rentpsf"],
            f"{str(row['year'])}",
            fontsize=9,
            ha="right",
        )
    # Adding labels and title
    ax.set_xlabel("Quantity: RDI Growth and Supply Growth")
    ax.set_ylabel("Rent per Square Foot ($)")
    ax.set_title("Supply and Demand Curves for Phoenix - AZ (2012-2021)")
    # ax.set_title("Supply and Demand Curves")
    ax.legend()

    # Display the plot

    plt.savefig(
        Path(__file__).resolve().parent.parent / "Figs" / "phoenix_example.pdf",
        format="pdf",
        bbox_inches="tight",  # crop extra white
        pad_inches=0.02,
    )
    plt.show()


def show_summary_statistics():
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    variables = [
        "pop",
        "inventory",
        "supply_growth",
        "RDI",
        "RDI_growth",
        "real_rentpsf",
        "real_rent_growth",
    ]
    summary = df[variables].describe().transpose()
    summary["median"] = df[variables].median()
    summary["missing_values"] = df[variables].isnull().sum()
    print(
        summary[
            [
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
                "median",
                "missing_values",
            ]
        ]
    )
    # Print records where RDI is greatest and least
    df = df.dropna()
    print("Record with greatest RDI:")
    print(df.loc[df["RDI_growth"].idxmax()])
    print("\nRecord with least RDI:")
    print(df.loc[df["RDI_growth"].idxmin()])

    # Print records where real_rent_growth is greatest and least
    print("\nRecord with greatest real_rent_growth:")
    print(df.loc[df["real_rent_growth"].idxmax()])
    print("\nRecord with least real_rent_growth:")
    print(df.loc[df["real_rent_growth"].idxmin()])


def plot_group_averages_with_confidence():
    var = "real_rent_growth_next_year"
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "supply_demand_annual.csv"
    ).dropna(subset=var)
    df["year"] = df["year"] + 1
    # Calculate the average for True-False and False-True groups
    df["group"] = df["group"].astype(str)
    df_pivot = df.pivot_table(index="year", columns="group", values=var, aggfunc="mean")

    # Calculate the overall average by year
    df_avg = df.groupby("year")[var].mean()

    # Plot the averages
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_pivot.index,
        df_pivot["True-False"],
        label="Renter-Favorable Average",
        color="blue",
    )
    plt.plot(
        df_pivot.index,
        df_pivot["False-True"],
        label="Landlord-Favorable Average",
        color="orange",
    )
    plt.plot(
        df_avg.index,
        df_avg,
        label="Overall Average",
        color="green",
        linestyle="--",
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Year")
    plt.ylabel("Average Real Rent Growth")
    plt.title("Average Real Rent Growth by Group and Overall")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(
        Path(__file__).resolve().parent.parent
        / "Figs"
        / "group_averages_over_time.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.show()


# plot_group_averages_with_confidence()
# df = get_data(filter="top_100").to_pandas()
# show_summary_statistics()
# dx = supply_demand_annual(get_data().to_pandas())
# event_study()
# plot_phoenix_supply_demand()
plot_austin_supply_demand()
# simplify_anova()
# plot_national_averages()
# choropleth_rdi_by_msa()
# predict_future(how="naive")
# summary = predict_future(how="ARIMA")
# compare_predictions()
