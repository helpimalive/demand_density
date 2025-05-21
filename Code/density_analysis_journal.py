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
from linearmodels.iv import IV2SLS

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


def get_data(filter=100):
    df = pl.read_excel(
        Path(__file__).resolve().parent.parent / "data" / "costar_raw.xlsx",
        # schema_overrides={"FIPS": pl.Utf8},
    )
    df = df.with_columns(pl.col("Period").str.slice(0, 4).cast(pl.Int16).alias("year"))
    df = df.with_columns(pl.col("Geography Name").str.replace(" USA$", "").alias("msa"))
    df = df.rename({"Market Effective Rent/SF": "rentpsf"})
    df = df.rename({"Market Effective Rent Growth 12 Mo": "rent_growth"})
    df = df.rename({"Occupancy Rate": "occ"})
    df = df.rename({"Inventory Units": "inventory"})
    df = df.rename({"Net Delivered Units 12 Mo": "delivered"})
    df = df.rename({"Population": "pop"})
    df = df.rename({"Demand Units": "demand_units"})
    df = df.rename({"Absorption %": "absorption"})
    df = df.rename({"Construction Starts Units 12 Mo": "starts"})
    df = df.rename({"CBSA Code": "cbsa"})
    df = df.rename({"Demolished Units": "demolished_units"})
    df = df.rename({"Sales Volume Transactions": "sales_volume"})

    df = df.select(
        [
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
            "starts",
            "demolished_units",
            "cbsa",
            "sales_volume",
        ]
    )

    df = df.with_columns(pl.col("year").cast(pl.Int16)).filter(
        pl.col("msa") != "New Orleans - LA"
    )

    top_n_msas = (
        df.filter(pl.col("year") == 2001)
        .sort("inventory", descending=True)
        .head(filter)
        .select("msa")
    )
    df = df.filter(pl.col("msa").is_in(top_n_msas))
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
    df = df.with_columns((pl.col("pop") / (pl.col("inventory"))).alias("RDI"))

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
            (pl.col("starts") / pl.col("inventory").shift(1).over("msa")).alias(
                "starts_pct"
            ),
            (100 * (pl.col("pop") / pl.col("pop").shift(1).over("msa") - 1)).alias(
                "pop_growth"
            ),
            (
                pl.col("demolished_units") / pl.col("inventory").shift(1).over("msa")
            ).alias("demolished_pct"),
            (pl.col("sales_volume") / pl.col("inventory").shift(1).over("msa")).alias(
                "sales_volume_growth"
            ),
        ]
    )

    df = df.with_columns(
        [
            pl.col("RDI").pct_change().over("msa").alias("RDI_growth"),
        ]
    )
    nyear = 5
    df = df.with_columns(
        (
            pl.col("real_rent_growth_next_year")
            - pl.col("real_rent_growth_next_year").median().over("year")
        ).alias("real_relative_rg_next_year"),
        (pl.col("occ") - pl.col("occ").shift(1).over("msa")).alias("occupancy_delta"),
        (pl.col("absorption") - pl.col("absorption").shift(1).over("msa")).alias(
            "absorption_delta"
        ),
        pl.col("RDI_growth")
        .rolling_sum(window_size=nyear, min_samples=nyear)
        .over("msa")
        .alias("RDI_growth_5yr"),
        pl.col("supply_growth")
        .rolling_sum(window_size=nyear, min_samples=nyear)
        .shift(-nyear)
        .over("msa")
        .alias("supply_5yr_fwd"),
    )
    df = df.with_columns(
        (
            pl.col("real_rent_growth")
            - pl.col("real_rent_growth").median().over("year")
        ).alias("real_relative_rent_growth")
    ).with_columns(
        pl.col("real_rent_growth")
        .rolling_sum(window_size=nyear, min_samples=nyear)
        .shift(-nyear)
        .over("msa")
        .alias("rrg_5yr_fwd")
    )

    df = df.with_columns(demand=pl.col("RDI_growth") > 0)
    migration = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "msa_migration.csv"
    ).with_columns(pl.col("value").cast(pl.Float64))
    migration = migration.with_columns(
        pl.col("costar_msa").str.replace(" USA$", "").alias("costar_msa")
    )
    migration = migration.filter(pl.col("series_name") == "pct_international_mig_")
    threshold = migration["value"].quantile(0.9)
    migration = migration.with_columns(
        (pl.col("value") >= threshold).cast(pl.Int8).alias("exog_shock")
    )
    df = df.join(
        migration.rename({"costar_msa": "msa"}), on=["msa", "year"], how="left"
    )
    preds = iv_model(df)
    df = df.join(
        preds.select(["msa", "year", "predicted_demand"]),
        on=["msa", "year"],
        how="left",
    )

    df = df.select(
        [
            "year",
            "absorption",
            "absorption_delta",
            "msa",
            "occ",
            "occupancy_delta",
            "pop",
            "demolished_pct",
            "rent_growth",
            "real_rentpsf",
            "real_rent_growth",
            "starts_pct",
            "inventory",
            "RDI",
            "real_rent_growth_next_year",
            "real_relative_rent_growth",
            "RDI_growth",
            "supply_growth",
            "real_relative_rg_next_year",
            "supply_5yr_fwd",
            "rrg_5yr_fwd",
            "RDI_growth_5yr",
            "pop_growth",
            "sales_volume_growth",
            "demand",
            "predicted_demand",
            "exog_shock",
        ]
    )
    df.write_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    return df


def iv_model(df):
    # Base model: contemporaneous exog_shock instrument
    df_clean = df.to_pandas().dropna(
        subset=[
            "real_relative_rg_next_year",
            "real_relative_rent_growth",
            "RDI_growth",
            "exog_shock",
            "pop_growth",
            "sales_volume_growth",
        ]
    )
    pred_demand = df_clean[["msa", "year"]]
    df_clean = pd.get_dummies(df_clean, columns=["msa", "year"], drop_first=True)

    y = df_clean["real_relative_rg_next_year"]
    X = df_clean[["pop_growth", "sales_volume_growth"]]
    endog = df_clean["RDI_growth"]
    instrument = df_clean["exog_shock"]
    X = sm.add_constant(X)

    print(
        "\n",
        "Running IV2SLS with contemporaneous exog_shock as instrument for RDI_growth",
        "\n",
    )
    iv_model = IV2SLS(dependent=y, exog=X, endog=endog, instruments=instrument)
    results = iv_model.fit()
    print(results.summary)
    pred_demand["predicted_demand"] = results.fitted_values

    print("\n", "Running placebo test: using same-year rent growth as outcome", "\n")
    y_placebo = df_clean["real_relative_rent_growth"]
    placebo_model = IV2SLS(
        dependent=y_placebo, exog=X, endog=endog, instruments=instrument
    )
    placebo_results = placebo_model.fit()
    print(placebo_results.summary)

    # Lagged instrument test
    df_lag = df.to_pandas()
    df_lag["exog_shock_lag"] = df_lag.groupby("msa")["exog_shock"].shift(1)
    df_lag = pd.get_dummies(df_lag, columns=["msa", "year"], drop_first=True)
    df_iv_lag = df_lag.dropna(
        subset=[
            "real_relative_rg_next_year",
            "RDI_growth",
            "exog_shock_lag",
            "pop_growth",
            "sales_volume_growth",
        ]
    )

    y = df_iv_lag["real_relative_rg_next_year"]
    X = df_iv_lag[["pop_growth", "sales_volume_growth"]]
    endog = df_iv_lag["RDI_growth"]
    instrument = df_iv_lag["exog_shock_lag"]
    X = sm.add_constant(X)

    print(
        "\n",
        "Running IV2SLS with lagged exog_shock (t-1) as instrument for RDI_growth",
        "\n",
    )
    iv_model_lag = IV2SLS(dependent=y, exog=X, endog=endog, instruments=instrument)
    results_lag = iv_model_lag.fit(cov_type="robust")
    print(results_lag.summary)
    return pl.DataFrame(pred_demand)


def predict_future(how, years=10):
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    if how == "naive":
        holder = []
        for year in range(2001 + years, 2024 - years):
            for msa in df["msa"].unique():
                df_train = df[
                    (df["year"] < year)
                    & (df["year"] >= year - years)
                    & (df["msa"] == msa)
                ].sort_values("year")
                y_hat = df_train["real_rent_growth"].sum()
                next_year_row = df[(df["year"] == year) & (df["msa"] == msa)]
                if not next_year_row.empty:
                    real_rent_growth_next_year = next_year_row[
                        "real_rent_growth_next_year"
                    ].values[0]
                    holder.append(
                        {
                            "year": year,
                            "msa": msa,
                            "y_hat": y_hat,
                            "real_rent_growth_next_year": real_rent_growth_next_year,
                        }
                    )
        holder = pd.DataFrame(holder)
        holder.to_csv(
            Path(__file__).resolve().parent.parent
            / "data"
            / f"naive_summary_{years}.csv",
            index=False,
        )
    if how == "ARIMA":
        summary = pd.DataFrame()
        holder = []
        for year in range(2001 + years, 2024 - years):
            for msa in df["msa"].unique():
                df_train = df[
                    (df["year"] < year)
                    # & (df["year"] >= year - years)
                    & (df["msa"] == msa)
                ].sort_values("year")
                y = df_train["real_rent_growth"].to_numpy()
                try:
                    model = ARIMA(y, order=(years, 0, 0)).fit()
                    y_hat = model.forecast(steps=years).sum()
                    holder.append(
                        {
                            "year": year + 1,
                            "msa": msa,
                            "y_hat": y_hat,
                            "real_rent_growth_next_year": df_train[
                                df_train["year"] == df_train["year"].max()
                            ]["real_rent_growth_next_year"].values[0],
                        }
                    )
                except Exception as e:
                    print(f"ARIMA failed for {msa} in {year}: {e}")
        holder = pd.DataFrame(holder)
        holder.to_csv(
            Path(__file__).resolve().parent.parent
            / "data"
            / f"arima_summary_{years}.csv",
            index=False,
        )

        return summary


def compare_predictions(quantiles=5, years=10):
    z = quantiles
    least = str(min(range(z)))
    top = str(max(range(z)))
    processed = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    processed = processed.sort_values(["msa", "year"])
    processed[f"real_rent_growth_{years}yr"] = processed.groupby("msa")[
        "real_relative_rent_growth"
    ].transform(
        lambda x: x.rolling(window=years, min_periods=years).sum().shift(-years)
    )

    arima_file = (
        Path(__file__).resolve().parent.parent / "data" / f"arima_summary_{years}.csv"
    )
    ar = pd.read_csv(arima_file)
    ar = ar.sort_values(["msa", "year"])
    ar["ARIMA_growth_group"] = ar.groupby("year")["y_hat"].transform(
        lambda x: pd.qcut(
            x,
            q=z,
            labels=[str(n) for n in range(z)],
        )
    )
    ar = ar[["year", "msa", "ARIMA_growth_group"]]
    ar = ar.merge(
        processed[["year", "msa", f"real_rent_growth_{years}yr"]],
        on=["year", "msa"],
        how="left",
    ).drop("msa", axis=1)
    ar = ar.groupby(["year", "ARIMA_growth_group"], observed=False).mean().reset_index()
    ar = ar.pivot(
        index="year", columns="ARIMA_growth_group", values=f"real_rent_growth_{years}yr"
    ).reset_index()
    ar["arima_spread"] = ar[top] - ar[least]
    ar = ar[["year", "arima_spread"]]

    naive_file = (
        Path(__file__).resolve().parent.parent / "data" / f"naive_summary_{years}.csv"
    )
    ol = pd.read_csv(naive_file).dropna(subset=["real_rent_growth_next_year", "y_hat"])
    ol["naive_growth_group"] = ol.groupby("year")["y_hat"].transform(
        lambda x: pd.qcut(x, q=z, labels=[str(n) for n in range(z)])
    )

    # Calculate average spread for each method
    # These will be used as labels in the plot
    ol = ol.merge(
        processed[["year", "msa", f"real_rent_growth_{years}yr"]],
        on=["year", "msa"],
        how="left",
    ).drop("msa", axis=1)
    ol = ol[["year", "naive_growth_group", f"real_rent_growth_{years}yr"]]
    ol = ol.groupby(["year", "naive_growth_group"], observed=False).mean().reset_index()
    ol = ol.pivot(
        index="year", columns="naive_growth_group", values=f"real_rent_growth_{years}yr"
    ).reset_index()
    ol["naive_spread"] = ol[top] - ol[least]
    ol = ol[["year", "naive_spread"]]

    rd = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    ).dropna(subset=["real_rent_growth_next_year", "RDI_growth"])
    rd["RDI_growth"] = rd.groupby("msa")["RDI_growth"].transform(
        lambda x: x.rolling(window=years, min_periods=years).sum()
    )
    rd = rd.dropna(subset=["real_rent_growth_next_year", "RDI_growth"])
    rd["RDI_growth_group"] = rd.groupby("year")["RDI_growth"].transform(
        lambda x: pd.qcut(x, q=z, labels=[str(n) for n in range(z)])
    )
    rd = rd[["year", "msa", "RDI_growth_group"]]
    rd = rd.merge(
        processed[["year", "msa", f"real_rent_growth_{years}yr"]],
        on=["year", "msa"],
        how="left",
    ).drop("msa", axis=1)
    rd = rd[["year", "RDI_growth_group", f"real_rent_growth_{years}yr"]]
    rd = rd.groupby(["year", "RDI_growth_group"], observed=False).mean().reset_index()
    rd = rd.pivot(
        index="year", columns="RDI_growth_group", values=f"real_rent_growth_{years}yr"
    ).reset_index()
    rd["spread"] = rd[top] - rd[least]
    rd = rd[["year", "spread"]]
    rd.columns = ["year", "rdi_spread"]
    rd = rd.merge(ar, on="year", how="left")
    rd = rd.merge(ol, on="year", how="left").dropna()

    avg_rdi_spread = rd["rdi_spread"].mean()
    avg_arima_spread = ar["arima_spread"].mean()
    avg_naive_spread = ol["naive_spread"].mean()
    # Plot the spreads over time
    plt.figure(figsize=(10, 6))
    plt.plot(rd["year"], rd["rdi_spread"], label="RDI Spread", marker="o")
    plt.plot(rd["year"], rd["arima_spread"], label="ARIMA Spread", marker="s")
    plt.plot(rd["year"], rd["naive_spread"], label="Naive Spread", marker="^")
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # Add labels, title, and legend
    plt.xlabel("Year")
    plt.ylabel(
        f"Difference in {years}-Year Real Rent Growth (Top Quartile - Bottom Quartile)"
    )
    plt.title(
        f"Forecast Performance: Difference in {years}-Year Real Rent Growth in Top Quartile - Bottom Quartile"
    )
    plt.gca().yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=1))
    # Add average spread as labels
    plt.text(
        rd["year"].iloc[-1] + 0.5,
        rd["rdi_spread"].iloc[-1],
        f"Total Average: {avg_rdi_spread:.2%}",
        color="C0",
        va="center",
        fontsize=10,
    )
    plt.text(
        rd["year"].iloc[-1] + 0.5,
        rd["arima_spread"].iloc[-1],
        f"Total Average: {avg_arima_spread:.2%}",
        color="C1",
        va="center",
        fontsize=10,
    )
    plt.text(
        rd["year"].iloc[-1] + 0.5,
        rd["naive_spread"].iloc[-1],
        f"Total Average: {avg_naive_spread:.2%}",
        color="C2",
        va="center",
        fontsize=10,
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the plot
    plt.savefig(
        Path(__file__).resolve().parent.parent
        / "Figs"
        / f"spread_comparison_over_time_{years}yr.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )

    # Remove min and max from each column and calculate the mean
    plt.show()


def simplify_anova():
    # Choose from 'real_rent_growth_next_year', 'real_relative_rg_next_year'
    y_var = "real_relative_rg_next_year"
    # y_var = "real_rent_growth_next_year"
    # Choose from 'occupancy_delta','absorption_delta','demand'
    x_var = "demand"
    # x_var = "occupancy_delta"
    # x_var = "absorption_delta"

    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    ).dropna(subset=y_var)
    # Calculate mean next-year real-rent growth
    summary = (
        df.groupby([x_var])
        .agg(
            **{
                f"mean_{y_var}": (y_var, "mean"),
                "n_obs": (y_var, "size"),
            }
        )
        .reset_index()
    )
    # Calculate means and 95% confidence intervals for each group
    group_means = df.groupby(x_var)[y_var].mean()
    group_se = df.groupby(x_var)[y_var].sem()
    ci_lower = group_means - 1.96 * group_se
    ci_upper = group_means + 1.96 * group_se
    # Calculate p-values for each group mean

    # Perform one-sample t-tests for each group mean
    p_values = {}
    for x in df[x_var].unique():
        x_var_data = df[df[x_var] == x][y_var]
        result = ttest_1samp(x_var_data, 0, nan_policy="omit")
        t_stat, p_value = result.statistic, result.pvalue
        p_values[x] = p_value
        # Add n_obs to the summary DataFrame
    summary["n_obs"] = df.groupby(x_var)[y_var].size().values
    # Add p-values to the summary DataFrame
    summary["p-value"] = summary[x_var].map(p_values)
    # Combine results into a DataFrame
    ci_summary = pd.DataFrame(
        {
            f"{x_var}": summary[x_var],
            "mean (bps)": [int(x * 10000) for x in group_means],
            "95% CI Lower (bps)": [int(x * 10000) for x in ci_lower],
            "95% CI Upper (bps)": [int(x * 10000) for x in ci_upper],
            "p-value": summary["p-value"].values,
            "n_obs": summary["n_obs"].values,
        }
    ).reset_index(drop=True)

    print(ci_summary)

    # Clustered standard errors by MSA
    model = smf.ols(f"{y_var} ~ C({x_var})", data=df).fit()

    # Prepare a DataFrame with the same structure for prediction
    group_means_df = df.groupby(x_var).mean(numeric_only=True).reset_index()
    group_means_df[x_var] = group_means_df[x_var].astype("category")

    # Predict group means and extract standard errors
    group_means = model.predict(group_means_df)
    group_se = model.get_robustcov_results().bse
    # Calculate 95% confidence intervals for the group means
    ci_lower = group_means - 1.96 * group_se[: len(summary)]
    ci_upper = group_means + 1.96 * group_se[: len(summary)]
    summary["95% CI"] = list(zip(ci_lower.round(4) * 10000, ci_upper.round(4) * 10000))
    summary["mean (bps)"] = (summary[f"mean_{y_var}"] * 10000).astype(int)
    summary["se (bps)"] = (group_se[: len(summary)] * 10000).astype(
        int
    )  # Match SEs to groups

    # Two-way ANOVA
    anova_model = smf.ols(f"{y_var} ~ C({x_var})", data=df).fit()
    anova_table = sm.stats.anova_lm(anova_model, typ=2)

    # Tukey HSD post-hoc test
    tukey = pairwise_tukeyhsd(
        endog=df[df[x_var].isin([True, False])][y_var],
        groups=df[df[x_var].isin([True, False])][x_var],
        alpha=0.05,
    )

    # Format results into a table block
    print("\nMean Next-Year Real-Rent Growth and Standard Errors:")
    print(
        summary[[x_var, "mean (bps)", "se (bps)", "95% CI", "n_obs"]]
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
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    holder = []
    for event in [
        "true",
        "false",
        "same",
    ]:
        df = original.with_columns(
            pl.when(pl.col("demand") == pl.col("demand").shift(1).over("msa"))
            .then(pl.lit("same"))
            .otherwise(pl.lit("switch"))
            .alias("transition"),
            pl.col("demand").shift(1).over("msa").alias("prev_group"),
        ).drop_nulls(subset="prev_group")
        if event != "same":

            event_years = (
                df.filter(pl.col("transition").shift(-1).over("msa") == "same")
                .filter(pl.col("transition").shift(-2).over("msa") == "same")
                # .filter(pl.col("transition").shift(1).over("msa") == "same")
                # .filter(pl.col("transition").shift(2).over("msa") == "same")
                .filter(pl.col("transition") == "switch")
                .filter(pl.col("demand") == event)
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
                [
                    "msa",
                    "transition",
                    "event_time",
                    "real_relative_rent_growth",
                ]
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
        if event == "false":
            color = "orange"
        elif event == "true":
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
    for event in ["true", "false", "same"]:
        event_data = holder[holder["event"] == event]

        # Calculate means for two years before and after the event
        before_means = event_data[["rr_rent_growth_m1", "rr_rent_growth_m2"]].mean()
        after_means = event_data[
            ["rr_rent_growth_0", "rr_rent_growth_p1", "rr_rent_growth_p2"]
        ].mean()

        # Perform t-tests for differences in means
        t_stat, p_value = ttest_ind(
            event_data[["rr_rent_growth_m1", "rr_rent_growth_m2"]].values.flatten(),
            event_data[
                ["rr_rent_growth_0", "rr_rent_growth_p1", "rr_rent_growth_p2"]
            ].values.flatten(),
            nan_policy="omit",
        )

        results.append(
            {
                "Event": event,
                "Mean Before": int(before_means.mean() * 10000),
                "Mean After": int(after_means.mean() * 10000),
                "Difference": int(
                    after_means.mean() * 10000 - before_means.mean() * 10000
                ),
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
    cbsa2fips = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "cbsa2fipsxw.csv"
    )
    cbsa2fips = (
        cbsa2fips.with_columns(
            pl.col("fipscountycode").cast(pl.Utf8).str.zfill(2),
            pl.col("fipsstatecode").cast(pl.Utf8).str.zfill(3),
            (pl.col("fipsstatecode") + pl.col("fipscountycode")).alias("FIPS"),
        )
        .select(["FIPS", "cbsacode"])
        .unique()
    )
    print(df.shape)
    df = df.join(cbsa2fips, left_on="cbsa", right_on="cbsacode", how="left")
    print(df.shape)
    assert False
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
    varx = "RDI_growth"
    vary = "real_rent_growth"
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    ).dropna(subset=[varx, vary])
    ds = (
        df.groupby(["year"])
        .agg(
            {
                varx: "mean",
                vary: "mean",
            }
        )
        .reset_index()
    )
    plt.figure(figsize=(12, 8))
    plt.scatter(
        ds[varx],
        ds[vary],
    )
    for i, row in ds.iterrows():
        plt.text(
            row[varx],
            row[vary],
            str(int(row["year"])),
            fontsize=8,
            ha="right",
            va="bottom",
            color="black",
        )
    plt.xlabel("Δ RDI (2001-2024)")
    plt.ylabel("Δ Rent Growth (2001-2024)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")
    plt.grid()
    plt.tight_layout()
    slope, intercept, r_value, p_value, std_err = linregress(ds[varx], ds[vary])
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
    x_vals = np.linspace(ds[varx].min(), ds[varx].max(), 100)
    y_vals = slope * x_vals + intercept
    plt.plot(
        x_vals, y_vals, color="black", label=f"Line of Best Fit (R²={r_value**2:.2f})"
    )
    # Calculate 95% confidence interval
    y_pred = slope * ds[varx] + intercept
    residuals = ds[vary] - y_pred
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
    plt.title("Mean Change in RDI vs. Mean Real Rent Growth Change by Year")
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
    var = "real_relative_rg_next_year"
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    ).dropna(subset=var)
    df["year"] = df["year"] + 1
    df["demand"] = df["demand"].astype(str)
    df_pivot = df.pivot_table(
        index="year", columns="demand", values=var, aggfunc="mean"
    )
    # Calculate the overall average by year
    df_avg = df.groupby("year")[var].mean()

    # Plot the averages
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_pivot.index,
        df_pivot["False"],
        label="Renter-Favorable Average",
        color="blue",
    )
    plt.plot(
        df_pivot.index,
        df_pivot["True"],
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


def plot_rdi_positive_counts_vs_rent_growth():
    # Load preprocessed data
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    ).dropna(subset=["RDI_growth", "real_relative_rg_next_year"])
    # Filter for years 2001-2011 for RDI_growth and 2012-2021 for rent growth
    msas = df["msa"].unique()
    results = []
    for msa in msas:
        for year in range(2011, 2015):
            msa_df = df[df["msa"] == msa]
            rdi_count = msa_df[
                (msa_df["year"] >= year - 10)
                & (msa_df["year"] < year)
                & (msa_df["RDI_growth"] >= 0)
            ].shape[0]
            rent_growth_sum = msa_df[
                (msa_df["year"] >= year) & (msa_df["year"] < year + 10)
            ]["real_relative_rent_growth"].mean()
            rdi_years = msa_df[(msa_df["year"] >= year - 10) & (msa_df["year"] < year)][
                "year"
            ].unique()
            rent_growth_years = msa_df[
                (msa_df["year"] >= year) & (msa_df["year"] < year + 10)
            ]["year"].unique()
            assert (
                len(rdi_years) == 10
            ), f"Expected 10 unique years in rdi_count, got {len(rdi_years)}"
            assert (
                len(rent_growth_years) == 10
            ), f"Expected 10 unique years in rent_growth_sum, got {len(rent_growth_years)}"
            assert set(rdi_years).isdisjoint(
                rent_growth_years
            ), "Years in rdi_count and rent_growth_sum overlap"
            results.append(
                {
                    "msa": msa,
                    "rdi_positive_count": rdi_count,
                    "rent_growth_sum": rent_growth_sum,
                }
            )
    results_df = pd.DataFrame(results)
    grouped = results_df.groupby("rdi_positive_count")["rent_growth_sum"].apply(list)
    labels = [str(k) for k in grouped.index]
    plt.figure(figsize=(10, 6))
    plt.boxplot(grouped, tick_labels=labels, showmeans=True)
    # Add n = count above each box
    for i, label in enumerate(labels):
        n = len(grouped.iloc[i])
        plt.text(
            i + 1,  # boxplot x positions are 1-based
            max(grouped.iloc[i]) if len(grouped.iloc[i]) > 0 else 0,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )
    plt.xlabel("Count of Years with RDI_growth > 0; Trailing 10 Years 2001-2010")
    plt.ylabel("Sum of Real Rent Growth Next 10 Years 2011-2020")
    plt.title(
        "Positive RDI Growth as an Indicator of Future Rent Growth: 10 Year Window"
    )
    plt.grid(True, axis="y")
    plt.tight_layout()

    msas = df["msa"].unique()
    results = []
    for msa in msas:
        for year in range(2006, 2020):
            msa_df = df[df["msa"] == msa]
            rdi_count = msa_df[
                (msa_df["year"] >= year - 5)
                & (msa_df["year"] < year)
                & (msa_df["RDI_growth"] > 0)
            ].shape[0]
            rent_growth_sum = msa_df[
                (msa_df["year"] >= year) & (msa_df["year"] < year + 5)
            ]["real_relative_rent_growth"].mean()
            # Assert that the number of unique years in rdi_count == 5 and in rent_growth_sum == 5, and that the years don't overlap
            rdi_years = msa_df[(msa_df["year"] >= year - 5) & (msa_df["year"] < year)][
                "year"
            ].unique()
            rent_growth_years = msa_df[
                (msa_df["year"] >= year) & (msa_df["year"] < year + 5)
            ]["year"].unique()
            assert (
                len(rdi_years) == 5
            ), f"Expected 5 unique years in rdi_count, got {len(rdi_years)}"
            assert (
                len(rent_growth_years) == 5
            ), f"Expected 5 unique years in rent_growth_sum, got {len(rent_growth_years)}"
            assert set(rdi_years).isdisjoint(
                rent_growth_years
            ), "Years in rdi_count and rent_growth_sum overlap"
            results.append(
                {
                    "msa": msa,
                    "rdi_positive_count": rdi_count,
                    "rent_growth_sum": rent_growth_sum,
                }
            )
    results_df = pd.DataFrame(results)
    grouped = results_df.groupby("rdi_positive_count")["rent_growth_sum"].apply(list)
    filtered = grouped[grouped.apply(lambda x: len(x) > 0)]
    data = [filtered[k] for k in filtered.index]
    labels = [str(k) for k in filtered.index]
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.xlabel("Count of Years with RDI_growth > 0; Trailing 5 Years 2005-2017")
    plt.ylabel("Sum of Real Rent Growth Next 5 Years 2012-2021")
    plt.title(
        "Positive RDI Growth as an Indicator of Future Rent Growth: 5 Year Window"
    )
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def analyze_delta_vs_rent_growth():
    preprocessed_csv = (
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = pd.read_csv(preprocessed_csv).dropna(
        subset=["RDI_growth", "supply_growth", "real_relative_rg_next_year"]
    )
    # df["delta"] = df["predicted_demand"]
    df["supply_growth"] = df.groupby("msa")["supply_growth"].shift(-1)
    df["delta"] = df["RDI_growth"] - df["supply_growth"]
    results = []
    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year]
        top10 = year_df[year_df["delta"] >= 0]["real_rent_growth_next_year"].mean()
        bottom10 = year_df[year_df["delta"] < 0]["real_rent_growth_next_year"].mean()
        # top10 = year_df[year_df["delta"] >= 0]["real_relative_rg_next_year"].mean()
        # bottom10 = year_df[year_df["delta"] < 0]["real_relative_rg_next_year"].mean()
        # top10 = year_df.nlargest(20, "RDI_growth")["real_relative_rg_next_year"].mean()
        # bottom10 = year_df.nsmallest(20, "RDI_growth")[
        #     "real_relative_rg_next_year"
        # ].mean()
        results.append(
            {
                "year": year,
                "rdi_gt_supply": top10,
                "rdi_lt_supply": bottom10,
            }
        )

    results_df = pd.DataFrame(results)
    results_df["delta"] = results_df["rdi_gt_supply"] - results_df["rdi_lt_supply"]
    print(results_df)
    print(results_df.dropna().mean())
    return results_df


# TODO also exhibits on how well this works with filter = 200 and
# TODO compare trailing 10 rdi growth with future 10 supply growth and show that those with rdi_growth-supply_growth>0 have highest rent_growth
# TODO show trailing rdi growth more correlated with rent growth than population (trailing? current?)
# df = get_data(filter=100).to_pandas()
# plot_group_averages_with_confidence()
# simplify_anova()
"""
ANOVA of the difference in rent growth in the groups in the following year
"""
# event_study()
"""
Showing the results of switching to a RDI positive/negative segment and 
showing the rent after switching to True is significantly higher
than the rent after switching to false
"""
# show_summary_statistics()
# plot_phoenix_supply_demand()
# plot_austin_supply_demand()
# plot_national_averages()
# choropleth_rdi_by_msa()
# predict_future(how="naive", years=1)
# summary = predict_future(how="ARIMA", years=1)
# compare_predictions(quantiles=4, years=10)
"""
Comparison of 10-year predictions of rent growth using RDI, ARIMA and naive methods
"""
plot_rdi_positive_counts_vs_rent_growth()
"""
Looking at the count of years with RDI growth > 0 over 5 year and 10 year 
horizons as predictive of the next 5 and 10 years of rent growth
"""
# analyze_delta_vs_rent_growth()
"""
When using the RDI with supply growth and comparing >0 and <0
"""
