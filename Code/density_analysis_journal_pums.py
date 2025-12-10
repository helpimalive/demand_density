import pandas as pd
from scipy.stats import linregress
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


def get_data(filter_number=100):
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

    # Load CPI and calculate cumulative inflation adjustment
    cpi = pl.read_csv(Path(__file__).resolve().parent.parent / "data" / "cpi.csv")
    # cpi = pl.read_csv(
    #     Path(__file__).resolve().parent.parent / "data" / "cpi_ex_shelter.csv"
    # )
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

    # To calculate RDI using costar data
    # df = df.with_columns((pl.col("pop") / (pl.col("inventory"))).alias("RDI"))

    rdi = pl.read_csv(
        r"C:\Users\mlarriva\OneDrive - Brookfield\Documents\Github\demand_density\Data\pums_data\puma_metro_pop_density.csv"
    )
    rdi = rdi.with_columns(
        RDI=(pl.col("POPULATION_RENTED")) / pl.col("HOUSEHOLDS_RENTED")
    )
    rdi = rdi.with_columns(pl.col("costar_name").str.strip_suffix(" USA"))
    # Filter out MSAs that do not have the maximum record count
    msas = (
        rdi.drop_nulls("costar_name")
        .group_by("costar_name")
        .len()
        .filter(pl.col("len") == pl.col("len").max())
        .select("costar_name")
        .to_series()
        .to_list()
    )
    rdi = rdi.filter(pl.col("costar_name").is_in(msas))
    df = df.join(
        rdi, how="inner", left_on=["msa", "year"], right_on=["costar_name", "YEAR"]
    )
    top_n_msas = (
        (
            df.filter(pl.col("year") == 2005)
            .sort("inventory", descending=True)
            .head(filter_number)
            .select("msa")
        )
        .to_series()
        .to_list()
    )
    df = df.filter(pl.col("msa").is_in(top_n_msas))
    print(f"Number of unique MSAs: {df['msa'].n_unique()}")
    df = df.sort("msa", "year")
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
            # (pl.col("RDI") / pl.col("occ")).alias("RDI"),
        ]
    )

    df = df.sort(["msa", "year"]).with_columns(
        [
            pl.col("RDI").pct_change().over("msa").alias("RDI_growth"),
        ]
    )
    nyear = 5
    df = df.sort("msa", "year").with_columns(
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
        .alias("rrg_5yr_fwd"),
        pl.col("real_relative_rent_growth")
        .rolling_sum(window_size=nyear, min_samples=nyear)
        .shift(-nyear)
        .over("msa")
        .alias("rrrg_5yr_fwd"),
    )
    df = df.with_columns(demand=pl.col("RDI_growth") > 0)

    nat = (
        pl.read_csv(Path(__file__).resolve().parent.parent / "data" / "natality.csv")
        .with_columns(pl.col("County Code").cast(pl.Utf8).str.zfill(5).alias("FIPS"))
        .select("FIPS", "Year", "Births")
    ).with_columns((pl.col("Year").cast(pl.Int16) + 20).alias("year_20"))
    cbsa = (
        pl.read_csv(
            Path(__file__).resolve().parent.parent / "data" / "cbsa2fipsxw.csv",
            schema_overrides={"FIPS": pl.Utf8, "cbsajoin": pl.Utf8},
        )
        .with_columns(pl.col("FIPS").str.zfill(5))
        .select(["FIPS", "costar_msa"])
        .unique()
    )
    nat = (
        nat.join(cbsa, on="FIPS", how="inner")
        .select(
            pl.col("costar_msa").alias("msa"),
            pl.col("Births").alias("births_20y_ago"),
            pl.col("year_20"),
        )
        .group_by(["msa", "year_20"])
        .agg(pl.col("births_20y_ago").sum().alias("births_20y_ago"))
        .sort(["msa", "year_20"])
    )
    df = df.join(
        nat, right_on=["msa", "year_20"], left_on=["msa", "year"], how="left"
    ).with_columns(
        # pct_twty_yold=(pl.col("births_20y_ago") / pl.col("pop") > 0.01).cast(pl.Int8)
        pct_twty_yold=(pl.col("births_20y_ago") / pl.col("pop"))
    )
    # df = df.with_columns(pct_twty_yold=pl.col("pct_twty_yold"))
    # Print the percent of entries that have pct_twty_yold == 1
    percent_pct_twty_yold = (
        100 * df.filter(pl.col("pct_twty_yold") == 1).height / df.height
    )
    print(f"Percent of entries with pct_twty_yold == 1: {percent_pct_twty_yold:.2f}%")
    preds = iv_model(df)
    df = df.join(
        preds.select(["msa", "year", "predicted_demand"]),
        on=["msa", "year"],
        how="left",
    )

    df = df.select(
        [
            "year",
            "msa",
            "HOUSEHOLDS_RENTED",
            "POPULATION_RENTED",
            "POPULATION_OWNED",
            "HOUSEHOLDS_OWNED",
            "RDI",
            "RDI_growth",
            "real_relative_rg_next_year",
            "real_rent_growth_next_year",
            "rrg_5yr_fwd",
            "rrrg_5yr_fwd",
            "absorption",
            "absorption_delta",
            "occ",
            "occupancy_delta",
            "pop",
            "demolished_pct",
            "real_relative_rent_growth",
            "rent_growth",
            "real_rentpsf",
            "real_rent_growth",
            "supply_growth",
            "supply_5yr_fwd",
            "RDI_growth_5yr",
            "pop_growth",
            "sales_volume_growth",
            "demand",
            "predicted_demand",
            "pct_twty_yold",
            # "pct_international_mig_",
            "starts_pct",
            "inventory",
        ]
    )
    df.write_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = (
        df.select(
            [
                "msa",
                "year",
                "pop",
                "RDI",
                "RDI_growth",
                "real_rent_growth",
                "real_relative_rent_growth",
                "real_rent_growth_next_year",
                "real_relative_rg_next_year",
            ]
        )
        .drop_nulls()
        .with_columns(spread=pl.col("real_relative_rent_growth") - pl.col("RDI_growth"))
    )
    df.write_csv(Path(__file__).resolve().parent.parent / "data" / "sample_data.csv")
    return df


def iv_model(df):
    # Base model: contemporaneous exog instrument
    df_clean = df.to_pandas().dropna(
        subset=[
            "real_relative_rg_next_year",
            "real_relative_rent_growth",
            "RDI",
            "pct_twty_yold",
            "pop_growth",
            "sales_volume_growth",
        ]
    )

    # df_clean = df_clean[df_clean["year"].isin([2015, 2016, 2017, 2018,2019,])]
    pred_demand = df_clean[["msa", "year"]]
    # Calculate household formation: year-over-year difference by msa of sum(HOUSEHOLDS_RENTED, HOUSEHOLDS_OWNED)
    df_clean["household_formation"] = (
        df_clean["HOUSEHOLDS_RENTED"] + df_clean["HOUSEHOLDS_OWNED"]
    )
    df_clean["household_formation"] = df_clean.groupby("msa")[
        "household_formation"
    ].pct_change()
    df_clean = pd.get_dummies(df_clean, columns=["msa", "year"], drop_first=True)
    y = df_clean["real_relative_rg_next_year"]
    X = df_clean[["pop_growth", "starts_pct"]]
    endog = df_clean["RDI"]
    instrument = df_clean["pct_twty_yold"]
    X = sm.add_constant(X)

    print(
        "\n",
        "Running IV2SLS with the pct of 20 yos as the exog instrument for RDI",
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
    df_lag["pct_twty_yold_lag"] = df_lag.groupby("msa")["pct_twty_yold"].shift(1)
    df_lag = pd.get_dummies(df_lag, columns=["msa", "year"], drop_first=True)
    df_iv_lag = df_lag.dropna(
        subset=[
            "real_relative_rg_next_year",
            "RDI",
            "pct_twty_yold_lag",
            "pop_growth",
            "occupancy_delta",
            "starts_pct",
        ]
    )

    y = df_iv_lag["real_relative_rg_next_year"]
    X = df_iv_lag[["pop_growth", "starts_pct"]]
    endog = df_iv_lag["RDI"]
    instrument = df_iv_lag["pct_twty_yold_lag"]
    X = sm.add_constant(X)

    print(
        "\n",
        "Running IV2SLS with lagged pct_twty_yold (t-1) as instrument for RDI_growth",
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
        for year in range(2001 + years, 2024):
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
    ].transform(lambda x: x.rolling(window=years).sum().shift(-years))
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
    ).dropna(subset=["real_rent_growth_next_year", "RDI"])
    rd["RDI"] = rd.groupby("msa")["RDI"].transform(
        lambda x: x.rolling(window=years, min_periods=int(years / 2)).sum()
    )
    rd = rd.dropna(subset=["real_rent_growth_next_year", "RDI"])
    rd["RDI_group"] = rd.groupby("year")["RDI"].transform(
        lambda x: pd.qcut(x, q=z, labels=[str(n) for n in range(z)])
    )
    rd = rd[["year", "msa", "RDI_group"]]
    rd = rd.merge(
        processed[["year", "msa", f"real_rent_growth_{years}yr"]],
        on=["year", "msa"],
        how="left",
    ).drop("msa", axis=1)

    rd = rd[["year", "RDI_group", f"real_rent_growth_{years}yr"]]
    rd = rd.groupby(["year", "RDI_group"], observed=False).mean().reset_index()
    rd = rd.pivot(
        index="year", columns="RDI_group", values=f"real_rent_growth_{years}yr"
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
    y_var = "real_rent_growth_next_year"
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    ).dropna(subset=[y_var, "RDI_growth"])

    print("Demand = RDI > median")
    df["RDI_level"] = df["RDI"] > df["RDI"].median()
    df["RDI_delta"] = df["RDI_growth"] > 0


def simplify_anova():
    y_var = "real_rent_growth_next_year"
    y_var = "rrg_5yr_fwd"
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    ).dropna(subset=[y_var, "RDI_growth", "RDI"])
    df[y_var] = df[y_var] * 10000
    # Map to descriptive labels
    df["RDI_level"] = np.where(
        df["RDI"] > df["RDI"].median(), "Above Median RDI", "Below Median RDI"
    )
    df["RDI_delta"] = np.where(
        df["RDI_growth"] > 0, "RDI Growth Positive", "RDI Growth Negative"
    )
    df["Combined_Group"] = df["RDI_level"] + " & " + df["RDI_delta"]

    # ANOVA for RDI_level
    print("\nANOVA: Real Rent Growth Next Year by RDI Level")
    model_level = ols(f"{y_var} ~ C(RDI_level)", data=df).fit()
    print(sm.stats.anova_lm(model_level, typ=2))

    # ANOVA for RDI_delta
    print("\nANOVA: Real Rent Growth Next Year by RDI Growth Direction")
    model_delta = ols(f"{y_var} ~ C(RDI_delta)", data=df).fit()
    print(sm.stats.anova_lm(model_delta, typ=2))

    # ANOVA for combined groups
    print("\nANOVA: Real Rent Growth Next Year by Combined RDI Level & Growth")
    model_combined = ols(f"{y_var} ~ C(Combined_Group)", data=df).fit()
    print(sm.stats.anova_lm(model_combined, typ=2))

    # Group means for exhibit
    print("\nGroup Means by RDI Level:")
    print(df.groupby("RDI_level")[y_var].mean().rename("Mean Rent Growth Next Year"))
    print("\nGroup Means by RDI Growth Direction:")
    print(df.groupby("RDI_delta")[y_var].mean().rename("Mean Rent Growth Next Year"))
    print("\nGroup Means by Combined Group:")
    print(
        df.groupby("Combined_Group")[y_var].mean().rename("Mean Rent Growth Next Year")
    )


def event_study():
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    rdi_med = df.select(pl.col("RDI").median()).to_series()[0]
    # Section 1: Events where RDI switches from negative to positive (above median)
    events_pos = (
        df.sort(["msa", "year"])
        .with_columns(
            pl.col("RDI").shift(1).over("msa").alias("RDI_prior"),
            pl.col("RDI_growth").shift(-1).over("msa").alias("RDI_growth_after"),
            pl.col("RDI_growth").shift(1).over("msa").alias("RDI_growth_prior"),
            pl.col("RDI_growth").shift(2).over("msa").alias("RDI_growth_prior_2"),
            pl.col("RDI_growth").shift(3).over("msa").alias("RDI_growth_prior_3"),
        )
        .filter(
            (
                (pl.col("RDI_growth") < 0)
                & (pl.col("RDI_growth_prior") > 0)
                & (pl.col("RDI_growth_prior_2") > 0)
                & (pl.col("RDI_growth_prior_3") > 0)
            )
        )
        .select(["msa", "year"])
        .group_by(["msa"])
        .agg(pl.col("year").min())
    )

    # Section 2: Events where RDI switches from positive to negative (below median)
    events_neg = (
        df.sort(["msa", "year"])
        .with_columns(
            pl.col("RDI").shift(1).over("msa").alias("RDI_prior"),
            pl.col("RDI_growth").shift(1).over("msa").alias("RDI_growth_prior"),
            pl.col("RDI_growth").shift(2).over("msa").alias("RDI_growth_prior_2"),
            pl.col("RDI_growth").shift(3).over("msa").alias("RDI_growth_prior_3"),
        )
        .filter(
            (
                # (pl.col("RDI") < rdi_med)
                (pl.col("RDI_growth") > 0)
                & (pl.col("RDI_growth_prior") < 0)
                & (pl.col("RDI_growth_prior_2") < 0)
                & (pl.col("RDI_growth_prior_3") < 0)
            )
        )
        .select(["msa", "year"])
        .group_by(["msa"])
        .agg(pl.col("year").min())
    )

    # Calculate before/after averages for both event types
    def get_before_after(events, df_pd):
        before_avgs, after_avgs = [], []
        event_rows = events.to_pandas()
        for _, row in event_rows.iterrows():
            msa = row["msa"]
            event_year = row["year"]
            before = df_pd[
                (df_pd["msa"] == msa)
                & (df_pd["year"] >= event_year - 3)
                & (df_pd["year"] < event_year)
            ]["real_relative_rent_growth"].tolist()
            after = df_pd[
                (df_pd["msa"] == msa)
                & (df_pd["year"] > event_year)
                & (df_pd["year"] <= event_year + 3)
            ]["real_relative_rent_growth"].tolist()
            before_avgs.extend(before)
            after_avgs.extend(after)
        return before_avgs, after_avgs

    df_pd = df.to_pandas()
    before_pos, after_pos = get_before_after(events_pos, df_pd)
    print(after_pos)
    before_neg, after_neg = get_before_after(events_neg, df_pd)

    # Plot the results
    import matplotlib.pyplot as plt

    # Add data labels and confidence intervals to the bars
    def add_bar_labels(ax, data, errors):
        for i, (group_data, group_err) in enumerate(zip(data, errors)):
            for j, (val, err) in enumerate(zip(group_data, group_err)):
                ax.text(
                    j + (i - 0.5) * width - 0.1,
                    val + 0.001,
                    f"{val:.2%}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black",
                )
                # Add error bars (confidence intervals)
                ax.errorbar(
                    j + (i - 0.5) * width,
                    val,
                    yerr=err,
                    fmt="none",
                    ecolor="black",
                    capsize=4,
                    linewidth=1,
                )

    # Calculate confidence intervals (standard error * 1.96 for 95% CI)
    before_pos_err = 1.96 * sem(before_pos, nan_policy="omit")
    after_pos_err = 1.96 * sem(after_pos, nan_policy="omit")
    before_neg_err = 1.96 * sem(before_neg, nan_policy="omit")
    after_neg_err = 1.96 * sem(after_neg, nan_policy="omit")
    errors = [
        [before_pos_err, after_pos_err],
        [before_neg_err, after_neg_err],
    ]

    labels = ["Before", "After"]
    data = [
        [np.nanmean(before_pos), np.nanmean(after_pos)],
        [np.nanmean(before_neg), np.nanmean(after_neg)],
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x = np.arange(len(labels))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=1))
    add_bar_labels(ax, data, errors)
    ax.bar(
        x - width / 2,
        data[0],
        width,
        label="RDI reverses positive trend; market begins to de-densify; more rental households are formed",
        color="green",
    )
    ax.bar(
        x + width / 2,
        data[1],
        width,
        label="RDI reverses downward trend; market begins to densify; rental households consolidate",
        color="blue",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Avg. Real Relative Rent Growth (3 Year Avg.)")
    ax.set_title("Real Relative Rent Growth Before and After RDI Switch Events")
    ax.legend()
    # plt.tight_layout()
    plt.savefig(
        Path(__file__).resolve().parent.parent / "Figs" / "event_study.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.show()

    # Perform t-test for difference of means between before and after groups
    t_stat_pos, p_val_pos = ttest_ind(before_pos, after_pos, nan_policy="omit")
    t_stat_neg, p_val_neg = ttest_ind(before_neg, after_neg, nan_policy="omit")
    print(f"Positive RDI reversal: t={t_stat_pos:.3f}, p={p_val_pos:.3g}")
    print(f"Negative RDI reversal: t={t_stat_neg:.3f}, p={p_val_neg:.3g}")
    # Test if means are significantly different from zero (one-sample t-test)
    for label, values in zip(
        ["Before Positive", "After Positive", "Before Negative", "After Negative"],
        [before_pos, after_pos, before_neg, after_neg],
    ):
        t_stat, p_val = ttest_1samp(values, 0, nan_policy="omit")
        print(f"{label}: mean={np.nanmean(values):.4f}, t={t_stat:.3f}, p={p_val:.3g}")


def choropleth_rdi_by_msa():
    # Load the supply_demand_annual.csv file
    cbsa2fips = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "cbsa2fipsxw.csv",
        dtypes={"FIPS": pl.Utf8, "cbsajoin": pl.Utf8},
    ).with_columns(pl.col("FIPS").str.zfill(5))
    cbsa2fips = cbsa2fips.select(["FIPS", "cbsajoin"]).unique()
    df = pl.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    costar = pl.read_excel(
        Path(__file__).resolve().parent.parent / "data" / "costar_raw.xlsx",
        schema_overrides={"CBSA Code": pl.Utf8, "Geography Name": pl.Utf8},
    ).select(["Geography Name", "CBSA Code"])
    costar = (
        costar.with_columns(
            pl.col("Geography Name").str.replace(" USA$", "").alias("msa"),
            pl.col("CBSA Code").alias("cbsa"),
        )
        .select(["msa", "cbsa"])
        .unique()
    )
    df = df.join(costar, on=["msa"], how="left")
    df = df.join(cbsa2fips, left_on="cbsa", right_on="cbsajoin", how="left")[
        ["msa", "year", "RDI", "FIPS", "cbsa"]
    ]
    print(df)
    df = df.to_pandas()
    df["RDI"] = df["RDI"]
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
        range_color=(df["RDI"].min(), df["RDI"].max()),
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
    plt.xlabel("←   Expanding        Δ RDI      Crowding    →")
    plt.ylabel("Real Rent Growth")
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
    plt.title("Mean Change in RDI vs. Mean Real Rent Growth Anually")
    plt.savefig(
        Path(__file__).resolve().parent.parent / "Figs" / "rdi_rent_growth_2024.pdf",
        format="pdf",
        bbox_inches="tight",  # crop extra white
        pad_inches=0.02,
    )
    plt.show()
    # Plot a histogram of the change in RDI
    plt.figure(figsize=(10, 6))
    plt.hist(
        df["RDI_growth"].dropna(), bins=30, color="blue", alpha=0.7, edgecolor="black"
    )
    plt.xlabel("Change in RDI (Δ RDI)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Change in RDI: All MSAs (2001-2024)")
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
    df_2001 = df[df["year"] == 2006]

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
    # Write summary statistics to CSV
    summary.to_csv(
        Path(__file__).resolve().parent.parent / "data" / "summary_statistics.csv"
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
    var = "rrrg_5yr_fwd"
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    ).dropna(subset=var)
    # Calculate the number of positive RDI_growth records by msa over the trailing 5 years
    df["demand"] = (
        df.sort_values(["msa", "year"])
        .groupby("msa")["RDI_growth"]
        .transform(
            lambda x: x.rolling(window=5, min_periods=5).apply(
                lambda y: (y > 0).sum(), raw=True
            )
        )
    )
    df_pivot = df.pivot_table(
        index="year", columns="demand", values=var, aggfunc="mean"
    )
    print(df_pivot, df_pivot.mean(skipna=True))
    # Calculate the overall average by year
    df_avg = df.groupby("year")[var].mean()
    # Plot the averages
    plt.figure(figsize=(10, 6))
    plt.plot(
        df_pivot.index,
        df_pivot[True],
        label="De-densifying",
        color="blue",
    )
    plt.plot(
        df_pivot.index,
        df_pivot[False],
        label="Densifying",
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
    percent_positive = (
        df.groupby("year")["RDI_growth"]
        .apply(lambda x: (x > 0).mean())
        .diff()
        .reset_index(name="percent_positive")
    )
    print(percent_positive.min(), percent_positive.max())
    avg_rent_growth = (
        df.groupby("year")["real_rent_growth_next_year"]
        .mean()
        .reset_index(name="avg_rent_growth")
    )
    summary = percent_positive.merge(avg_rent_growth, on="year").dropna()
    plt.figure(figsize=(8, 6))
    plt.scatter(summary["percent_positive"], summary["avg_rent_growth"], s=60)
    # Add line of best fit
    slope, intercept = np.polyfit(
        summary["percent_positive"], summary["avg_rent_growth"], 1
    )
    x_vals = np.linspace(
        summary["percent_positive"].min(), summary["percent_positive"].max(), 100
    )
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color="black", linestyle="--", label="Line of Best Fit")
    # Calculate R-squared
    y_pred = slope * summary["percent_positive"] + intercept
    y_true = summary["avg_rent_growth"]
    ss_res = np.sum((y_true - y_pred) ** 2)
    # Calculate beta (slope) and p-value
    beta, _, r_value, p_value, _ = linregress(
        summary["percent_positive"], summary["avg_rent_growth"]
    )
    # Annotate beta and p-value on the plot
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    plt.text(
        0.05,
        0.90,
        f"$R^2$ = {r_squared:.2f}\nβ = {beta:.2f}\np = {p_value:.2e}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )
    for _, row in summary.iterrows():
        plt.text(
            row["percent_positive"],
            row["avg_rent_growth"],
            str(int(row["year"])),
            fontsize=8,
            ha="right",
            va="bottom",
        )
    plt.xlabel("Year over Year Change in Percent of MSAs with Positive RDI Growth")
    plt.gca().xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1, decimals=0))
    plt.ylabel("Average Real Rent Growth Next Year(All MSAs)")
    plt.title(
        "Year over Year Change in Percent of MSAs with Positive RDI Growth vs. Avg. Real Rent Growth Next Year by Year"
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        Path(__file__).resolve().parent.parent / "Figs" / "national_rdi_pct.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.show()


def analyze_delta_vs_rent_growth():
    preprocessed_csv = (
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df = pd.read_csv(preprocessed_csv).dropna(
        subset=["RDI", "supply_growth", "real_relative_rg_next_year"]
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


def plot_max_supply_growth_vs_rent_growth():
    # Read the processed data
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    # For each MSA, keep the row with the maximum supply_growth
    df = df[df["year"] != 2020]
    idx = df.groupby("msa")["supply_growth"].idxmax()
    df_max = df.loc[idx].dropna(subset=["supply_growth", "real_relative_rg_next_year"])
    # Color by sign of real_relative_rg_next_year
    colors = df_max["real_relative_rg_next_year"].apply(
        lambda x: "green" if x >= 0 else "red"
    )
    print((df_max[df_max["real_relative_rg_next_year"] > 0].shape[0]) / df_max.shape[0])

    # plt.figure(figsize=(10, 6))
    plt.scatter(
        df_max["supply_growth"],
        df_max["real_relative_rg_next_year"],
        c=colors,
        edgecolor="k",
        alpha=0.7,
    )

    # Find min/max x and y points
    min_x_idx = df_max["supply_growth"].idxmin()
    max_x_idx = df_max["supply_growth"].idxmax()
    min_y_idx = df_max["real_relative_rg_next_year"].idxmin()
    max_y_idx = df_max["real_relative_rg_next_year"].idxmax()

    # Add labels for min/max x
    idx = min_x_idx
    row = df_max.loc[idx]
    plt.annotate(
        f"{row['msa']} {int(row['year'])}",
        (row["supply_growth"], row["real_relative_rg_next_year"]),
        textcoords="offset points",
        xytext=(5, 5),
        ha="left",
        va="top",
        fontsize=8,
        color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
    )
    idx = max_x_idx
    row = df_max.loc[idx]
    plt.annotate(
        f"{row['msa']} {int(row['year'])}",
        (row["supply_growth"], row["real_relative_rg_next_year"]),
        textcoords="offset points",
        xytext=(5, 5),
        ha="right",
        fontsize=8,
        color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
    )
    # Add line of best fit and R-squared
    slope, intercept = np.polyfit(
        df_max["supply_growth"], df_max["real_relative_rg_next_year"], 1
    )
    x_vals = np.linspace(
        df_max["supply_growth"].min(), df_max["supply_growth"].max(), 100
    )
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, color="black", linestyle="--", label="Line of Best Fit")

    # Calculate R-squared
    y_pred = slope * df_max["supply_growth"] + intercept
    y_true = df_max["real_relative_rg_next_year"]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Annotate R-squared on the plot
    plt.text(
        0.05,
        0.95,
        f"$R^2$ = {r_squared:.2f}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )
    # Add labels for min/max y (avoid duplicate if already labeled)
    for idx in [min_y_idx, max_y_idx]:
        if idx not in [min_x_idx, max_x_idx]:
            row = df_max.loc[idx]
            plt.annotate(
                f"{row['msa']} {int(row['year'])}",
                (row["supply_growth"], row["real_relative_rg_next_year"]),
                textcoords="offset points",
                xytext=(5, -10),
                ha="left",
                fontsize=8,
                color="black",
                arrowprops=dict(arrowstyle="->", color="black", lw=0.5),
            )
    plt.xlabel("Supply Growth as a Percentage of Inventory")
    plt.ylabel("Real Relative Rent Growth Year After Max Supply Growth")
    plt.title(
        "Real Rent Growth the year after each MSA experiences its max supply growth (2001-2023)"
    )
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    # plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(
        Path(__file__).resolve().parent.parent
        / "Figs"
        / "max_supply_growth_vs_rent_growth.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.show()


def plot_max_supply_growth_vs_RDI_growth():
    # Read the processed data
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    df["rdi_x_supply"] = df["RDI_growth"] + df["supply_growth"] / df["RDI"]
    xvar = "rdi_x_supply"
    yvar = "real_rent_growth_next_year"
    df = df.dropna(subset=[xvar, "supply_growth", yvar])
    idx = df.groupby("msa")["supply_growth"].idxmax()
    df_max = df.loc[idx]
    avg_y_by_x = df_max.groupby(df_max[xvar] > 0)[yvar].mean()
    df_max.to_csv(
        Path(__file__).resolve().parent.parent / "data" / "max_supply_growth_rdi.csv",
        index=False,
    )
    print("Average", yvar, "when", xvar, "> 0:", avg_y_by_x[True] * 10000)
    print("Average", yvar, "when", xvar, "<= 0:", avg_y_by_x[False] * 10000)

    avg_y_by_supply = df_max.groupby(
        df_max["supply_growth"] > df_max["supply_growth"].median()
    )[yvar].mean()
    print("Average", yvar, "when", xvar, "> median:", avg_y_by_supply[True] * 10000)
    print("Average", yvar, "when", xvar, "<= median:", avg_y_by_supply[False] * 10000)
    # T-test for difference of means between groups

    t_stat_rdi, p_val_rdi = ttest_ind(
        df_max[df_max[xvar] > 0][yvar],
        df_max[df_max[xvar] <= 0][yvar],
        nan_policy="omit",
    )
    t_stat_supply, p_val_supply = ttest_ind(
        df_max[df_max["supply_growth"] > df_max["supply_growth"].median()][yvar],
        df_max[df_max["supply_growth"] <= df_max["supply_growth"].median()][yvar],
        nan_policy="omit",
    )
    print(
        f"T-test for {xvar} > 0 vs <= 0: t={t_stat_rdi:.3f}, p={p_val_rdi:.3g}\n"
        f"T-test for supply_growth > median vs <= median: t={t_stat_supply:.3f}, p={p_val_supply:.3g}"
    )
    # Prepare data for bar graph: average yvar for each group with confidence intervals
    import matplotlib.pyplot as plt

    # Calculate means and standard errors for each group
    group_means = df_max.groupby(df_max[xvar] > 0)[yvar].mean()
    group_sems = df_max.groupby(df_max[xvar] > 0)[yvar].apply(
        lambda x: sem(x, nan_policy="omit")
    )
    supply_groups = df_max.groupby(
        df_max["supply_growth"] > df_max["supply_growth"].median()
    )[yvar].mean()
    supply_sems = df_max.groupby(
        df_max["supply_growth"] > df_max["supply_growth"].median()
    )[yvar].apply(lambda x: sem(x, nan_policy="omit"))

    # Bar plot
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_labels = [
        f"{xvar} > 0",
        f"{xvar} <= 0",
        "Supply Growth > Median",
        "Supply Growth <= Median",
    ]
    means = [
        group_means[True] * 10000,
        group_means[False] * 10000,
        supply_groups[True] * 10000,
        supply_groups[False] * 10000,
    ]
    errors = [
        1.96 * group_sems[True] * 10000,
        1.96 * group_sems[False] * 10000,
        1.96 * supply_sems[True] * 10000,
        1.96 * supply_sems[False] * 10000,
    ]
    ax.bar(
        bar_labels,
        means,
        yerr=errors,
        capsize=8,
        color=["green", "green", "blue", "blue"],
    )
    ax.set_ylabel("Average Real Rent Growth Next Year (basis points)")
    ax.set_title("Average Rent Growth by Group with 95% Confidence Intervals")
    plt.tight_layout()
    plt.savefig(
        Path(__file__).resolve().parent.parent
        / "Figs"
        / "bar_group_avg_rent_growth.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.show()


def orthogonal():
    # Read in processed_data.csv
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )

    # Rename columns to match the required names
    df = df.rename(
        columns={
            "pct_international_mig_": "foreign_migration_share",
            # Add more renames here if needed
        }
    )
    df["msa"] = df["msa"].str.replace("-", " ").str.strip()
    df["msa"] = df["msa"].str.replace(" ", "_")
    df = df.sort_values(["msa", "year"])
    df["rent_growth_lag"] = (
        df.sort_values(["msa", "year"]).groupby("msa")["real_rent_growth"].shift(1)
    )
    df_clean = df.dropna(
        subset=[
            "foreign_migration_share",
            "rent_growth_lag",
            "pop_growth",
            "sales_volume_growth",
        ]
    )
    df_fe = pd.get_dummies(df_clean, columns=["msa", "year"], drop_first=True)
    df_fe.columns = [col.replace(" ", "_") for col in df_fe.columns]
    controls = " + ".join(
        [
            col
            for col in df_fe.columns
            if col.startswith("msa_") or col.startswith("year_")
        ]
    )
    formula = f"foreign_migration_share ~ rent_growth_lag + pop_growth + sales_volume_growth + {controls}"

    # Step 5: Fit model
    model = smf.ols(formula=formula, data=df_fe).fit(cov_type="HC3")  # Robust SEs

    # Step 6: Print results
    print(model.summary())


def spillover():
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    pivot = df.pivot(index="year", columns="msa", values="RDI_growth")

    # Select example regional clusters
    msa_pairs = [
        ("New York - NY", "Northern New Jersey - NJ"),
        ("San Francisco - CA", "San Jose - CA"),
        ("Dallas-Fort Worth - TX", "Austin - TX"),
        ("Los Angeles - CA", "Inland Empire - CA"),
        ("Miami - FL", "Palm Beach - FL"),
    ]

    for msa1, msa2 in msa_pairs:
        corr = pivot[msa1].corr(pivot[msa2])
        print(f"{msa1} vs {msa2}: r = {corr:.2f}")


# choropleth_rdi_by_msa()
get_data(200)
# plot_national_averages()
plot_max_supply_growth_vs_RDI_growth()
# plot_max_supply_growth_vs_rent_growth()
# df = get_data(filter=100).to_pandas()
# plot_group_averages_with_confidence()
# show_summary_statistics()
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
# plot_phoenix_supply_demand()
# plot_austin_supply_demand()
# predict_future(how="naive", years=5)
# summary = predict_future(how="ARIMA", years=5)
# compare_predictions(quantiles=4, years=5)

"""
Comparison of 10-year predictions of rent growth using RDI, ARIMA and naive methods
"""
# plot_rdi_positive_counts_vs_rent_growth()
"""
Looking at the count of years with RDI growth > 0 over 5 year and 10 year 
horizons as predictive of the next 5 and 10 years of rent growth
"""
# analyze_delta_vs_rent_growth()
"""
When using the RDI with supply growth and comparing >0 and <0
"""
# orthogonal()
# spillover()
