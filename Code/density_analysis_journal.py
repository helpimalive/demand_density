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
    df = (
        df.filter(
            pl.col("msa").is_in(
                (
                    df.filter(pl.col("year") == 2001)
                    .sort(pl.col("inventory"), descending=True)
                    .head(100)
                    .select("msa")
                )
            )
        )
        .fill_nan(0)
        .fill_null(0)
    )
    # subm = pl.read_csv(
    #     r"C:\Users\mlarriva\OneDrive - FCP\Personal Folders\Documents\Supply Study\Data\submarket_supply_and_area_data.csv",
    #     infer_schema_length=10000,
    #     ignore_errors=True,
    # )
    # cols = [col_name.strip() for col_name in subm.columns]
    # subm.columns = cols
    # subm = (
    #     subm.with_columns(
    #         pl.col("Period").str.extract(r"(\d{4})").cast(pl.Int16).alias("year"),
    #         pl.col("SqMi").alias("size").cast(pl.Int32),
    #         pl.col("Geography Name").str.extract(r"^(.*?) USA - ").alias("costar_msa"),
    #     )
    #     .group_by("costar_msa")
    #     .agg(
    #         pl.col("size").mean(),
    #     )
    # ).drop_nulls()
    # df = df.join(subm, left_on="msa", right_on="costar_msa", how="left").filter(
    #     pl.col("size") > 0
    # )
    ##### Transformations
    ## CPI All Items
    cpi = pl.read_csv(Path(__file__).resolve().parent.parent / "data" / "cpi.csv")
    cpi = cpi.with_columns(pl.col("year").cast(pl.Int16))
    cpi = (
        cpi.filter(pl.col("year") > 1999).select(pl.col("year"), pl.col("cpi_pct"))
    ).with_columns(cpi_cum=(pl.col("cpi_pct") + 1).cum_prod())

    ## CPI Ex_Shelter
    # cpi = pl.read_csv(
    #     Path(__file__).resolve().parent.parent / "data" / "cpi_ex_shelter.csv"
    # )
    # cpi = cpi.with_columns(
    #     pl.col("DATE").str.strptime(pl.Date, "%m/%d/%Y").alias("date")
    # )
    # cpi = cpi.with_columns(pl.col("date").dt.year().cast(pl.Int16).alias("year"))
    # cpi = (
    #     cpi.filter(pl.col("year") > 1999)
    #     .with_columns((pl.col("CUSR0000SA0L2").cast(pl.Float32) / 100).alias("cpi_pct"))
    #     .select(pl.col("year"), pl.col("cpi_pct"))
    #     .sort("year")
    # ).with_columns(cpi_cum=(pl.col("cpi_pct") + 1).cum_prod())

    df = df.join(cpi, on="year", how="left")
    df = df.with_columns(
        (pl.col("rent_growth") - pl.col("cpi_pct")).alias("rent_growth"),
        (pl.col("rentpsf") / pl.col("cpi_cum")).alias("rentpsf"),
    )
    df = (
        df.sort("msa", "year", descending=False)
        .with_columns(
            ((pl.col("pop")) / (pl.col("inventory"))).alias("total_density"),
            (pl.col("delivered") / pl.col("inventory").shift(1).over("msa")).alias(
                "supply_growth"
            ),
            pl.col("rent_growth").shift(-1).over("msa").alias("rent_growth_next_year"),
            pl.col("rent_growth").shift(1).over("msa").alias("rent_growth_prior_year"),
            (pl.col("occ") - pl.col("occ").shift(1).over("msa")).alias("occ_growth"),
            (pl.col("demand_units").pct_change().over("msa")).alias(
                "demand_units_growth"
            ),
            (pl.col("absorption") - pl.col("absorption").shift(1).over("msa")).alias(
                "absorption_growth"
            ),
            ((pl.col("demand_units") / pl.col("inventory"))).alias("demand_pct"),
        )
        .sort("msa", "year", descending=False)
        .with_columns(
            (((pl.col("total_density").pct_change().over("msa")))).alias(
                "total_density_growth"
            )
        )
        .with_columns(implied_demand=pl.col("total_density_growth"))
    )
    # Excluding NOLA because Katrina made density go haywire
    # df = df.filter((pl.col("msa") != "New Orleans - LA") | (pl.col("year") != 2006))
    df.write_csv(
        Path(__file__).resolve().parent.parent / "data" / "preprocessed_data.csv"
    )
    return df


def density_growth_vs_rent_growth(df, cuts=4):
    df["density_quintile"] = df.groupby("year")["total_density_growth"].transform(
        lambda x: pd.qcut(x, cuts, labels=range(cuts))
    )
    df = df[df["supply_growth"] == df.groupby("msa")["supply_growth"].transform("max")]
    print(df.sort_values("supply_growth"))
    fig, axes = plt.subplots(1, len(df["density_quintile"].unique()) + 1)
    for i in range(cuts + 1):
        if i < cuts:
            sub_df = df[df["density_quintile"] == i]
        else:
            sub_df = df
        slope, intercept, r_value, p_value, std_err = linregress(
            sub_df["supply_growth"], sub_df["rent_growth"]
        )
        ax = axes[i]
        ax.plot(
            sub_df["supply_growth"],
            intercept + slope * sub_df["supply_growth"],
            label=f"Density Quintile {i} Implied D {sub_df['total_density_growth'].mean():.2%}",
        )
        ax.plot(sub_df["supply_growth"], sub_df["rent_growth"], "o")
        ax.plot(
            np.linspace(
                sub_df["supply_growth"].min(),
                sub_df["supply_growth"].max(),
                10,
            ),
            [sub_df["rent_growth_next_year"].mean()] * 10,
            linestyle=":",
        )
        if i < cuts:
            ax.set_title(
                f"Rank #{i+1} Implied Demand \n Rsquared: {r_value**2:.2f}",
                fontsize=11,
            )
            ax.legend()
        else:
            ax.set_title("All MSAs", fontsize=11)
        ax.set_ylim(-0.05, 0.15)
        ax.set_xlim(0, 0.15)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))

        if i == 0:
            ax.set_xlabel("MSA's all-time high Supply Growth (as a % of existing)")
            ax.set_ylabel("Rent Growth")
    fig.suptitle(
        "100 Largest MSAs: rent-growth response to supply shock \n ordered from lowest implied demand to greatest",
        fontsize=16,
    )
    # plt.tight_layout()
    plt.show()


def max_supply_growth(df):
    df = df[df["supply_growth"] == df.groupby("msa")["supply_growth"].transform("max")]
    fig, ax = plt.subplots(1, 1)
    ax.scatter(
        df[df["rent_growth"] >= 0]["supply_growth"],
        df[df["rent_growth"] >= 0]["rent_growth"],
        marker="+",
        label="MSA's rent and supply growth \n when its supply was at an all-time high",
        color="green",
    )
    ax.scatter(
        df[df["rent_growth"] < 0]["supply_growth"],
        df[df["rent_growth"] < 0]["rent_growth"],
        marker="_",
        label="MSA's rent and supply growth \n when its supply was at an all-time high",
        color="red",
    )
    slope, intercept, r_value, p_value, std_err = linregress(
        df["supply_growth"], df["rent_growth"]
    )
    ax.plot(
        df["supply_growth"],
        intercept + slope * df["supply_growth"],
        label="Line of Best Fit",
        color="black",
    )
    ax.plot(
        np.linspace(df["supply_growth"].min(), df["supply_growth"].max(), 20),
        [df["rent_growth"].mean()] * 20,
        label="Average",
        color="gray",
        linestyle="--",
    )
    ax.set_xlabel("Supply Growth as a % of Existing Inventory")
    ax.set_ylabel("YoY Effective Rent Growth")
    ax.set_title(
        f"Rent Growth Following All-Time High Supply in the 100 Largest Markets \n R-squared {r_value**2:.2f}"
    )
    positive_rent = df[df["rent_growth"] > 0].shape[0]
    negative_rent = df[df["rent_growth"] < 0].shape[0]
    print(f"Number of markets with positive rent growth: {positive_rent}")
    print(f"Number of markets with negative rent growth: {negative_rent}")

    ids = []
    ids.append(df["rent_growth"].idxmax())
    ids.append(df["rent_growth"].nlargest(2).index[1])
    ids.append(df["rent_growth"].nlargest(3).index[1])
    ids.append(df["supply_growth"].idxmax())
    for outlier_index in ids:
        outlier_x = df.loc[outlier_index, "supply_growth"] - 0.01
        outlier_y = df.loc[outlier_index, "rent_growth"] - 0.01
        if outlier_x > 0.1:
            outlier_x -= 0.025
        if outlier_y < 0:
            outlier_y -= 0.005
        label = f"{df.loc[outlier_index, 'msa']} - {df.loc[outlier_index, 'year']}"
        ax.annotate(
            label,
            xy=(outlier_x, outlier_y),
            xytext=(outlier_x, outlier_y),  # Offset the label slightly
            # arrowprops=dict(facecolor="red", shrink=0.05),
        )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    plt.show()


def max_supply_min_rent(df):
    df["max_supply_growth"] = df.groupby("msa")["supply_growth"].transform("max")
    df["min_rent_growth"] = df.groupby("msa")["rent_growth"].transform("min")
    df = df[df["supply_growth"] == df["max_supply_growth"]]
    df = df[df["rent_growth"] == df["min_rent_growth"]]
    print(df)


def max_supply_max_rent(df):
    df["max_supply_growth"] = df.groupby("msa")["supply_growth"].transform("max")
    df["max_rent_growth"] = df.groupby("msa")["rent_growth"].transform("max")
    df = df[df["supply_growth"] == df["max_supply_growth"]]
    df = df[df["rent_growth"] == df["max_rent_growth"]]
    print(df)


def plot_what(var_x, var_y, df):
    fig, ax = plt.subplots()
    df = df.dropna()
    slope, intercept, r_value, p_value, std_err = linregress(df[var_x], df[var_y])
    ax.scatter(df[var_x], df[var_y])
    ax.plot(df[var_x], intercept + slope * df[var_x], color="red")
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.set_title(f"{var_x} vs {var_y} Rsquare: {r_value**2:.2f}")
    return plt


def pos_neg_supply_rent(var_x, var_y, df):
    fig, ax = plt.subplots()
    slope, intercept, r_value, p_value, std_err = linregress(df[var_x], df[var_y])
    ax.plot(
        df[var_x],
        intercept + slope * df[var_x],
        color="black",
        label="line_of_best_fit",
    )
    # ax.plot(
    #     np.linspace(df[var_x].min(), df[var_x].max(), 20),
    #     [df[var_y].mean()] * 20,
    #     color="grey",
    #     linestyle="--",
    #     label="average",
    # )
    ax.scatter(
        df[df[var_y] > 0][var_x],
        df[df[var_y] > 0][var_y],
        color="green",
        label="positive_rent_growth",
    )
    ax.scatter(
        df[df[var_y] <= 0][var_x],
        df[df[var_y] <= 0][var_y],
        color="red",
        label="negative_rent_growth",
    )
    ax.set_xlabel("Supply Growth as a % of Existing Inventory")
    ax.set_ylabel("YoY Effective Rent Growth")
    ax.set_title(
        f"When Supply Grows, Rent is Anyone's Guess \n R-squared {r_value**2:.2f}"
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    plt.show()


def traditional_vars_vs_rent_nominal(df):
    vars = ["occ", "demand_units", "absorption"]
    # df = df.groupby("msa")[[*vars, "rentpsf"]].mean()
    fig, ax = plt.subplots(1, 3)
    for i, var in enumerate(vars):
        slope, intercept, r_value, p_value, std_err = linregress(df[var], df["rentpsf"])
        ax[i].scatter(df[var], df["rentpsf"], label=f"{var}")
        ax[i].plot(
            df[var], intercept + slope * df[var], color="red", label="Line of Best Fit"
        )
        ax[i].set_xlabel(var)
        ax[i].set_ylabel("Effective Rent PSF")
        ax[i].set_title(f"{var} vs Rent PSF: R-squared {r_value**2:.2f}")
    plt.show()


def traditional_vars_vs_rent_change(df):
    vars = ["occ_growth", "demand_units_growth", "absorption_growth"]
    # df = df.groupby("msa")[[*vars, "rent_growth_next_year"]].mean()
    fig, ax = plt.subplots(1, 3)
    for i, var in enumerate(vars):
        slope, intercept, r_value, p_value, std_err = linregress(
            df[var], df["rent_growth_next_year"]
        )
        ax[i].scatter(df[var], df["rent_growth_next_year"], label=f"{var}")
        ax[i].plot(
            df[var], intercept + slope * df[var], color="red", label="Line of Best Fit"
        )
        ax[i].set_xlabel(var)
        ax[i].set_ylabel("YoY Effective Rent Change: NTM")
        ax[i].set_title(
            f"{var} vs Rent Effective Rent Growth Next Year \n R-squared {r_value**2:.2f}"
        )
    plt.show()


def supply_demand_curves(df, show=True):
    var_y = "rentpsf"
    df = df.dropna()
    # df = df[df["msa"] == "New York - NY"]
    # df = df[df["year"] == 2023]
    var_x = "implied_demand"
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


def stats_check(df):
    df["interaction"] = (
        df["total_density_growth"] * df["supply_growth"] * df["rent_growth"]
    )
    vars = [
        "supply_growth",
        "total_density_growth",
        "rent_growth",
        "interaction",
    ]
    full_df = df
    df = df.dropna()
    avg_gap = []
    holder = []
    for year in range(df.year.unique().min(), df.year.unique().max() - 1):
        df_train = df[df["year"] <= year]
        X = df_train[vars]
        y = df_train["rent_growth_next_year"]
        X = sm.add_constant(X)  # Add constant term to the predictor variable
        model = sm.OLS(y, X).fit()
        df_test = df[df["year"] == year + 1]
        X_test = df_test[vars]
        X_test = sm.add_constant(X_test)
        y_hat = model.predict(X_test)
        y_true = pd.concat(
            [df_test["rent_growth_next_year"], pd.DataFrame(y_hat, columns=["y_hat"])],
            axis=1,
        )
        hi = y_true.nlargest(25, "y_hat")["rent_growth_next_year"].mean()
        md = (
            y_true.nlargest(75, "y_hat")
            .nsmallest(50, "y_hat")["rent_growth_next_year"]
            .mean()
        )
        lo = y_true.nsmallest(25, "y_hat")["rent_growth_next_year"].mean()
        holder.append((hi, md, lo))
        avg_gap.append(hi - lo)
    print(
        f"On average there is a {np.mean(avg_gap):.2%} gap between the top and bottom 20 MSAs"
    )
    holder = pd.DataFrame(holder, columns=["hi", "md", "lo"])
    fig, ax = plt.subplots()
    ax.bar(
        ["Forecasted Top 25 MSAs", "Forecasted Middle 50", "Forecasted Bottom 25 MSAs"],
        holder.mean().values,
    )
    for i, v in enumerate(holder.mean().values):
        ax.text(i, v, f"{v:.2%}", ha="center", va="bottom")
    ax.set_xlabel("Forecasted Rank")
    ax.set_ylabel("Actual Effective Rent Growth the Following Year of Forecast")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(
        f"Using Implied Demand model \n top quartile markets outperform bottom quartile by {np.mean(avg_gap):.1%}"
    )
    plt.show()

    df = full_df[full_df["year"] <= 2023].dropna()
    X = df[df["year"] <= 2023][vars]
    y = df[df["year"] <= 2023]["rent_growth_next_year"]
    X = sm.add_constant(X)  # Add constant term to the predictor variable
    model = sm.OLS(y, X).fit()
    print(model.summary())


def predict_future(df):
    df["interaction"] = (
        df["total_density_growth"] * df["supply_growth"] * df["rent_growth"]
    )
    vars = [
        "supply_growth",
        "total_density_growth",
        "rent_growth",
        "interaction",
    ]
    year = 2023
    df_train = df[df["year"] <= year].dropna()
    X = df_train[vars]
    y = df_train["rent_growth_next_year"]
    X = sm.add_constant(X)  # Add constant term to the predictor variable
    model = sm.OLS(y, X).fit()
    df_test = df[df["year"] == 2024]
    X_test = df_test[vars]
    X_test = sm.add_constant(X_test)
    y_hat = model.predict(X_test)
    forecast = pd.concat(
        [df_test[["msa", *vars]], pd.DataFrame(y_hat, columns=["y_hat"])], axis=1
    ).sort_values("y_hat")
    print(forecast.head(20))
    print(forecast.tail(20))
    # forecast.to_csv(r"C:\users\mlarriva\desktop\prediction.csv")
    sns.boxplot(
        x=pd.qcut(df["total_density_growth"], q=4), y=df["rent_growth_next_year"]
    )
    plt.xlabel("Total Density Growth Quartiles")
    plt.ylabel("Rent Growth Next Year")
    plt.title("Rent Growth Next Year by Total Density Growth Quartile")
    plt.show()


def supply_demand_annual(df, past_current="past"):
    if past_current == "global":
        df = df.dropna()
        years = df[df["year"] > 2004]["year"].unique()
        holder = []
        for year in years:
            df_t = df[(df["year"] <= year)]
            df_current = df[(df["year"] == year)]
            try:
                intersection_x, intersection_y = supply_demand_curves(df_t, show=False)
            except:
                intersection_x = np.nan
                intersection_y = np.nan
            holder.append(
                [
                    year,
                    intersection_x,
                    intersection_y,
                    df_current.rentpsf.mean(),
                    df_current.implied_demand.mean(),
                    df_current.supply_growth.mean(),
                ]
            )
        holder = pd.DataFrame(
            holder,
            columns=[
                "year",
                "quantity",
                "price",
                "rentpsf",
                "implied_demand",
                "supply_growth",
            ],
        )
        return holder
    if past_current == "each":
        # df = df.dropna()
        df["relative_rg_next_year"] = df["rent_growth_next_year"] - df.groupby("year")[
            "rent_growth_next_year"
        ].transform("median")
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
                    df_t[df_t["rent_growth"] < 0].shape[0],
                    df_current.rentpsf.values[0],
                    df_current.implied_demand.values[0],
                    df_current.supply_growth.values[0],
                    df_current.rent_growth_next_year.values[0],
                    df_current.relative_rg_next_year.values[0],
                    df_t["supply_growth"].mean(),
                    df_t["implied_demand"].mean(),
                ]
            )
        holder = pd.DataFrame(
            holder,
            columns=[
                "msa",
                "year",
                "quantity",
                "price",
                "pers_disinflation",
                "rentpsf",
                "implied_demand",
                "supply_growth",
                "rent_growth_next_year",
                "relative_rg_next_year",
                "hist_supply_growth_mean",
                "hist_implied_demand_mean",
            ],
        )
        return holder

    elif past_current == "current":
        dfm = df[df["year"] == df["year"].max()]
        dfm["relative_rg_next_year"] = df["rent_growth_next_year"] - df.groupby("year")[
            "rent_growth_next_year"
        ].transform("mean")
        holder = []
        raw = get_data().to_pandas()
        for x in dfm.itertuples():
            df_t = raw[(raw["msa"] == x.msa) & (raw["year"] <= x.year)]
            intersection_x, intersection_y = supply_demand_curves(df_t, show=False)
            holder.append(
                [
                    x.msa,
                    intersection_x,
                    x.implied_demand,
                    x.supply_growth,
                    x.rent_growth_next_year,
                    x.relative_rg_next_year,
                ]
            )
        holder = pd.DataFrame(
            holder,
            columns=[
                "msa",
                "quantity",
                "implied_demand",
                "supply_growth",
                "rent_growth_next_year",
                "relative_rg_next_year",
            ],
        )
        print(holder.sort_values("quantity"))
    if past_current == "past":
        df = df.dropna()
        dfm = df[
            df["supply_growth"] == df.groupby("msa")["supply_growth"].transform("max")
        ]
        print(
            holder.groupby(pd.qcut(holder["quantity"], 4))["rent_growth_next_year"].agg(
                ["mean", "count"]
            )
        )
        rg = holder.groupby(
            pd.cut(
                holder["quantity"],
                bins=[-1, 0, 0.01, 0.02, 1],
                labels=["<0", "0-1%", "1%-2%", ">2%"],
            )
        )["relative_rg_next_year"].agg(["mean", "count"])
        fig, ax = plt.subplots()
        ax.bar(
            rg.index,
            rg["mean"].values,
        )
        for i, v in enumerate(rg["mean"].values):
            ax.text(i, v, f"{v:.2%}", ha="center", va="bottom")
        ax.set_title(
            "Rent Growth Depends on Supply & Demand Intersection Point\n Avg Rent Growth In 100 Markets the Year After Supply Shocks"
        )
        ax.set_xlabel("Quantity where Supply and Demand Intersect")
        ax.set_ylabel("Rent Growth the Following Year\nRelative to All other Markets")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        plt.show()

    return holder


# 1) The relationship between supply and rent is weak and inconsistent
# A) Scatterplot of supply growth and rent growth
def part_one():
    pos_neg_supply_rent(var_x="supply_growth", var_y="rent_growth", df=df.dropna())
    pos_neg_supply_rent(
        var_x="supply_growth", var_y="rent_growth_next_year", df=df.dropna()
    )
    pos_neg_supply_rent(
        var_x="supply_growth",
        var_y="rent_growth",
        df=df.dropna().groupby("msa")[["supply_growth", "rent_growth"]].mean(),
    )
    # B) ...even at its most extreme the average rent growth is positive
    max_supply_growth(df.dropna())
    max_supply_min_rent(df)
    max_supply_max_rent(df)


# 2) Supply should be evaluated relative to demand, and the demand should meet supply to determine the price...
# A) But the demand curve is inverted: demand more units at a higher price?
def part_two():
    plt = plot_occ(
        var_x="occ",
        var_y="rentpsf",
        df=df,
    )
    plt.show()

    # B) This is because all our measures of demand are flawed, and derivative of supply
    # So they do a poor job of explaining price change... Scatterplot of growth of traditional demand growth vars vs rent growth
    traditional_vars_vs_rent_change(df)
    traditional_vars_vs_rent_nominal(df)


# 3) A better measure of demand would...
# A) Be negative sloping: demand increases as price decreases
def part_three():
    plot_what("implied_demand", "rentpsf", df.dropna()).show()
    # A) Disambiguate the relationship between supply and price
    density_growth_vs_rent_growth(df.dropna(), 3)
    # B) Intersect the Supply Curve at the market-clearing price
    supply_demand_curves(df.dropna())
    supply_demand_annual(df, past_current="past")
    # C) Would have predictive power alone and when combined with supply change
    plot_what("total_density_growth", "rent_growth_next_year", df).show()
    plot_what("total_density_growth", "rent_growth", df).show()
    # D) Would be statistically strong
    stats_check(df.dropna())
    # E) Could be used to predict future rent growth
    predict_future(df.dropna())

    # F) Bin analysis of rent growth and supply/demand intersection
    supply_demand_annual(df, past_current="each").to_csv(
        r"C:/users/mlarriva/desktop/output.csv"
    )
    df = pd.read_csv(r"C:/users/mlarriva/desktop/output.csv").dropna()
    df["quantity_group"] = pd.cut(
        df["quantity"],
        bins=[-1, -0.01, 0, 0.005, 0.01, 0.02, 0.03, 1],
        labels=["<-1%", "-1%-0", "0-0.5%", "0.5%-1%", "1%-2%", "2%-3%", ">3%"],
    )
    print(df[np.abs(df["quantity"]) >= 0.25].shape[0])
    df = df[np.abs(df["quantity"]) < 0.25]
    var_x = "quantity"
    var_x = "quantity_group"
    var_y = "relative_rg_next_year"
    var_y = "rent_growth_next_year"
    a = df.groupby(var_x)[var_y].agg(["mean", "count"]).reset_index()
    fig, ax = plt.subplots()
    ax.bar(
        a[var_x].to_list(),
        a["mean"].to_list(),
    )

    for i, v in enumerate(a["mean"].to_list()):
        ax.text(i, v, f"{v:.2%}, {a.iloc[i]['count']}", ha="center", va="bottom")
    ax.set_xlabel("Quantity where Supply and Implied Demand Intersect")
    ax.set_ylabel("Actual Effective Rent Growth the Following Year")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(
        f"Supply and Implied Demand Intersection predicts Next Year's Rent Growth\n Spread between Optimal Supply and Over Supply = {a['mean'].max()-a['mean'].min():.2%}"
    )
    plt.show()


def present_density():
    df = get_data()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="year", y="total_density", data=df.to_pandas())
    plt.title("Box and Whisker Plot of Density over Time in the 100 Largest MSAs")
    plt.xlabel("Year")
    plt.ylabel("Density (Population divided by Occupied Rental Units)")
    plt.xticks(rotation=45)
    plt.savefig(
        Path(__file__).resolve().parent.parent / "Figs" / "box_whisker_density_time.png"
    )
    plt.show()
    df = df.filter(pl.col("msa") == "New York - NY")
    plt.figure(figsize=(10, 6))
    sns.barplot(x="year", y="total_density", data=df.to_pandas(), ci=None)
    plt.title("Total Density in New York City MSA over Time")
    plt.xlabel("Year")
    plt.ylabel("Density (Population divided by Occupied Rental Units)")
    plt.xticks(rotation=45)
    plt.savefig(
        Path(__file__).resolve().parent.parent / "Figs" / "bar_total_density_year.png"
    )
    plt.show()


def present_density_growth_vs_rent_growth():
    df = get_data().to_pandas()
    df = df[df["year"] < 2022]
    df_grouped = df.groupby("msa").mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="total_density_growth", y="rent_growth", data=df_grouped)
    slope, intercept, r_value, p_value, std_err = linregress(
        df_grouped["total_density_growth"], df_grouped["rent_growth"]
    )
    plt.title(
        f"Average Density Growth vs Average Rent Growth by MSA\n 100 Largest MSAs 2001-2021 \n Rsquared: {r_value**2:.2f}"
    )
    plt.plot(
        np.linspace(
            df_grouped["total_density_growth"].min(),
            df_grouped["total_density_growth"].max(),
            10,
        ),
        intercept
        + slope
        * np.linspace(
            df_grouped["total_density_growth"].min(),
            df_grouped["total_density_growth"].max(),
            10,
        ),
        color="black",
        linestyle="--",
    )
    max_x = df_grouped["total_density_growth"].idxmax()
    min_x = df_grouped["total_density_growth"].idxmin()
    max_y = df_grouped["rent_growth"].idxmax()
    min_y = df_grouped["rent_growth"].idxmin()
    plt.annotate(
        df_grouped.loc[max_x, "msa"],
        (
            df_grouped.loc[max_x, "total_density_growth"],
            df_grouped.loc[max_x, "rent_growth"],
        ),
        textcoords="offset points",
        xytext=(0, 0),
        ha="right",
        color="black",
    )
    plt.annotate(
        df_grouped.loc[min_x, "msa"],
        (
            df_grouped.loc[min_x, "total_density_growth"],
            df_grouped.loc[min_x, "rent_growth"],
        ),
        textcoords="offset points",
        xytext=(0, 0),
        ha="left",
        color="black",
    )
    plt.annotate(
        df_grouped.loc[max_y, "msa"],
        (
            df_grouped.loc[max_y, "total_density_growth"],
            df_grouped.loc[max_y, "rent_growth"],
        ),
        textcoords="offset points",
        xytext=(0, 0),
        ha="right",
        color="black",
    )
    plt.annotate(
        df_grouped.loc[min_y, "msa"],
        (
            df_grouped.loc[min_y, "total_density_growth"],
            df_grouped.loc[min_y, "rent_growth"],
        ),
        textcoords="offset points",
        xytext=(0, 0),
        ha="right",
        color="black",
    )
    plt.legend()
    plt.xlabel("<-- Dedensifying        Average Density Growth      Densifying -->")
    plt.ylabel("<-- Increasing      Average Rent Growth     Decreasing -->")
    plt.savefig(
        Path(__file__).resolve().parent.parent
        / "Figs"
        / "density_growth_vs_rent_growth.png"
    )
    plt.show()


present_density_growth_vs_rent_growth()


def national_example():
    df = (
        get_data()
        # .filter(pl.col("msa") == "Inland Empire - CA")
        .filter(pl.col("year") < 2022)
        .select(
            "year",
            pl.col("rentpsf").alias("real_rent_psf"),
            pl.col("total_density").alias("RDI"),
            pl.col("rent_growth_next_year").alias("real_rent_growth"),
            pl.col("implied_demand").alias("delta_RDI"),
        )
        .group_by("year")
        .agg(
            pl.col("real_rent_psf").mean().alias("real_rent_psf"),
            pl.col("RDI").mean().alias("RDI"),
            pl.col("real_rent_growth").mean().alias("real_rent_growth"),
            pl.col("delta_RDI").mean().alias("delta_RDI"),
        )
    ).sort("year")
    fig, ax2 = plt.subplots(1, figsize=(10, 6))
    # Scatter plot of real_rent_growth vs delta_RDI
    df = (
        (df.select(["year", "real_rent_growth", "delta_RDI"]).drop_nulls())
        .to_pandas()
        .dropna()
    )
    slope, intercept, r_value, p_value, std_err = linregress(
        df["delta_RDI"], df["real_rent_growth"]
    )
    ax2.scatter(
        df["delta_RDI"], df["real_rent_growth"], color="black", label="Data points"
    )
    ax2.plot(
        df["delta_RDI"],
        intercept + slope * df["delta_RDI"],
        color="gray",
        label="Line of Best Fit",
    )
    max_x = df["delta_RDI"].idxmax()
    min_x = df["delta_RDI"].idxmin()
    max_y = df["real_rent_growth"].idxmax()
    min_y = df["real_rent_growth"].idxmin()
    plt.annotate(
        df.loc[max_x, "year"],
        (
            df.loc[max_x, "delta_RDI"],
            df.loc[max_x, "real_rent_growth"],
        ),
        textcoords="offset points",
        xytext=(0, 0),
        ha="right",
        color="black",
    )
    plt.annotate(
        df.loc[min_x, "year"],
        (
            df.loc[min_x, "delta_RDI"],
            df.loc[min_x, "real_rent_growth"],
        ),
        textcoords="offset points",
        xytext=(0, 0),
        ha="left",
        color="black",
    )
    plt.annotate(
        df.loc[min_y, "year"],
        (
            df.loc[min_y, "delta_RDI"],
            df.loc[min_y, "real_rent_growth"],
        ),
        textcoords="offset points",
        xytext=(0, 0),
        ha="right",
        color="black",
    )
    rsquare = r_value**2
    ax2.set_title(
        f"Average Density Growth vs Average Rent Growth by Year\n 100 Largest MSAs 2001-2021 \n Rsquared: {r_value**2:.2f}"
    )
    ax2.set_xlabel("<-- Dedensifying        Average Density Growth      Densifying -->")
    ax2.set_ylabel("<-- Increasing      Average Rent Growth     Decreasing -->")

    plt.tight_layout()
    plt.savefig(
        Path(__file__).resolve().parent.parent / "Figs" / "national_example.png"
    )
    plt.show()


national_example()
assert False


# Emperical Evidence
def anova():
    # The implied intersection of the supply and demand curves show distinct groupings of rent
    # Examining Relative Real Rent growth
    df = supply_demand_annual(get_data().to_pandas(), past_current="each")
    df = df[df["year"] < 2023]
    df["rent_ratio"] = df["rentpsf"] / df["price"]
    df["supply_delta"] = df["supply_growth"] / df["quantity"]
    df["med_supply_delta"] = df.groupby("year")["supply_delta"].transform("median")
    df["med_rent_ratio"] = df.groupby("year")["rent_ratio"].transform("median")
    df["group_P"] = None
    df["group_Q"] = None
    df.loc[(df["rent_ratio"] >= df["med_rent_ratio"]), "group_P"] = "overPriced"
    df.loc[(df["rent_ratio"] < df["med_rent_ratio"]), "group_P"] = "underPriced"
    df.loc[(df["supply_delta"] >= df["med_supply_delta"]), "group_Q"] = "overSupplied"
    df.loc[(df["supply_delta"] < df["med_supply_delta"]), "group_Q"] = "underSupplied"
    print(
        df.groupby(["group_P", "group_Q"])[
            ["relative_rg_next_year", "rent_growth_next_year"]
        ]
        .agg("mean")
        .map(lambda x: int(x * 10000))
        .reset_index()
    )
    df["group"] = df["group_P"] + "-" + df["group_Q"]
    df.to_csv(
        Path(__file__).resolve().parent.parent / "data" / "emperical_evidence.csv"
    )

    print("#### Relative Rent Growth ####")
    model = ols("relative_rg_next_year ~ C(group_P) + C(group_Q)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    tukey = pairwise_tukeyhsd(
        endog=df.dropna()["relative_rg_next_year"],
        groups=df.dropna()["group"],
        alpha=0.05,
    )
    print(tukey)
    print("#### Absolute Rent Growth ####")
    model = ols("rent_growth_next_year~ C(group_P) + C(group_Q)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    tukey = pairwise_tukeyhsd(
        endog=df.dropna()["rent_growth_next_year"],
        groups=df.dropna()["group"],
        alpha=0.05,
    )
    print(tukey)


def simplify_anova():
    df = supply_demand_annual(get_data().to_pandas(), past_current="each")
    df["overpriced"] = df["price"] < df["rentpsf"]
    df["oversupplied"] = df["quantity"] < df["supply_growth"]
    df["group"] = df["oversupplied"].astype(str) + "-" + df["overpriced"].astype(str)
    print(
        df.groupby(["overpriced", "oversupplied"])[
            ["relative_rg_next_year", "rent_growth_next_year"]
        ]
        .agg("mean")
        .map(lambda x: int(x * 10000))
        .reset_index()
    )
    model = ols(
        "relative_rg_next_year ~ C(overpriced) + C(oversupplied)",
        data=df,
    ).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    tukey = pairwise_tukeyhsd(
        endog=df.dropna()["relative_rg_next_year"],
        groups=df.dropna()["group"],
        alpha=0.05,
    )
    print(tukey)


# simplify_anova()
# assert False
def hist_rdi():
    df = get_data().to_pandas()
    plt.figure(figsize=(10, 6))
    plt.hist(df["total_density_growth"].dropna(), bins=30, edgecolor="k", alpha=0.7)
    plt.title(
        "Histogram of Change in Rental Density Index (ΔRDI) \n 100 Largest MSAs 2001-2023"
    )
    plt.xlabel("Change in Rental Density Index (ΔRDI)")
    plt.ylabel("Frequency")
    plt.grid(True)
    # Overlay summary statistics in a box on the plot
    mean_density_growth = df["total_density_growth"].mean()
    median_density_growth = df["total_density_growth"].median()
    std_density_growth = df["total_density_growth"].std()
    skewness_density_growth = skew(df["total_density_growth"].dropna())
    kurtosis_density_growth = kurtosis(df["total_density_growth"].dropna())
    textstr = "\n".join(
        (
            f"Mean: {mean_density_growth:.4f}",
            f"Median: {median_density_growth:.4f}",
            f"Std Dev: {std_density_growth:.4f}",
            f"Skewness: {skewness_density_growth:.4f}",
            f"Kurtosis: {kurtosis_density_growth:.4f}",
            f"IQR: {np.percentile(df['total_density_growth'].dropna(), 75) - np.percentile(df['total_density_growth'].dropna(), 25):.2f}",
        )
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    plt.gca().text(
        0.95,
        0.95,
        textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
    )
    plt.savefig(Path(__file__).resolve().parent.parent / "Figs" / "hist_deltardi.png")
    plt.show()


def example_plot():
    df = get_data().to_pandas()
    ny = df[(df["year"] < 2011) & (df["year"] > 2000) & (df["msa"] == "Austin - TX")]
    no = df[(df["year"] == 2011) & (df["msa"] == "Austin - TX")]
    fig, ax = plt.subplots(1, 3)
    # Target MSA historical supply vs rent
    ax[0].scatter(
        ny["supply_growth"],
        ny["rentpsf"],
        color="blue",
        label="supply points",
        zorder=5,
    )
    slope, intercept, r_value, p_value, std_err = linregress(
        ny["supply_growth"], ny["rentpsf"]
    )
    x_span_min = min([ny["implied_demand"].min(), ny["supply_growth"].min()])
    x_span_max = max([ny["implied_demand"].max(), ny["supply_growth"].max()])
    y_span_min = min([ny["rentpsf"].min()])
    y_span_max = max([ny["rentpsf"].max()])

    ax[0].plot(
        np.linspace(x_span_min, x_span_max, 10),
        intercept + slope * np.linspace(x_span_min, x_span_max, 10),
        color="blue",
        label="supply line",
    )
    # Target MSA implied demand vs rent
    ax[0].scatter(
        ny["implied_demand"],
        ny["rentpsf"],
        color="red",
        label="implied demand points",
        zorder=5,
    )
    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
        ny["implied_demand"], ny["rentpsf"]
    )
    ax[0].plot(
        np.linspace(x_span_min, x_span_max, 10),
        intercept2 + slope2 * np.linspace(x_span_min, x_span_max, 10),
        color="red",
        label="implied demand line",
    )
    # Derived Intersection
    derived_x = (intercept2 - intercept) / (slope - slope2)
    derived_y = intercept + slope * derived_x
    ax[0].plot(
        derived_x,
        derived_y,
        "*",
        label="derived intersection",
        color="gray",
        markersize=10,
        zorder=1,
    )
    ax[0].vlines(
        x=derived_x,
        ymin=0,
        ymax=derived_y,
        linestyle="--",
        color="gray",
    )
    ax[0].hlines(
        y=derived_y,
        xmin=-0.05,
        xmax=derived_x,
        linestyle="--",
        color="gray",
        label="derived quantity | price --",
    )
    # Observed Rent and Supply
    observed_x = no["supply_growth"].values[0]
    observed_y = no["rentpsf"].values[0]
    ax[0].plot(
        observed_x,
        observed_y,
        "s",
        label="observed intersection",
        color="black",
        markersize=7,
        zorder=1,
    )
    ax[0].vlines(
        x=observed_x,
        ymin=0,
        ymax=observed_y,
        linestyle="--",
        color="black",
    )
    ax[0].hlines(
        y=observed_y,
        xmin=-0.05,
        xmax=observed_x,
        linestyle="--",
        color="black",
        label="observed quantity | price--",
    )

    distance = abs(observed_y - derived_y)
    rent_ratio = observed_y / derived_y
    ax[0].set_xlim(-0.03, 0.03)
    ax[0].set_ylim(0.75, 1.4)
    ax[0].yaxis.set_major_formatter(plt.FormatStrFormatter("$%.2f"))
    # Annotate the Y axis
    x_min = ax[0].get_xlim()[0]
    x_max = ax[0].get_xlim()[1]
    ax[0].annotate(
        f"Observed Rent (${observed_y:.3f})",
        xy=(x_min, observed_y),
        xytext=(x_min, observed_y + 0.025),
        color="black",
        # arrowprops=dict(facecolor="black", shrink=0.02),
    )
    ax[0].annotate(
        f"Derived Rent (${derived_y:.3f})",
        xy=(x_min, derived_y),
        xytext=(x_min, derived_y + 0.025),
        color="gray",
        # arrowprops=dict(facecolor="gray", shrink=0.02),
    )
    # ax[0].annotate(
    #     f"Rent Ratio = {rent_ratio:.2f}",
    #     xy=(x_min, (derived_y + observed_y) / 2),
    #     xytext=(x_min, (derived_y + observed_y) / 2),
    #     color="blue",
    # )
    # Annotate the X axis
    y_min = ax[0].get_ylim()[0]
    y_max = ax[0].get_ylim()[1]
    supply_delta = observed_x / derived_x
    ax[0].annotate(
        f"Observed\nSupply\n ({observed_x:.3f})",
        xy=(observed_x - 0.0075, y_min),
        xytext=(observed_x - 0.0075, y_min),
        color="black",
        # arrowprops=dict(facecolor="black", shrink=0.02),
    )
    ax[0].annotate(
        f"Derived\nSupply\n({derived_x:.3f})",
        xy=(derived_x + 0.0025, y_min),
        xytext=(derived_x + 0.0025, y_min),
        color="gray",
        # arrowprops=dict(facecolor="gray", shrink=0.02),
    )
    # ax[0].annotate(
    #     f"Ratio =\n{supply_delta:.2f}",
    #     xy=((derived_x + observed_x) / 2, y_min),
    #     xytext=((derived_x + observed_x) / 2, y_min),
    #     color="red",
    # )
    ax[0].set_title(
        "Supply and Implied Demand Curves: Orange County - CA \n Using data from 2001-2011"
    )
    ax[0].legend()

    # Ratios
    df = pd.read_csv(
        Path(__file__).resolve().parent.parent / "data" / "emperical_evidence.csv"
    ).dropna()
    dr = df[(df["year"] == 2011) & (df["msa"] == "Austin - TX")]
    ax[1].plot(
        dr["supply_delta"],
        dr["rent_ratio"],
        "*",
        label="observed intersection",
        color="purple",
        markersize=7,
        zorder=1,
    )
    ax[1].annotate(
        f"Quotient of\n Observed Rent ${observed_y:.3} \n to Derived Rent ${derived_y:.3}",
        xy=(dr["supply_delta"], dr["rent_ratio"]),
        xytext=(dr["supply_delta"], dr["rent_ratio"]),
        color="blue",
    )
    ax[1].annotate(
        f"Quotient of\n Observed Supply {observed_x:.3} \n to Derived Demand {derived_x:.3}",
        xy=(dr["supply_delta"], dr["rent_ratio"]),
        xytext=(dr["supply_delta"] * 0.95, dr["rent_ratio"] * 0.95),
        color="red",
    )
    ax[1].set_ylim(0.75, 1.25)
    ax[1].set_xlim(-1, 1)

    # National Intersections in 2011
    df = df[df["year"] <= 2011]
    ax[2].scatter(
        df[df["year"] == 2011]["supply_delta"],
        df[df["year"] == 2011]["rent_ratio"],
        color="gray",
    )
    avg_x = df[df["year"] == 2011]["supply_delta"].median()
    print(df.loc[df["year"] == 2011, "supply_delta"].sort_values())
    avg_y = df[df["year"] == 2011]["rent_ratio"].median()
    ax[2].vlines(
        x=avg_x,
        ymin=0,
        ymax=avg_y,
        linestyle="--",
        color="orange",
    )
    ax[2].hlines(
        y=avg_y,
        xmin=-5,
        xmax=avg_x,
        linestyle="--",
        color="orange",
    )
    ax[2].set_ylim(0.75, 1.25)
    ax[2].set_xlim(-1, 1)
    ratio_x = observed_x / derived_x
    ratio_y = observed_y / derived_y
    ax[2].plot(
        ratio_x,
        ratio_y,
        "*",
        label="q* p*",
        color="red",
        markersize=5,
    )
    # # Observed Rent and Supply
    # observed_x = df[df["year"] == 2011]["supply_growth"].mean()
    # observed_y = df[df["year"] == 2011]["rentpsf"].mean()
    # ax[1].plot(observed_x, observed_y, "ro", label="Intersection")
    # ax[1].vlines(
    #     x=observed_x,
    #     ymin=0,
    #     ymax=observed_y,
    #     linestyle="--",
    #     color="black",
    # )
    # ax[1].hlines(
    #     y=observed_y,
    #     xmin=-0.05,
    #     xmax=observed_x,
    #     linestyle="--",
    #     color="black",
    # )

    # ax[1].set_xlim(-0.02, 0.05)
    # ax[1].set_ylim(0, 3)
    plt.show()
    # nova


anova()
# example_plot()
# national_example()
# present_density()
# hist_rdi()
