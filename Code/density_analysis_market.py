import pandas as pd
import polars as pl
from scipy.stats import linregress
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
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


def get_data():
    df = pl.read_csv(
        r"C:\Users\mlarriva\OneDrive - FCP\Personal Folders\Documents\Supply Study\Data\msa_data.csv"
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
    # Transformations
    cpi = pl.read_csv(
        r"C:\Users\mlarriva\OneDrive - FCP\Personal Folders\Documents\Supply Study\Data\cpi_ex_shelter.csv"
    )
    cpi = cpi.with_columns(
        pl.col("DATE").str.strptime(pl.Date, "%m/%d/%Y").alias("date")
    )
    cpi = cpi.with_columns(pl.col("date").dt.year().cast(pl.Int16).alias("year"))
    cpi = (
        cpi.filter(pl.col("year") > 1999)
        .with_columns((pl.col("CUSR0000SA0L2").cast(pl.Float32) / 100).alias("cpi_pct"))
        .select(pl.col("year"), pl.col("cpi_pct"))
    )
    df = df.join(cpi, on="year", how="left")
    df = df.with_columns(
        (pl.col("rent_growth") - pl.col("cpi_pct")).alias("rent_growth")
    )
    df = (
        df.sort("msa", "year", descending=False)
        .with_columns(
            ((pl.col("pop")) / (pl.col("occ") * pl.col("inventory"))).alias(
                "total_density"
            ),
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
            ((pl.col("total_density").pct_change().over("msa"))).alias(
                "total_density_growth"
            )
        )
        .with_columns(implied_demand=pl.col("total_density_growth"))
    )
    # Excluding NOLA because Katrina made density go haywire
    # df = df.filter((pl.col("msa") != "New Orleans - LA") | (pl.col("year") != 2006))
    df.write_csv(r"C:\users\mlarriva\desktop\output.csv")
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
    # supply_demand_annual(df, past_current="each").to_csv(
    #     r"C:/users/mlarriva/desktop/output.csv"
    # )
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
    df.loc[(df["rent_ratio"] >= df["med_rent_ratio"]), "group_P"] = "overP"
    df.loc[(df["rent_ratio"] < df["med_rent_ratio"]), "group_P"] = "underP"
    df.loc[(df["supply_delta"] >= df["med_supply_delta"]), "group_Q"] = "overQ"
    df.loc[(df["supply_delta"] < df["med_supply_delta"]), "group_Q"] = "underQ"
    print(
        df.groupby(["group_P", "group_Q"])["relative_rg_next_year"].agg(
            ["mean", "count"]
        )
    )
    df["group"] = df["group_P"] + df["group_Q"]
    df.to_csv(
        r"C:\Users\mlarriva\OneDrive - FCP\Personal Folders\Documents\Supply Study\Data\emperical_evidence.csv"
    )
    model = ols("relative_rg_next_year ~ C(group_P) + C(group_Q)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    tukey = pairwise_tukeyhsd(
        endog=df.dropna()["relative_rg_next_year"],
        groups=df.dropna()["group"],
        alpha=0.05,
    )
    print(tukey)


anova()
assert False
do = get_data().to_pandas().dropna()
do = do[(do["year"] <= 2011) & (do["year"] > 2000)]
# National Intersections in 2011
df = pd.read_csv(
    r"C:\Users\mlarriva\OneDrive - FCP\Personal Folders\Documents\Supply Study\Data\emperical_evidence.csv"
).dropna()
df = df[df["year"] <= 2011]

# Target MSA values in 2011
ny = do[(do["msa"] == "Los Angeles - CA") & (do["year"] <= 2011)]
fix, ax = plt.subplots(1, 2)
# Target MSA historical supply vs rent
ax[0].scatter(ny["supply_growth"], ny["rentpsf"], color="blue", label="supply points")
slope, intercept, r_value, p_value, std_err = linregress(
    ny["supply_growth"], ny["rentpsf"]
)
ax[0].plot(
    ny["supply_growth"],
    intercept + slope * ny["supply_growth"],
    color="blue",
    label="supply line",
)
# Target MSA implied demand vs rent
ax[0].scatter(
    ny["implied_demand"], ny["rentpsf"], color="red", label="implies demand points"
)
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
    ny["implied_demand"], ny["rentpsf"]
)
ax[0].plot(
    ny["implied_demand"],
    intercept2 + slope2 * ny["implied_demand"],
    color="red",
    label="implied demand line",
)
ax[0].set_xlim(-0.02, 0.02)
ax[0].set_ylim(1.5, 2.5)

# Derived Intersection
intersection_x = (intercept2 - intercept) / (slope - slope2)
intersection_y = intercept + slope * intersection_x
ax[0].plot(intersection_x, intersection_y, "ro", label="derived intersection")
ax[0].vlines(
    x=intersection_x,
    ymin=0,
    ymax=intersection_y,
    linestyle="--",
    color="black",
    label="derived quantity",
)
ax[0].hlines(
    y=intersection_y,
    xmin=-0.05,
    xmax=intersection_x,
    linestyle="--",
    color="black",
    label="derived price",
)
# Observed Rent and Supply
observed_x = ny[ny["year"] == 2011]["supply_growth"].values[0]
observed_y = ny[ny["year"] == 2011]["rentpsf"].values[0]
ax[0].plot(observed_x, observed_y, "ro", label="observed intersection")
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
)

# National Suppy vs Rent
# ax[1].scatter(do["supply_growth"], do["rentpsf"])
# slope, intercept, r_value, p_value, std_err = linregress(
#     do["supply_growth"], do["rentpsf"]
# )
# ax[1].plot(
#     do["supply_growth"],
#     intercept + slope * do["supply_growth"],
#     color="black",
# )
# National Intersections
print(
    df[df["year"] == 2011][["msa", "supply_delta", "rent_ratio"]].sort_values(
        "supply_delta"
    )
)
print(
    df[df["year"] == 2011][["msa", "supply_delta", "rent_ratio"]].sort_values(
        "rent_ratio"
    )
)
ax[1].scatter(
    df[df["year"] == 2011]["supply_delta"], df[df["year"] == 2011]["rent_ratio"]
)
avg_x = df[df["year"] == 2011]["supply_delta"].mean()
avg_y = df[df["year"] == 2011]["rent_ratio"].mean()
ax[1].vlines(
    x=avg_x,
    ymin=0,
    ymax=avg_y,
    linestyle="--",
    color="red",
)
ax[1].hlines(
    y=avg_y,
    xmin=-0.5,
    xmax=avg_x,
    linestyle="--",
    color="red",
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
