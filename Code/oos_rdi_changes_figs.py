import pandas as pd
import numpy as np
import polars as pl
import statsmodels.api as sm
from preprocess import load_data
import numpy as np


def oos_density_delta_forecast():

    # ---------------- LOAD DATA ----------------
    df = (
        load_data()
        .select(
            "year",
            "met_name",
            "real_relative_rent_growth_next_year",
            "real_relative_rent_growth_this_year",
            "supply_growth",
            "occupancy",
            "absorption",
            "density_rented",
            "bedroom_density_rented",
            "density_rented_change",
            "age_under_18_share",
            "age_25_to_34_share",
            "age_34_to_49_share",
        )
        .drop_nulls()
        .with_columns(np.sign(pl.col("density_rented_change")).alias("drccs"))
        .to_pandas()
    )
    x = 2
    # ---------------- ROLLING OOS RENT FORECASTS ----------------
    records = []
    train_test_pers = [
        [(2005, 2009), (2010, 2014)],
        [(2010, 2014), (2015, 2019)],
        [(2015, 2019), (2020, 2025)],
    ]

    for (train_start, train_end), (test_start, test_end) in train_test_pers:
        train = df[(df["year"] >= train_start) & (df["year"] <= train_end)]
        test = df[(df["year"] >= test_start) & (df["year"] <= test_end)]
        y_true = test.groupby("met_name")["real_relative_rent_growth_next_year"].sum()

        def fit_and_predict(X_cols, label, direction, train=train, test=test):
            X_train = train[["met_name"] + X_cols]
            y_hat = X_train.groupby("met_name")[X_cols].mean() * direction
            y_hat = (
                (test[["met_name"]].merge(y_hat, on="met_name", how="left"))
                .iloc[:, 1]
                .values
            )

            for yt, yh in zip(y_true, y_hat):
                records.append(
                    {
                        "year": test_start,
                        "model": label,
                        "y_true": yt,
                        "y_hat": yh,
                    }
                )

        # if higher is better, direction = 1; if lower is better, direction = -1
        specs = [
            (["supply_growth"], "Supply growth only", -1),
            (["real_relative_rent_growth_this_year"], "Lagged rent growth only", 1),
            (["occupancy"], "Occupancy only", 1),
            (["absorption"], "Absorption only", 1),
            (["drccs"], "RDI_delta", 1),
        ]

        for cols, label, direction in specs:
            fit_and_predict(cols, label, direction)

    res_df = pd.DataFrame(records)

    # ---------------- QUARTILE MEAN DIFFERENCES ----------------
    quart_records = []
    for (model, year), g in res_df.groupby(["model", "year"]):
        g = g.copy()
        g["quartile"] = pd.qcut(g["y_hat"], x, labels=False, duplicates="drop")
        for q, sub in g.groupby("quartile"):
            mean_y_hat = sub["y_hat"].mean()
            mean_y_true = sub["y_true"].mean()
            quart_records.append(
                {
                    "model": model,
                    "year": year,
                    "quartile": int(q) + 1,  # 1-indexed for readability
                    "mean_y_hat": mean_y_hat * 10000,
                    "mean_y_true": mean_y_true * 10000,
                }
            )

    df = pd.DataFrame(quart_records)
    df = df.pivot(index=["model", "year"], columns="quartile", values="mean_y_true")
    print(df)
    df["delta_q4_q1"] = df[x] - df[1]
    df = df.groupby("model")["delta_q4_q1"].mean().abs().reset_index()
    print(df.sort_values("delta_q4_q1", ascending=False))
    import matplotlib.pyplot as plt

    plot_df = df.sort_values("delta_q4_q1", ascending=False).reset_index(drop=True)
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, plot_df["delta_q4_q1"], color="C0")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model"], rotation=45, ha="right")
    ax.set_xlabel("Variable Used")
    ax.set_ylabel(
        "Difference in Real Relative Rent Growth Means: Top Half vs Bottom Half (bps)"
    )
    # ax.set_title("Out-of-Sample Forecasts of Future Rent Growth: Top vs Bottom Halfs")

    for i, v in enumerate(plot_df["delta_q4_q1"]):
        ax.text(i, v, f"{v:.0f}bps", ha="center", va="bottom" if v >= 0 else "top")

    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(
        rf"Figs/fig_oos_density_delta_halfs.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()


def graph_pos_neg_deltas_and_rg():
    train_test_pers = [
        [(2006, 2010), (2011, 2015)],
        [(2011, 2015), (2016, 2020)],
        [(2016, 2020), (2021, 2025)],
    ]
    df = (
        load_data()
        .select(
            "year",
            "met_name",
            "real_relative_rent_growth_next_year",
            "real_relative_rent_growth_this_year",
            "supply_growth",
            "occupancy",
            "absorption",
            "density_rented",
            "bedroom_density_rented",
            "density_rented_change",
            "age_under_18_share",
            "age_25_to_34_share",
            "age_34_to_49_share",
        )
        .drop_nulls()
        .with_columns(np.sign(pl.col("density_rented_change")).alias("drccs"))
        .to_pandas()
    )
    records = []
    for (train_start, train_end), (test_start, test_end) in train_test_pers:
        train = df[(df["year"] >= train_start) & (df["year"] <= train_end)]
        test = df[(df["year"] >= test_start) & (df["year"] <= test_end)]
        X_train = train[["met_name", "drccs"]]
        y_hat = X_train.groupby("met_name")["drccs"].sum()
        test = pl.DataFrame(
            test[["met_name", "real_relative_rent_growth_this_year"]]
            .groupby("met_name")
            .sum()
            .merge(y_hat, on="met_name", how="left")
        )
        records.append(test)
    df = pl.concat(records)

    import matplotlib.pyplot as plt

    ## Bar Graph
    df_one = (
        df.group_by("drccs")
        .agg(
            pl.col("real_relative_rent_growth_this_year").mean().alias("rrrgty"),
            pl.col("drccs").count().alias("instance_count"),
        )
        .sort("drccs")
    )

    pdf = df_one.to_pandas()
    x = pdf["drccs"].astype(str)
    y = pdf["rrrgty"] * 10000
    z = pdf["instance_count"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, y, color="C0")
    ax.set_xlabel("Instances of positive/negative density change: Last 5 Years")
    ax.set_ylabel("Mean Real Relative Rent Growth: Next 5 Years")
    # ax.set_title("Past Density Change and Future Rent Growth")
    for i, v in enumerate(y):
        ax.text(
            i, v, f"{v:.0f}bps, n={z[i]}", ha="center", va="bottom" if v >= 0 else "top"
        )
    # plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(
        rf"Figs/fig_oos_density_delta_forecast.pdf",
        format="pdf",
        bbox_inches="tight",
    )

    plt.show()

    # pdf = df.to_pandas()
    # counts = pdf.groupby("drccs").size()
    # valid_groups = counts[counts > 5].index.sort_values()

    # data = [
    #     pdf.loc[pdf["drccs"] == g, "real_relative_rent_growth_this_year"]
    #     for g in valid_groups
    # ]

    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.boxplot(
    #     data, labels=[str(g) for g in valid_groups], vert=True, patch_artist=True
    # )
    # ax.set_xlabel("drccs")
    # ax.set_ylabel("real_relative_rent_growth_this_year")
    # ax.set_title("Box-and-Whiskers: Future Rent Growth by drccs (n>5)")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    oos_density_delta_forecast()
    graph_pos_neg_deltas_and_rg()
