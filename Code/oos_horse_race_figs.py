import pandas as pd
import numpy as np
import polars as pl
import statsmodels.api as sm
from preprocess import load_data


START_FORECAST_YEAR = 2010


def oos_density_forecasts(quantiles=5):

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
            "real_rent_growth",
            "real_rent_growth_next_year"
        )
        .with_columns(
            pl.col("density_rented")
            .pct_change()
            .over("met_name", order_by="year")
            .alias("drpc")
        )
        .with_columns(
            (
                (pl.col("density_rented"))
                # * (pl.col("real_relative_rent_growth_this_year"))
                * (pl.col("real_rent_growth"))
                #  (pl.col('drpc'))

            ).alias("RDI_var")
        )
        .to_pandas()
        .sort_values(["met_name", "year"])
    )
    splits = quantiles
    # ---------------- ROLLING OOS RENT FORECASTS ----------------
    records = []

    years = sorted(df["year"].unique())
    forecast_years = [y for y in years if y >= START_FORECAST_YEAR]

    for t in forecast_years:
        train = df[df["year"] < t]
        test = df[df["year"] == t]

        y_true = test["real_relative_rent_growth_next_year"].values

        def fit_and_predict(X_cols, label):
            X_train = train[X_cols]
            y_train = train["real_relative_rent_growth_next_year"]
            # if label == "RDI_based":
            #     y_hat = test["RDI_var"]
            # else:
            model = sm.OLS(
                y_train,
                sm.add_constant(X_train),
                missing="drop",
            ).fit()

            y_hat = model.predict(sm.add_constant(test[X_cols]))
            for yt, yh in zip(y_true, y_hat):
                records.append(
                    {
                        "year": t,
                        "model": label,
                        "y_true": yt,
                        "y_hat": yh,
                    }
                )

        specs = [
            (["supply_growth"], "Supply growth only"),
            (["real_relative_rent_growth_this_year"], "Lagged rent growth only"),
            (["occupancy"], "Occupancy only"),
            (["absorption"], "Absorption only"),
            (["RDI_var"], "RDI*Lagged Rent Growth"),
            (
                [
                    "supply_growth",
                    "real_relative_rent_growth_this_year",
                    "occupancy",
                    "absorption",
                ],
                "All predictors except RDI",
            ),
        ]

        for cols, label in specs:
            fit_and_predict(cols, label)

    res_df = pd.DataFrame(records)

    # ---------------- QUARTILE MEAN DIFFERENCES ----------------
    quart_records = []
    for (model, year), g in res_df.groupby(["model", "year"]):
        if g.shape[0] == 0:
            continue
        g = g.copy()
        g["quartile"] = pd.qcut(g["y_hat"], splits, labels=False, duplicates="drop")
        for q, sub in g.groupby("quartile"):
            mean_y_hat = sub["y_hat"].mean()
            mean_y_true = sub["y_true"].mean()
            diff = mean_y_hat - mean_y_true
            quart_records.append(
                {
                    "model": model,
                    "year": year,
                    "quartile": int(q) + 1,  # 1-indexed for readability
                    "mean_y_hat": mean_y_hat,
                    "mean_y_true": mean_y_true,
                }
            )

    df = pd.DataFrame(quart_records)

    # Average the quartile mean differences across years for each model
    # Get max and min quartile rows by model/year
    max_quartile = df.loc[df.groupby(["model", "year"])["quartile"].idxmax()]
    min_quartile = df.loc[df.groupby(["model", "year"])["quartile"].idxmin()]

    # Merge and compute difference
    result = max_quartile[["model", "year", "mean_y_true", "quartile"]].reset_index(
        drop=True
    )
    result["mean_y_true_min"] = min_quartile["mean_y_true"].values
    result["diff"] = result["mean_y_true"] - result["mean_y_true_min"]
    result = result.rename(columns={"mean_y_true": "mean_y_true_max"})
    result = (
        result.groupby("model")["diff"]
        .mean()
        .reset_index()
        .sort_values("diff", ascending=False)
    )
    import matplotlib.pyplot as plt

    if splits == 5:
        quantile = "Quintile"
    elif splits == 10:
        quantile = "Decile"
    plt.figure(figsize=(10, 6))
    x = result["model"]
    y = result["diff"] * 10000  # convert to bps
    plt.bar(x, y, color="C0")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Model")
    plt.ylabel(f"Difference of Means (bps) top {quantile} - bottom {quantile}")
    for i, v in enumerate(y):
        plt.text(i, v, f"{v:.0f}bps", ha="center", va="bottom" if v >= 0 else "top")
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(
        rf"Figs/fig_oos_density_forecast_plot_{quantile}.pdf",
        format="pdf",
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    oos_density_forecasts(5)
    oos_density_forecasts(10)
