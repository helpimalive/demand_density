import matplotlib.pyplot as plt
import polars as pl
from preprocess import load_data
from matplotlib.ticker import MaxNLocator


def graph_spread_between_excess_crowding_and_rent_growth():
    df = pl.read_csv(r"Exhibits\excess_crowding_vs_rent_growth.csv").select(
        "year",
        "excess_crowding",
        "met_name",
        "real_relative_rent_growth_next_year",
        "forecast_rent",
    )
    yvar = "real_relative_rent_growth_next_year"
    xvar = "excess_crowding"  # or forecast_rent
    df = df.with_columns((pl.col("forecast_rent")).alias("excess_crowding"))

    df = df.with_columns(
        pl.col("excess_crowding").median().over("year").alias("excess_crowding_median")
    ).with_columns(
        pl.when(pl.col("excess_crowding") >= pl.col("excess_crowding_median"))
        .then(pl.lit("High Excess Crowding"))
        .otherwise(pl.lit("Low Excess Crowding"))
        .alias("excess_crowding_group"),
    )
    df = (
        df.group_by(["year", "excess_crowding_group"])
        .agg(pl.col(yvar).mean().alias("rent_growth_mean") * 10000)
        .pivot(index="year", on="excess_crowding_group", values="rent_growth_mean")
        .sort("year")
        .with_columns(
            (pl.col("High Excess Crowding") - pl.col("Low Excess Crowding")).alias(
                "rent_growth_diff"
            )
        )
    )
    pdf = df.select(["year", "rent_growth_diff"]).to_pandas()
    plt.figure(figsize=(10, 6))
    avg = pdf["rent_growth_diff"].mean()
    plt.bar(
        pdf["year"], pdf["rent_growth_diff"], color="#4C72B0", edgecolor="k", width=0.6
    )
    plt.axhline(avg, color="black", linestyle="--", linewidth=1.5)
    for x, y in zip(pdf["year"], pdf["rent_growth_diff"]):
        va = "bottom" if y >= 0 else "top"
        plt.text(x, y, f"{y:.1f}", ha="center", va=va, fontsize=8)
    plt.text(pdf["year"].max(), avg, f"  Avg {avg:.1f}", va="top", color="black")
    plt.plot(pdf["year"], pdf["rent_growth_diff"], marker="o", linewidth=2)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel("Excess Real Rent Growth \n in the Year Following Excess Crowding")
    plt.title("Excess Real Rent Growth in MSAs with Excess Crowding (basis points) ")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()


graph_spread_between_excess_crowding_and_rent_growth()
