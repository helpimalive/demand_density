import matplotlib.pyplot as plt
import polars as pl
import os
from preprocess import load_data
from matplotlib.ticker import MaxNLocator


def graph_spread_between_excess_crowding_and_rent_growth():
    # df = pl.read_csv(r"Exhibits\excess_crowding_vs_rent_growth.csv")
    # print(df.columns)
    # assert False
    df = pl.read_csv(r"Exhibits\excess_crowding_vs_rent_growth.csv").select(
        "year",
        "excess_crowding",
        "met_name",
        "real_relative_rent_growth_next_year",
    )
    yvar = "real_relative_rent_growth_next_year"
    xvar = "excess_crowding"
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
        plt.text(x, y, f"{y:.0f}", ha="center", va=va, fontsize=8)
    plt.text(2019.5, avg - 2, f"  Avg {avg:.0f}", va="top", color="black")
    plt.xlabel("Year")
    plt.ylabel("Excess Real Rent Growth \n in the Year Following Excess Crowding")
    plt.title("Excess Real Rent Growth in MSAs with Excess Crowding (basis points) ")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(__file__))[0]
    out_path = os.path.join(out_dir, f"{base_name}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to: {out_path}")
    plt.show()


graph_spread_between_excess_crowding_and_rent_growth()
