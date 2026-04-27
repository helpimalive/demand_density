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
    df = df.with_columns(
        pl.col("excess_crowding")
        .qcut(4, labels=["Q1", "Q2", "Q3", "Q4"], allow_duplicates=True)
        .alias("excess_crowding_group")
    )
    df = (
        df.group_by(["year", "excess_crowding_group"])
        .agg(pl.col(yvar).mean().alias("rent_growth_mean") * 10000)
        .pivot(index="year", on="excess_crowding_group", values="rent_growth_mean")
        .sort("year")
        .with_columns((pl.col("Q4") - pl.col("Q1")).alias("rent_growth_diff"))
    ).mean()
    plt.figure(figsize=(10, 6))
    plt.bar(
        ["Q1", "Q2", "Q3", "Q4"],
        df[["Q1", "Q2", "Q3", "Q4"]].to_numpy().flatten(),
        color="skyblue",
        edgecolor="black",
    )
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
