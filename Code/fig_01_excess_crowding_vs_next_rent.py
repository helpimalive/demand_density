import matplotlib.pyplot as plt
import polars as pl
import os
from preprocess import load_data
from matplotlib.ticker import MaxNLocator


def graph_spread_between_excess_crowding_and_rent_growth():
    df = (pl.read_csv(r"Exhibits\excess_crowding_vs_rent_growth.csv").select(
        "year",
        "met_name",
        "real_relative_rent_growth_next_year",
        "density_rented"))
    dd = pl.read_csv(r"Data\us_hh_density.csv").with_columns(pl.col('year').cast(pl.Int64))
    df = df.join(dd.select('year','us_hh_density'),how='left',on='year')

    yvar = "real_relative_rent_growth_next_year"

    x_var = 'density_rented'

    df = df.with_columns(
        (pl.when(pl.col(x_var)
        > pl.col('us_hh_density'))
        .then(pl.lit(f"High_{x_var}"))
        .otherwise(pl.when(pl.col(x_var)
        <pl.col('us_hh_density'))
        .then(pl.lit(f"Low_{x_var}")))
        .alias("excess_crowding_group"),
    ))

    print(
        df.group_by(["year", "excess_crowding_group"])
        .agg(pl.col(yvar).count().alias("rent_growth_count") ))
    
    df = (
        df.group_by(["year", "excess_crowding_group"])
        .agg(pl.col(yvar).mean().alias("rent_growth_mean") * 10000)
        .pivot(index="year", on="excess_crowding_group", values="rent_growth_mean")
        .sort("year")
        .with_columns(
            (pl.col(f"High_{x_var}") - pl.col(f"Low_{x_var}")).alias(
                "rent_growth_diff"
            )
        )
    )
    pdf = df.select(["year", "rent_growth_diff"]).to_pandas()
    plt.figure(figsize=(10, 6))
    avg = pdf["rent_growth_diff"].median()
    plt.bar(
        pdf["year"], pdf["rent_growth_diff"], color="#4C72B0", edgecolor="k", width=0.6
    )
    plt.axhline(avg, color="black", linestyle="--", linewidth=1.5)
    for x, y in zip(pdf["year"], pdf["rent_growth_diff"]):
        va = "bottom" if y >= 0 else "top"
        plt.text(x, y, f"{y:.0f}", ha="center", va=va, fontsize=8)
    plt.text(pdf["year"].max()-3.75, avg+10, f"  Median {avg:.0f}bps", va="bottom", color="black")
    # plt.plot(pdf["year"], pdf["rent_growth_diff"], marker="o", linewidth=2)
    plt.axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Year")
    plt.ylabel("Excess Real Rent Growth \n in the Year Following Excess Crowding")
    plt.title("Excess Real Rent Growth in MSAs with Excess Crowding (basis points) ")
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid(True, linestyle=":", linewidth=0.5)
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(__file__))[0]
    out_path = os.path.join(out_dir, f"{base_name}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()


graph_spread_between_excess_crowding_and_rent_growth()
