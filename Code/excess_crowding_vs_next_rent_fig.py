import matplotlib.pyplot as plt
import polars as pl


df = pl.read_csv(r"Exhibits\excess_crowding_vs_rent_growth.csv")
data = (
    df.filter(pl.col("excess_crowding") > 0)
    .group_by("year")
    .agg(pl.col("rent_growth_next").mean())
    .join(
        df.filter(pl.col("excess_crowding") < 0)
        .group_by("year")
        .agg(pl.col("rent_growth_next").mean()),
        on="year",
        suffix="_neg",
    )
).with_columns(diff=pl.col("rent_growth_next") - pl.col("rent_growth_next_neg"))
# prepare data per year
years = sorted(data.get_column("year").unique().to_list())
groups = [
    data.filter(pl.col("year") == yr).get_column("rent_growth_next").to_list()
    for yr in years
]

# plot
plt.figure(figsize=(10, 6))
plt.boxplot(groups, labels=[str(y) for y in years], patch_artist=True)
plt.xlabel("Year")
plt.ylabel("rent_growth_next")
plt.title("Distribution of next-period rent growth by year")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
