import polars as pl


def load_data():
    df = pl.read_csv(r"Data\rental_density_indices_enriched.csv")
    largest_met_names = (
        df.group_by("MET_NAME")
        .agg(pl.col("inventory_units").sum().alias("total_inventory"))
        .sort("total_inventory", descending=True)
        .head(100)
        .select("MET_NAME")
        .to_series()
        .to_list()
    )
    df = df.filter(pl.col("MET_NAME").is_in(largest_met_names))

    df = df.with_columns(
        (pl.col("rent_growth") - pl.col("rent_growth").mean().over("YEAR")).alias(
            "rent_growth_norm"
        )
    )
    df = df.rename({c: c.lower() for c in df.columns})
    df = df.with_columns(
        pl.col("rent_growth_norm")
        .shift(-1)
        .over("met_name", order_by="year")
        .alias("rent_growth_next"),
        pl.col("inventory_units")
        .pct_change()
        .over("met_name", order_by="year")
        .alias("supply_growth"),
        pl.col("density_rented")
        .diff()
        .over("met_name", order_by="year")
        .alias("density_rented_change"),
    )
    return df
