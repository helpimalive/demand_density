import polars as pl


def load_data(n_markets=100):
    df = pl.read_parquet(
        r"C:\Users\mlarriva\Documents\om\analyses\projects\rental_density_index\data\intermediate\features\rental_density_indices_enriched.parquet"
    ).with_columns(pl.col("YEAR").cast(pl.Int32))
    largest_met_names = (
        df.group_by("MET_NAME")
        .agg(
            pl.col("YEAR").n_unique().alias("num_years"),
            pl.col("inventory_units").sum().alias("total_inventory"),
        )
        .filter(pl.col("num_years") == pl.col("num_years").max())
        .sort("total_inventory", descending=True)
        .head(n_markets)
        .select("MET_NAME")
        .to_series()
        .to_list()
    )
    df = df.filter(pl.col("MET_NAME").is_in(largest_met_names))
    cpi = (
        pl.read_csv(r"Data\cpi.csv")
        .with_columns((pl.col("year").cast(pl.Int32) - 1).alias("year_next"))
        .select("year_next", "cpi")
    )
    df = df.join(cpi, left_on="YEAR", right_on="year_next", how="left")
    df = df.with_columns(
        (pl.col("rent_growth_next_year") - pl.col("cpi")).alias(
            "real_rent_growth_next_year"
        )
    ).with_columns(
        (
            pl.col("real_rent_growth_next_year")
            .shift(1)
            .over("MET_NAME", order_by="YEAR")
        ).alias("real_rent_growth")
    )

    df = df.with_columns(
        (
            pl.col("real_rent_growth_next_year")
            - pl.col("real_rent_growth_next_year").median().over("YEAR")
        ).alias("real_relative_rent_growth_next_year")
    )
    df = df.with_columns(
        pl.col("real_relative_rent_growth_next_year")
        .shift(1)
        .over("MET_NAME", order_by="YEAR")
        .alias("real_relative_rent_growth_this_year")
    )
    df = df.rename({c: c.lower() for c in df.columns})
    df = df.with_columns(
        pl.col("inventory_units")
        .pct_change()
        .over("met_name", order_by="year")
        .alias("supply_growth"),
        pl.col("density_rented")
        .diff()
        .over("met_name", order_by="year")
        .alias("density_rented_change"),
        (pl.col("households_rented") / pl.col("total_households")).alias("own_percent"),
        pl.col("total_population")
        .pct_change()
        .over("met_name", order_by="year")
        .alias("population_growth"),
    )
    sfh = pl.read_parquet(
        r"C:\Users\mlarriva\Documents\om\analyses\projects\rental_density_index\data\intermediate\features\annual_hh_overview.parquet"
    )
    mapping = (
        pl.read_parquet(
            r"C:\Users\mlarriva\Documents\om\analyses\projects\rental_density_index\data\intermediate\features\rental_density_indices_enriched.parquet"
        )
        .select("MET2013", "MET_NAME")
        .unique()
    )
    sfh = (
        sfh.join(mapping, left_on="MET2013", right_on="MET2013", how="left")
        .with_columns(
            pl.col("total_hh_count")
            / pl.col("total_hh_count")
            .sum()
            .over(["YEAR", "MET_NAME"])
            .alias("housing_share")
        )
        .filter(pl.col("dwelling_super_group") == "SINGLE_FAMILY_DETACHED")
    ).select(
        pl.col("YEAR").cast(pl.Int32).alias("year"),
        pl.col("MET_NAME").alias("met_name"),
        pl.col("total_hh_count").alias("sfh_share"),
    )
    df = df.join(sfh, how="left", on=["met_name", "year"])
    df.write_csv(r"Exhibits\rental_density_index_features.csv")
    return df
