import polars as pl


def load_data(n_markets=100, cached=True):
    if cached:
        return pl.read_csv(r"Exhibits\rental_density_index_features.csv")
    df = (
        pl.read_parquet(
            r"C:\Users\mlarriva\Documents\om\analyses\projects\rental_density_index\data\intermediate\features\rental_density_indices_enriched.parquet"
        )
        .with_columns(pl.col("YEAR").cast(pl.Int32))
        .join(
            pl.read_parquet(
                r"C:\Users\mlarriva\Documents\om\analyses\projects\rental_density_index\data\intermediate\features\renter_household_demographics_metro.parquet"
            ).with_columns(pl.col("YEAR").cast(pl.Int32)),
            how="left",
            left_on=["MET_CODE", "YEAR"],
            right_on=["MET2013", "YEAR"],
        )
        .filter(~pl.col("MET_NAME").str.contains("Orleans"))
    )
    # print(df.columns)

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
        .alias("real_relative_rent_growth_this_year"),
        (
            (pl.col("TOTAL_POPULATION") / ("TOTAL_BEDROOMS")).alias(
                "total_bedroom_density"
            )
        ),
        (
            (
                pl.col("median_owner_hh_income") / pl.col("median_renter_hh_income")
            ).alias("owner_to_renter_income_ratio")
        ),
        ((pl.col("TOTAL_POPULATION") / ("TOTAL_HOUSEHOLDS")).alias("total_density_hh")),
        (
            (pl.col("POPULATION_OWNED") / ("POPULATION_RENTED")).alias(
                "owned_to_rented_pop_ratio"
            )
        ),
    )
    # df = df.with_columns(pl.col())
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
        r"C:\Users\mlarriva\Documents\om\analyses\projects\rental_density_index\data\intermediate\features\annual_sfr_overview_metro.parquet"
    ).with_columns(pl.col("YEAR").cast(pl.Int32))

    df = df.join(
        sfh, how="left", left_on=["met_name", "year"], right_on=["CBSA Title", "YEAR"]
    ).with_columns(
        (
            pl.col("SFR_HOUSEHOLDS")
            / (pl.col("households_owned") + pl.col("households_rented"))
        ).alias("sfr_share")
    )
    df.write_csv(r"Exhibits\rental_density_index_features.csv")
    return df


# load_data(cached=False)
