import polars as pl
import os
import pandas as pd
import csv

# --- CONFIGURATION ---
CSV_PATH_RENT = r"C:\Users\mlarriva\OneDrive - Brookfield\Documents\Github\demand_density\Data\pums_data\usa_00009_rent.csv"
CSV_PATH_OWN = r"C:\Users\mlarriva\OneDrive - Brookfield\Documents\Github\demand_density\Data\pums_data\usa_00011_own.csv"
METRO_PARQUET = r"C:\Users\mlarriva\OneDrive - Brookfield\Documents\Github\demand_density\Data\pums_data\metros2013.parquet"
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "pums_data"
)


# === LOAD & PREPROCESS ===
def preprocess(rent_or_own="rent"):
    if rent_or_own == "rent":
        df = pl.scan_csv(CSV_PATH_RENT).collect()
    if rent_or_own == "own":
        df = pl.scan_csv(CSV_PATH_OWN).collect()
    print("read")
    zfill_map = {
        "YEAR": 4,
        "SAMPLE": 6,
        "SERIAL": 8,
        "NUMPREC": 2,
        "HHWT": 10,
        "GQ": 1,
        "OWNERSHP": 1,
        "OWNERSHPD": 2,
        "MET2013": 5,
    }
    for col, width in zfill_map.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Utf8).str.zfill(width).alias(col))
    print("mapped")
    df = df.with_columns(
        [
            pl.col("HHWT").cast(pl.Float64),
        ]
    )
    print("manipulated")
    df_var = pl.read_parquet(METRO_PARQUET)
    df = df.join(df_var, how="left", left_on=["MET2013"], right_on=["MET2013"])
    print("joined_metro")
    df = df.with_columns(
        pl.col("MET2013").alias("MET_CODE"), pl.col("met2013_name").alias("MET_NAME")
    ).select(
        "YEAR",
        "SAMPLE",
        "SERIAL",
        "NUMPREC",
        "HHWT",
        "MET_NAME",
        "MET_CODE",
        "GQ",
        "OWNERSHP",
    )
    parquet_path = os.path.join(output_dir, f"puma_data_{rent_or_own}.parquet")
    df.write_parquet(parquet_path)
    print(f"Parquet file written to: {parquet_path}")


# === INGEST ====
def ingest(rent_or_own="rent"):
    parquet_path = os.path.join(output_dir, f"puma_data_{rent_or_own}.parquet")
    df = pl.read_parquet(parquet_path).with_columns(
        pl.col("NUMPREC").cast(pl.Float64), pl.col("HHWT").cast(pl.Float64)
    )
    # df = df.filter(pl.col("MET_NAME") == "New York-Newark-Jersey City, NY-NJ-PA")
    # breakpoint()
    return df


def filter_relevant(df, subset):
    if subset == "rent":
        return (
            df.filter(pl.col("OWNERSHP").is_in(["2"]))  # rentals only = 2
            # .filter(pl.col("GQ").is_in(["0", "1", "2", "5"]))
            .filter(~pl.col("MET_NAME").is_null()).filter(
                pl.col("MET_NAME") != "Not in identifiable area"
            )
        )  # households only; not vacant, not group quarters
    if subset == "own":
        return (
            df.filter(pl.col("OWNERSHP").is_in(["1"]))  # rentals only = 2
            # .filter(pl.col("GQ").is_in(["0", "1", "2", "5"]))
            .filter(~pl.col("MET_NAME").is_null()).filter(
                pl.col("MET_NAME") != "Not in identifiable area"
            )
        )  # households only; not vacant, not group quarters


# === FIND PERSONS PER RENTED HOUSEHOLD ===
def renters_per_household():
    df = filter_relevant(ingest(rent_or_own="rent"), subset="rent")
    df = df.select("YEAR", "SERIAL", "MET_NAME", "NUMPREC", "HHWT")
    df = df.group_by(["YEAR", "MET_NAME", "SERIAL"]).agg(
        pl.col("NUMPREC").first(), pl.col("HHWT").first()
    )
    result = (
        df.with_columns((pl.col("NUMPREC") * pl.col("HHWT")).alias("weighted_people"))
        .group_by(["YEAR", "MET_NAME"])
        .agg(
            [
                pl.sum("weighted_people").alias("PPL_IN_RENTALS"),
                pl.sum("HHWT").alias("RENTAL_UNITS"),
            ]
        )
        .with_columns(
            (pl.col("PPL_IN_RENTALS") / pl.col("RENTAL_UNITS")).alias("PPL_PER_RENTAL")
        )
    )
    print(
        result.filter(pl.col("YEAR") == "2023").sort("PPL_IN_RENTALS", descending=True)
    )
    met_to_costar = pl.read_csv(
        r"C:\Users\mlarriva\OneDrive - Brookfield\Documents\Github\demand_density\Data\pums_data\matched_geographies.csv"
    )
    result = result.join(
        met_to_costar, how="left", left_on=["MET_NAME"], right_on=["MET_NAME"]
    )
    print(result.head())
    result.write_csv(os.path.join(output_dir, "wtd_avg_ppl_per_retner_hh.csv"))
    return result


def owners_per_household():
    df = filter_relevant(ingest(rent_or_own="own"), subset="own")
    df = df.select("YEAR", "SERIAL", "MET_NAME", "NUMPREC", "HHWT")
    df = df.group_by(["YEAR", "MET_NAME", "SERIAL"]).agg(
        pl.col("NUMPREC").first(), pl.col("HHWT").first()
    )
    result = (
        df.with_columns((pl.col("NUMPREC") * pl.col("HHWT")).alias("weighted_people"))
        .group_by(["YEAR", "MET_NAME"])
        .agg(
            [
                pl.sum("weighted_people").alias("PPL_IN_OWNED"),
                pl.sum("HHWT").alias("OWNED_UNITS"),
            ]
        )
        .with_columns(
            (pl.col("PPL_IN_OWNED") / pl.col("OWNED_UNITS")).alias("PPL_PER_OWNED")
        )
    )
    print(
        result.filter(pl.col("YEAR") == "2023").sort("PPL_PER_OWNED", descending=True)
    )
    met_to_costar = pl.read_csv(
        r"C:\Users\mlarriva\OneDrive - Brookfield\Documents\Github\demand_density\Data\pums_data\matched_geographies.csv"
    )
    result = result.join(
        met_to_costar, how="left", left_on=["MET_NAME"], right_on=["MET_NAME"]
    )
    print(result.head())
    result.write_csv(os.path.join(output_dir, "wtd_avg_ppl_per_owned_hh.csv"))
    return result


def main():
    # preprocess(rent_or_own="rent")
    renters_per_household()
    # preprocess(rent_or_own="own")
    # ingest(rent_or_own="own")
    owners_per_household()

    parquet_path = os.path.join(output_dir, "puma_data.parquet")
    df = pl.read_parquet(parquet_path)
    filtered = df.filter(
        (pl.col("YEAR") == "2023")
        & (pl.col("MET_NAME").str.to_lowercase().str.contains("los angeles"))
    )
    output_csv = os.path.join(output_dir, "la_2023_rent.csv")
    filtered.write_csv(output_csv)
    print(f"Filtered data written to: {output_csv}")
    parquet_path = os.path.join(output_dir, "puma_data_own.parquet")
    df = pl.read_parquet(parquet_path)
    filtered = df.filter(
        (pl.col("YEAR") == "2023")
        & (pl.col("MET_NAME").str.to_lowercase().str.contains("los angeles"))
    )
    output_csv = os.path.join(output_dir, "la_2023_owned.csv")
    filtered.write_csv(output_csv)
    print(f"Filtered data written to: {output_csv}")


if __name__ == "__main__":
    main()
