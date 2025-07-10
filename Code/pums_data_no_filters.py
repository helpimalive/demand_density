import polars as pl
import os
import pandas as pd
import csv

# --- CONFIGURATION ---
CSV_PATH = r"C:\Users\mlarriva\OneDrive - Brookfield\Documents\Github\demand_density\Data\pums_data\usa_0013.csv"
METRO_PARQUET = r"C:\Users\mlarriva\OneDrive - Brookfield\Documents\Github\demand_density\Data\pums_data\metros2013.parquet"
output_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "pums_data"
)


# === LOAD & PREPROCESS ===
def preprocess():
    df = pl.scan_csv(CSV_PATH)
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
        "PERNUM": 4,
        "CBPERNUM": 2,
        "PERWT": 10,
    }
    for col, width in zfill_map.items():
        if col in df.collect_schema().names():
            df = df.with_columns(pl.col(col).cast(pl.Utf8).str.zfill(width).alias(col))
    print("mapped")
    df = df.with_columns(
        [
            pl.col("HHWT").cast(pl.Float64),
            pl.col("PERWT").cast(pl.Float64),
            pl.col("PERNUM").cast(pl.Float64),
        ]
    )
    print("manipulated")
    df_var = pl.scan_parquet(METRO_PARQUET)
    df = df.join(df_var, how="left", left_on=["MET2013"], right_on=["MET2013"])
    df = df.with_columns(
        pl.col("MET2013").alias("MET_CODE"), pl.col("met2013_name").alias("MET_NAME")
    )
    print("joined_metro")
    filtered = df.filter(
        (pl.col("YEAR").is_in(["2011", "2019", "2023"]))
        & (pl.col("MET_NAME").str.to_lowercase().str.contains("los angeles"))
    )
    output_csv = os.path.join(output_dir, "la_2023.csv")
    filtered.collect().write_csv(output_csv)
    print("subset_written_to_csv")
    df = df.group_by(["YEAR", "SERIAL", "MET_NAME", "GQ", "OWNERSHP"]).agg(
        pl.col("PERWT").sum().alias("POPULATION"),
        pl.col("HHWT").first().alias("HOUSEHOLDS"),
    )
    print("reduced to population and household")
    parquet_path = os.path.join(output_dir, "puma_data_no_filters.parquet")
    df.collect().write_parquet(parquet_path)
    print(f"Parquet file written to: {parquet_path}")


# === INGEST ====
def ingest():
    parquet_path = os.path.join(output_dir, f"puma_data_no_filters.parquet")
    df = pl.read_parquet(parquet_path)
    return df


# === FIND POPULATION AND DENSITY METRICS AT THE METRO LEVEL ====
def population():
    df = ingest()
    met_to_costar = pl.read_csv(
        r"C:\Users\mlarriva\OneDrive - Brookfield\Documents\Github\demand_density\Data\pums_data\matched_geographies.csv"
    )
    df = df.with_columns(
        pl.when(pl.col("OWNERSHP") == "0")
        .then(pl.lit("OTHER"))
        .when(pl.col("OWNERSHP") == "1")
        .then(pl.lit("OWNED"))
        .when(pl.col("OWNERSHP") == "2")
        .then(pl.lit("RENTED"))
        .otherwise(pl.col("OWNERSHP"))
        .alias("OWNERSHP")
    )
    df = (
        (
            df.group_by(["YEAR", "MET_NAME", "OWNERSHP"])
            .agg(
                pl.col("POPULATION").sum().alias("POPULATION"),
                pl.col("HOUSEHOLDS").sum().alias("HOUSEHOLDS"),
            )
            .pivot(
                index=["YEAR", "MET_NAME"],
                columns="OWNERSHP",
                values=["POPULATION", "HOUSEHOLDS"],
            )
        )
        .with_columns(pl.col("POPULATION_OTHER").fill_null(0))
        .with_columns(
            TOTAL_POPULATION=pl.col("POPULATION_OWNED")
            + pl.col("POPULATION_RENTED")
            + pl.col("POPULATION_OTHER"),
            TOTAL_HOUSEHOLDS=pl.col("HOUSEHOLDS_OWNED")
            + pl.col("HOUSEHOLDS_RENTED")
            + pl.col("HOUSEHOLDS_OTHER"),
        )
        .with_columns(
            DENSITY_OWNED=(pl.col("POPULATION_OWNED") / pl.col("HOUSEHOLDS_OWNED")),
            DENSITY_RENTED=(pl.col("POPULATION_RENTED") / pl.col("HOUSEHOLDS_RENTED")),
            DENSITY_OTHER=(pl.col("POPULATION_OTHER") / pl.col("HOUSEHOLDS_OTHER")),
        )
    )
    result = df.join(
        met_to_costar, how="left", left_on=["MET_NAME"], right_on=["MET_NAME"]
    )
    print(result.filter(pl.col("MET_NAME").str.contains("Los Angeles")).sort("YEAR"))
    result.write_csv(os.path.join(output_dir, "puma_metro_pop_density.csv"))
    return result


def main():
    preprocess()
    df = population()


if __name__ == "__main__":
    main()
