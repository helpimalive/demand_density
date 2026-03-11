import polars as pl
import re
import os

raw_exports_dir = os.path.join(os.path.dirname(__file__), "raw_exports")


def raw_ingest():
    data = []
    for file_name in os.listdir(raw_exports_dir):
        if file_name.endswith(".csv") and file_name.startswith("data_export_"):
            parts = file_name.split("_")
            year = parts[2]
            quarter = parts[3].split(".")[0]
            period = f"{year}_{quarter}"
            if 2020 > int(year) > 2030:
                raise Exception("Unlikely year parsed; code stopped")
            if quarter not in ["Q1", "Q2", "Q3", "Q4"]:
                raise Exception(f"Unacceptable quarter found in file {file_name}")

            file_path = os.path.join(raw_exports_dir, file_name)
            df = pl.read_csv(file_path)
            # df = pl.read_excel(file_path)
            expected_columns = [
                "CSA Name",
                "Capital Pool",
                "Investment",
                "Property Name",
                "Brookfield",
                "Market",
                r"% of Portfolio",
                "Brookfield_duplicated_0",
                "Market_duplicated_0",
                r"% of Portfolio_duplicated_0",
                "Brookfield_duplicated_1",
                "Market_duplicated_1",
                r"% of Portfolio_duplicated_1",
                "Brookfield_duplicated_2",
                "Market_duplicated_2",
                r"% of Portfolio_duplicated_2",
                "Brookfield_duplicated_3",
                "Market_duplicated_3",
                r"% of Portfolio_duplicated_3",
                " ",
            ]
            if not df.columns == expected_columns:
                raise Exception("Columns ingested do not match expected columns")
            df = df.drop(" ")
            df = df.with_columns(pl.lit(period).alias("period"))
            data.append(df)
    return pl.concat(data)


def clean_raw(df):
    id_cols = [
        "CSA Name",
        "Capital Pool",
        "Investment",
        "Property Name",
        "period",
        "sector",
    ]
    results = []

    out = (
        df.with_columns(pl.lit("Hospitality").alias("sector"))
        .select(
            id_cols
            + [
                pl.col(f"Brookfield").alias("BAM"),
                pl.col(f"Market").alias("market"),
                pl.col(rf"% of Portfolio").alias("allocation"),
            ]
        )
        .unpivot(
            index=id_cols,
            variable_name="variable",
            value_name="value",
        )
    )
    results.append(out)
    num_sec = [(0, "Logistics"), (1, "Multifamily"), (2, "Office"), (3, "Retail")]
    for num, sec in num_sec:
        out = (
            df.with_columns(pl.lit(sec).alias("sector"))
            .select(
                id_cols
                + [
                    pl.col(f"Brookfield_duplicated_{num}").alias("BAM"),
                    pl.col(f"Market_duplicated_{num}").alias("market"),
                    pl.col(rf"% of Portfolio_duplicated_{num}").alias("allocation"),
                ]
            )
            .unpivot(
                index=id_cols,
                variable_name="variable",
                value_name="value",
            )
        )
        results.append(out)
    results = pl.concat(results)
    results.columns = [
        "csa",
        "pool",
        "investment",
        "property",
        "period",
        "sector",
        "variable",
        "value",
    ]
    results = results.with_columns(
        pl.col("period")
        .map_elements(
            lambda p: f"{p.split('_')[0]}-{int(p.split('_')[1][1:]) * 3 - 2:02d}-01",
            return_dtype=pl.Utf8,
        )
        .str.strptime(pl.Date, "%Y-%m-%d")
        .alias("date")
    )
    results = results.with_columns(
        pl.col("value")
        .map_elements(
            lambda v: (
                float(v.strip("%")) / 100
                if isinstance(v, str) and "%" in v
                else float(v)
            ),
            return_dtype=pl.Float64,
        )
        .alias("value")
    )
    return results


def prepare_data():
    return clean_raw(raw_ingest())
