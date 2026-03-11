from a_clean_data import prepare_data
import polars as pl


def highest_allocation(df):
    # df = df.filter(pl.col("date") == pl.col("date").max())
    df = df.filter(pl.col("period") == "2025_Q4")
    with pl.Config(set_tbl_width_chars=100):
        print(
            df.filter(pl.col("variable") == "allocation")
            .filter(~pl.col("value").is_null())
            .sort("value", descending=True)
            .head(5)
        )
        print(
            df.filter(pl.col("variable") == "allocation")
            .filter(~pl.col("value").is_null())
            .group_by("sector")
            .agg(pl.col("value").sum())
            .sort("value", descending=True)
        )


df = prepare_data()
print(highest_allocation(df))
