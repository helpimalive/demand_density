
import polars as pl


df = pl.read_csv(r"C:\Users\datbu\Documents\GitHub\demand_density\Data\emperical_evidence.csv")

print(df)

status_counts = (
    df.group_by('msa').agg([
        pl.col('group').map_elements(lambda x: (x == "overPriced-underSupplied").sum(), return_dtype = pl.Int64).alias("overPriced-underSupplied"),
        pl.col('group').map_elements(lambda x: (x == "overPriced-overSupplied").sum(), return_dtype = pl.Int64).alias("overPriced-overSupplied"),
        pl.col('group').map_elements(lambda x: (x == "underPriced-underSupplied").sum(), return_dtype = pl.Int64).alias("underPriced-underSupplied"),
        pl.col('group').map_elements(lambda x: (x == "underPriced-overSupplied").sum(), return_dtype = pl.Int64).alias("underPriced-overSupplied")
    ])
)

pl.Config.set_tbl_rows(1000)

pl.Config.set_fmt_str_lengths(100)

print(status_counts)

status_counts.write_csv('msa-price-supply.csv')
