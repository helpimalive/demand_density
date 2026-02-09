import numpy as np
import polars as pl
from preprocess import load_data
from linearmodels.panel import PanelOLS
from linearmodels.iv import IV2SLS


def create_2sls_data():
    # msa_id | year | outcome | bedroom_density_rented | sfh_share | is_ca
    df = load_data(100)
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
        pl.col("MET_NAME").alias("msa_name"),
        pl.col("total_hh_count").alias("sfh_share"),
    )
    ca_msas = [
        "Los Angeles-Long Beach-Anaheim, CA",
        "San Diego-Carlsbad, CA",
        "San Francisco-Oakland-Hayward, CA",
    ]
    df = (
        (
            df.join(
                sfh,
                left_on=["year", "met_name"],
                right_on=["year", "msa_name"],
                how="left",
            )
            .select(
                "met_name",
                "year",
                "real_relative_rent_growth_next_year",
                "density_rented",
                "sfh_share",
            )
            .with_columns(
                pl.col("met_name").str.contains(", CA").alias("is_ca"),
                (
                    pl.when(pl.col("year") > 2015).then(pl.lit(1)).otherwise(pl.lit(0))
                ).alias("post2019"),
            )
        )
        .with_columns(
            pl.col("real_relative_rent_growth_next_year").alias("outcome"),
            (pl.col("sfh_share") * pl.col("post2019")).alias("adu_iv"),
        )
        .with_columns((pl.col("adu_iv") * pl.col("is_ca")).alias("adu_iv"))
        .to_pandas()
    )
    df = df.set_index(["met_name", "year"])
    return df


def run_2sls(panel_df):
    ols = PanelOLS(
        panel_df["outcome"],
        panel_df[["density_rented"]],
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    print("\n=== FE OLS ===")
    print(ols.summary)

    first_stage = PanelOLS(
        panel_df["density_rented"],
        panel_df[["adu_iv"]],
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    print("\n=== First Stage ===")
    print(first_stage.summary)

    # =========================
    # 7. Double-demean for FE-IV
    # =========================
    def double_demean(x, entity, time):
        return (
            x
            - x.groupby(entity).transform("mean")
            - x.groupby(time).transform("mean")
            + x.mean()
        )

    for v in ["outcome", "density_rented", "adu_iv"]:
        panel_df[v + "_dd"] = double_demean(
            panel_df[v],
            panel_df.index.get_level_values(0),
            panel_df.index.get_level_values(1),
        )

    # =========================
    # 8. FE 2SLS
    # =========================

    iv_model = IV2SLS(
        dependent=panel_df["outcome_dd"],
        exog=None,
        endog=panel_df["density_rented_dd"],
        instruments=panel_df[["adu_iv_dd"]],
    )

    iv_res = iv_model.fit(
        cov_type="clustered",
        clusters=panel_df.index.get_level_values(0)[iv_model.notnull],
    )

    print("\n=== FE 2SLS ===")
    print(iv_res.summary)

    # =========================
    # 9. Reduced form (optional check)
    # =========================
    rf = PanelOLS(
        panel_df["outcome"],
        panel_df[["adu_iv"]],
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    print("\n=== Reduced Form ===")
    print(rf.summary)


if __name__ == "__main__":
    df_2sls = create_2sls_data()
    run_2sls(df_2sls)
