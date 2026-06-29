import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
from preprocess import load_data


def dynamic_ordering_tests():
    """
    Dynamic ordering tests for the current ARDL specification.
    
    Test 1 (B6 analog): Does RDI predict future rent growth?
             (forward direction - should be significant)
    Test 2 (B7 analog): Does rent growth predict future RDI?
             (reverse direction - should be insignificant)
    """
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from linearmodels import PanelOLS

    # ── Load and prepare data (mirrors model_ardl exactly) ────────────────────
    df = (
        load_data(100, cached=False).select(
            "year",
            "met_name",
            "density_rented",
            "supply_growth",
            "real_rent_growth",
            "bedroom_density_rented",
            "real_rent_growth_next_year",
        )
        .to_pandas()
        .set_index(["met_name", "year"])
        .sort_index()
    )

    df["delta_rdi"]    = df.groupby(level="met_name")["density_rented"].diff()
    df["delta_supply"] = df.groupby(level="met_name")["supply_growth"].diff()
    df["rdi_lag1"]     = df.groupby(level="met_name")["density_rented"].shift(1)
    df["supply_lag1"]  = df.groupby(level="met_name")["supply_growth"].shift(1)

    # next-period RDI (for reverse test)
    df["rdi_next"]     = df.groupby(level="met_name")["density_rented"].shift(-1)

    # ── Test 1: RDI → future rent growth (forward direction) ──────────────────
    # Spec mirrors baseline ARDL exactly
    ardl_vars = [
        "rdi_lag1",
        "delta_rdi",
        "supply_lag1",
        "delta_supply",
        "real_rent_growth",
        "bedroom_density_rented",
    ]

    df_fwd = df[["real_rent_growth_next_year"] + ardl_vars].dropna()

    model_fwd = PanelOLS(
        df_fwd["real_rent_growth_next_year"],
        sm.add_constant(df_fwd[ardl_vars]),
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    print("\n── Dynamic Ordering Test 1: RDI → Future Rent Growth ──")
    print("   (Forward direction; should be significant)")
    print(model_fwd.summary)

    # ── Test 2: rent growth → future RDI (reverse direction) ──────────────────
    # Dependent variable: next-period RDI level
    # Regressors: current rent growth, current RDI level (persistence control),
    #             supply controls, bedroom density
    reverse_vars = [
        "real_rent_growth",   # this is the key variable of interest
        "rdi_lag1",           # RDI persistence (analogous to B7's EC_t)
        "supply_lag1",
        "delta_supply",
        "bedroom_density_rented",
    ]

    df_rev = df[["rdi_next"] + reverse_vars].dropna()

    model_rev = PanelOLS(
        df_rev["rdi_next"],
        sm.add_constant(df_rev[reverse_vars]),
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    print("\n── Dynamic Ordering Test 2: Rent Growth → Future RDI ──")
    print("   (Reverse direction; should be insignificant)")
    print(model_rev.summary)

    # ── Clean summary table ───────────────────────────────────────────────────
    print("\n── Dynamic Ordering: Key Coefficients ──")
    print(f"\n  Test 1 — RDI_lag1 → next rent growth:")
    print(f"    Coefficient: {model_fwd.params['rdi_lag1']:.4f}")
    print(f"    Std. Error:  {model_fwd.std_errors['rdi_lag1']:.4f}")
    print(f"    P-value:     {model_fwd.pvalues['rdi_lag1']:.4f}")

    print(f"\n  Test 1 — delta_RDI → next rent growth:")
    print(f"    Coefficient: {model_fwd.params['delta_rdi']:.4f}")
    print(f"    Std. Error:  {model_fwd.std_errors['delta_rdi']:.4f}")
    print(f"    P-value:     {model_fwd.pvalues['delta_rdi']:.4f}")

    print(f"\n  Test 2 — Rent growth → next RDI:")
    print(f"    Coefficient: {model_rev.params['real_rent_growth']:.4f}")
    print(f"    Std. Error:  {model_rev.std_errors['real_rent_growth']:.4f}")
    print(f"    P-value:     {model_rev.pvalues['real_rent_growth']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = pd.DataFrame({
        "test":        ["fwd_rdi_lag1", "fwd_delta_rdi", "rev_rent_growth"],
        "coefficient": [
            model_fwd.params["rdi_lag1"],
            model_fwd.params["delta_rdi"],
            model_rev.params["real_rent_growth"],
        ],
        "std_error": [
            model_fwd.std_errors["rdi_lag1"],
            model_fwd.std_errors["delta_rdi"],
            model_rev.std_errors["real_rent_growth"],
        ],
        "pvalue": [
            model_fwd.pvalues["rdi_lag1"],
            model_fwd.pvalues["delta_rdi"],
            model_rev.pvalues["real_rent_growth"],
        ],
    })
    results.to_csv(r"Exhibits\dynamic_ordering_results.csv", index=False)
    print("\n  Results saved to Exhibits\\dynamic_ordering_results.csv")


dynamic_ordering_tests()