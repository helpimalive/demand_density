import pandas as pd
import numpy as np
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
import matplotlib.pyplot as plt
from preprocess import load_data

def placebo_one_nominal_vs_real():
    # ── Load and prepare data (mirrors model_ardl exactly) ────────────────────
    df = (
        load_data(100, cached=False).select(
            "year",
            "met_name",
            "density_rented",
            "supply_growth",
            "rent_growth",
            "real_rent_growth",
            "bedroom_density_rented",
            "real_rent_growth_next_year",
            "rent_growth_next_year",
        )
        .to_pandas()
        .set_index(["met_name", "year"])
        .sort_index()
    )
    # print(df)

    # ── Placebo 1: Nominal rent growth as outcome ──────────────────────────────
    #
    df["nominal_rent_growth_next_year"] = df["rent_growth_next_year"]
    df["delta_rdi"]    = df.groupby(level="met_name")["density_rented"].diff()
    df["delta_supply"] = df.groupby(level="met_name")["supply_growth"].diff()
    df["rdi_lag1"]     = df.groupby(level="met_name")["density_rented"].shift(1)
    df["supply_lag1"]  = df.groupby(level="met_name")["supply_growth"].shift(1)

    ardl_vars = [
        "rdi_lag1",
        "delta_rdi",
        "supply_lag1",
        "delta_supply",
        "rent_growth",
        "bedroom_density_rented"
    ]
    df_nominal = df[["nominal_rent_growth_next_year"] + ardl_vars].dropna()
    placebo_nominal = PanelOLS(
        df_nominal["nominal_rent_growth_next_year"],
        sm.add_constant(df_nominal[ardl_vars]),
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)
    print("\n── Placebo 1: Nominal Rent Growth as Outcome ──")
    print(placebo_nominal.summary)
def placebo_two_randomized_rdi():
    # ── Placebo 2: Randomized RDI across metros within year ────────────────────
    # shuffle rdi_lag1 and delta_rdi across metros within each year
    # repeat n_iter times and collect coefficients on rdi_lag1 and delta_rdi
    # ── Load and prepare data (mirrors model_ardl exactly) ────────────────────
    df = (
        load_data(100, cached=False).select(
            "year",
            "met_name",
            "density_rented",
            "supply_growth",
            "rent_growth",
            "real_rent_growth",
            "bedroom_density_rented",
            "real_rent_growth_next_year",
            "rent_growth_next_year",
        )
        .to_pandas()
        .set_index(["met_name", "year"])
        .sort_index()
    )

    df["delta_rdi"]    = df.groupby(level="met_name")["density_rented"].diff()
    df["delta_supply"] = df.groupby(level="met_name")["supply_growth"].diff()
    df["rdi_lag1"]     = df.groupby(level="met_name")["density_rented"].shift(1)
    df["supply_lag1"]  = df.groupby(level="met_name")["supply_growth"].shift(1)

    ardl_vars = [
        "rdi_lag1",
        "delta_rdi",
        "supply_lag1",
        "delta_supply",
        "real_rent_growth",
        "bedroom_density_rented"
    ]

    df_clean = df[["real_rent_growth_next_year"] + ardl_vars].dropna()

    # ── Recover true coefficients from baseline model ─────────────────────────
    baseline_model = PanelOLS(
        df_clean["real_rent_growth_next_year"],
        sm.add_constant(df_clean[ardl_vars]),
        entity_effects=True,
        time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    true_coef_lag1     = baseline_model.params["rdi_lag1"]
    true_coef_delta    = baseline_model.params["delta_rdi"]
    print(f"  True rdi_lag1 coefficient:  {true_coef_lag1:.4f}")
    print(f"  True delta_rdi coefficient: {true_coef_delta:.4f}")

    # ── Placebo iterations ────────────────────────────────────────────────────
    n_iter = 1000
    rng    = np.random.default_rng(42)
    coefs_lag1  = []
    coefs_delta = []

    for i in range(n_iter):
        df_placebo = df_clean.copy()

        df_placebo = df_placebo.reset_index()
        df_placebo["rdi_lag1"] = (
            df_placebo
            .groupby("year")["rdi_lag1"]
            .transform(lambda x: x.values[rng.permutation(len(x))])
        )
        df_placebo["delta_rdi"] = (
            df_placebo
            .groupby("year")["delta_rdi"]
            .transform(lambda x: x.values[rng.permutation(len(x))])
        )
        df_placebo = df_placebo.set_index(["met_name", "year"])

        model = PanelOLS(
            df_placebo["real_rent_growth_next_year"],
            sm.add_constant(df_placebo[ardl_vars]),
            entity_effects=True,
            time_effects=True,
        ).fit(cov_type="clustered", cluster_entity=True)

        coefs_lag1.append(model.params["rdi_lag1"])
        coefs_delta.append(model.params["delta_rdi"])

        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{n_iter} iterations...")

    coefs_lag1  = np.array(coefs_lag1)
    coefs_delta = np.array(coefs_delta)

    # ── Summary stats ─────────────────────────────────────────────────────────
    for label, coefs, true_coef in [
        ("rdi_lag1",  coefs_lag1,  true_coef_lag1),
        ("delta_rdi", coefs_delta, true_coef_delta),
    ]:
        ci_low, ci_high = np.percentile(coefs, [2.5, 97.5])
        outside = not (ci_low <= true_coef <= ci_high)
        print(f"\n── Placebo: {label} ──")
        print(f"  Mean coefficient:          {coefs.mean():.5f}")
        print(f"  95% empirical interval:    [{ci_low:.3f}, {ci_high:.3f}]")
        print(f"  True coefficient:          {true_coef:.4f}")
        print(f"  True coef outside interval: {outside}")

    # ── Plot: side-by-side panels ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    plot_specs = [
        (axes[0], coefs_lag1,  true_coef_lag1,  "Randomized $\\text{RDI}_{m,t}$ (Lagged Level)"),
        (axes[1], coefs_delta, true_coef_delta, "Randomized $\\Delta\\text{RDI}_{m,t}$ (Change)"),
    ]

    for ax, coefs, true_coef, title in plot_specs:
        ci_low, ci_high = np.percentile(coefs, [2.5, 97.5])
        ax.hist(coefs, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(coefs.mean(), color="navy",    linestyle="--",
                   label=f"Mean: {coefs.mean():.4f}")
        ax.axvline(ci_low,       color="gray",    linestyle=":",
                   label=f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
        ax.axvline(ci_high,      color="gray",    linestyle=":")
        ax.axvline(true_coef,    color="crimson", linestyle="-",
                   label=f"True coef: {true_coef:.4f}")
        ax.set_xlabel("Placebo Coefficient")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle("Placebo Distribution: Randomized RDI Within Year", y=1.02)
    plt.tight_layout()
    plt.savefig(r"Exhibits\placebo_randomized_rdi.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Save results ──────────────────────────────────────────────────────────
    pd.DataFrame({
        "placebo_coef_lag1":  coefs_lag1,
        "placebo_coef_delta": coefs_delta,
    }).to_csv(r"Exhibits\placebo_randomized_coefs.csv", index=False)


placebo_two_randomized_rdi()