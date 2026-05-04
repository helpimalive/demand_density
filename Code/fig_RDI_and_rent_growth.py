import matplotlib.pyplot as plt
import polars as pl
from preprocess import load_data
from matplotlib.ticker import MaxNLocator
import numpy as np
import os


def graph():
    
    df = (
        (
            load_data().select(
                "year",
                "met_name",
                "real_relative_rent_growth_next_year",
                "density_rented_change",
                "density_rented",
            )
        )
        .group_by("met_name")
        .agg(
            pl.col("density_rented").mean().alias("density_rented"),
            (10000 * pl.col("real_relative_rent_growth_next_year").mean()).alias(
                "real_rent_growth"
            ),
        )
    )

    plt.figure(figsize=(8, 6))
    x = np.array(df["density_rented"].to_list(), dtype=float)
    y = np.array(df["real_rent_growth"].to_list(), dtype=float)

    # drop any non-finite pairs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    plt.scatter(x, y, alpha=0.6, s=30, edgecolor="k")

    # fit a first-degree polynomial (line of best fit) and plot it
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = np.polyval(coeffs, x_line)
    # compute and print R^2
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    print(f"R^2 = {r2:.4f}")
    plt.plot(x_line, y_line, color="red", linewidth=2, label=f"R^2 = {r2:.2f}")
    plt.legend()
    plt.xlabel("Rental Household Density")
    plt.ylabel("Real Relative Rent Growth (bps)")
    plt.title(
        "Rental Household Density vs Real Relative Rent Growth, 100 Largest MSAs: 2005-2024"
    )
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figs")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(__file__))[0]
    out_path = os.path.join(out_dir, f"{base_name}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Figure saved to: {out_path}")

    return plt


graph()
