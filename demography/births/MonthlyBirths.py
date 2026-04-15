""" MonthlyBirths.py

Combines the following to get estimated monthly birth counts by province:
    (1) Yearly birth rate estimates from YearlyBirths.py
    (2) Seasonality profiles from BirthSeasonality.py
    (3) Province population estimates (from World Bank national population 
        series and estimated province population shares)

Pakistan complication:
  - WB national population + CBR exclude AJK and Gilgit Baltistan.
  - Main province population shares come from DHS weights (sum to 1.0).
  - AJK/GB shares come from census touchpoints, expressed as fractions of
    the WB population (additive mass on top of 1.0).
  - Province population = pop_wb * popshare for all provinces.

Dependencies:
  - extrapolate_trends
  - World Bank population series CSV in _data folder
  - ../../pickle_jar/birth_rates_by_province.pkl 
  - ../../pickle_jar/birth_seasonality_by_province.pkl

Outputs:
  - ../../pickle_jar/monthly_births_by_province.pkl
  - ../../_plots/monthly_births_by_province.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from demography.extrapolate_trends import (
    GetWBPopulation,
    compute_subregion_popshares, extend_popshares,
    compute_additive_territory_popshares,
)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

COUNTRY = "pakistan"

IN_PROV_RATES = "../../pickle_jar/birth_rates_by_province.pkl"
IN_SEASONALITY = "../../pickle_jar/birth_seasonality_by_province.pkl"
IN_MOM_DIST = "../../pickle_jar/mom_distribution_main.pkl"

OUT_MONTHLY = "../../pickle_jar/monthly_births_by_province.pkl"
OUT_PDF = "../../_plots/monthly_births_by_province.pdf"

# Data file paths
DATA_DIR = "../../_data/"
WB_POPULATION_FILE = DATA_DIR + "Population.csv"

PLOT_START_YEAR = 2004

# AJK/GB population as % of WB national population (census touchpoints)
TERRITORY_TOUCHPOINTS = {
    1998: {"ajk": 2.27, "gb": 0.68},
    2023: {"ajk": 1.64, "gb": 0.61},
}

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def axes_setup(ax):
    ax.spines["left"].set_position(("axes", -0.025))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def renormalize_month_shares(seas_block: pd.DataFrame) -> pd.DataFrame:
    """
    Renormalize avg so months sum to 1 within each year.
    """
    seas_block = seas_block.copy()
    sums = seas_block["avg"].groupby(level=0).sum().replace(0.0, np.nan)
    seas_block["avg"] = (seas_block["avg"] / sums).fillna(1.0 / 12.0)
    return seas_block

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":
    # Load province-year birth rates (from YearlyBirths)
    rates = pd.read_pickle(IN_PROV_RATES).copy()
    rates["year"] = rates["year"].astype(int)

    rates = rates.rename(columns={"br_final": "br_per_1000"})
    rates["br_prob"] = rates["br_per_1000"] / 1000.0
    rates["count_var"] = rates["count_var"].astype(float)
        
    provinces = sorted(rates["province"].unique())
    
    # Load WB national population (WB universe excludes AJK/GB)
    pop = GetWBPopulation(WB_POPULATION_FILE, min_year=rates["year"].min(), countries=[COUNTRY])
    pop["year"] = pop["year"].astype(int)
    pop_wb = pop[["year", "population"]].set_index("year")["population"].astype(float)

    # Build population shares
    year_min = min(min(TERRITORY_TOUCHPOINTS.keys()), int(rates["year"].min()), int(pop_wb.index.min()))
    year_max = max(max(TERRITORY_TOUCHPOINTS.keys()), int(rates["year"].max()), int(pop_wb.index.max()))

    # DHS-based shares for main provinces
    dist_main = pd.read_pickle(IN_MOM_DIST)
    dist_main["year"] = dist_main["year"].astype(int)
    dhs_popshares = compute_subregion_popshares(dist_main, group_col="province", year_col="year")
    dhs_popshares = extend_popshares(
        dhs_popshares, year_min=year_min, year_max=year_max,
    )

    # Census-based additive shares for AJK/GB
    ajkgb_popshares = compute_additive_territory_popshares(
        TERRITORY_TOUCHPOINTS,
        year_min=year_min, year_max=year_max,
    )

    # Combine into one long-form DataFrame, then pivot to wide
    all_popshares = pd.concat([dhs_popshares, ajkgb_popshares], ignore_index=True)
    pop_frac = all_popshares.pivot(index="year", columns="province", values="popshare")

    # Load seasonality
    seasonality = pd.read_pickle(IN_SEASONALITY)
    seasonality = seasonality.reset_index()
    seasonality = seasonality.set_index(["province", "year", "month"]).sort_index()

    # Build monthly births per province (province-specific start years)
    outputs = []

    for prov in provinces:
        r_p = rates.loc[rates["province"] == prov].copy().sort_values("year")

        y0 = int(r_p["year"].min())
        y1 = int(r_p["year"].max())
        years_p = np.arange(y0, y1 + 1, dtype=int)

        # Birth rate time series for this province
        r_p = r_p.set_index("year").reindex(years_p)
        r_p["br_prob"] = r_p["br_prob"].ffill().bfill()

        # Population share for this province over the same years
        share_p = pop_frac[prov].reindex(years_p).interpolate(limit_direction="both").values

        # Province population = pop_wb * share
        # For main provinces (shares sum to 1), this splits WB pop correctly.
        # For AJK/GB (additive shares), this gives pop relative to WB universe.
        pop_wb_p = pop_wb.reindex(years_p).interpolate(limit_direction="both").values
        N = pop_wb_p * share_p

        # Yearly births
        p = np.clip(r_p["br_prob"].values, 0.0, 1.0)
        births_year_avg = N * p
        
        cv = r_p["count_var"].values
        if np.all(np.isfinite(cv)):
            # Cell-level variance: multiply by N 
            # count_var = Σ w_i * p_i * (1-p_i), needs one factor of N
            births_year_var = N * cv
        else:
            # Fallback: single-cell binomial 
            births_year_var = N * p * (1.0 - p)

        births_year_df = pd.DataFrame(
            {"avg": births_year_avg, "var": births_year_var},
            index=pd.Index(years_p, name="year")
        )

        # Seasonality block for this province
        seas_p = seasonality.loc[seasonality.index.get_level_values("province") == prov].copy()
        seas_p = seas_p.reset_index().set_index(["year", "month"]).sort_index()
        full_idx = pd.MultiIndex.from_product([years_p, range(1, 13)], names=["year", "month"])
        seas_p = seas_p.reindex(full_idx)

        # Fill within month across years
        seas_p["avg"] = seas_p["avg"].groupby(level=1).apply(lambda s: s.ffill().bfill())
        seas_p["var"] = seas_p["var"].groupby(level=1).apply(lambda s: s.ffill().bfill()).fillna(0.0)

        seas_p = renormalize_month_shares(seas_p)

        # Distribute to months
        ExpN = np.repeat(births_year_df["avg"].values, 12)
        VarN = np.repeat(births_year_df["var"].values, 12)

        p_m = np.clip(seas_p["avg"].values, 0.0, 1.0)
        var_p = np.clip(seas_p["var"].values, 0.0, np.inf)

        exp_monthly = ExpN * p_m

        var_monthly = (
            ExpN * p_m * (1.0 - p_m) +      # multinomial/binomial allocation
            VarN * (p_m ** 2) +             # yearly-birth uncertainty
            var_p * (ExpN ** 2)             # seasonality uncertainty 
        )

        monthly_df = pd.DataFrame(
            {
                "province": prov,
                "year": np.repeat(years_p, 12),
                "month": np.tile(np.arange(1, 13), len(years_p)),
                "avg": exp_monthly,
                "var": var_monthly,
            }
        )
        monthly_df["std"] = np.sqrt(np.clip(monthly_df["var"].values, 0.0, np.inf))
        monthly_df["time"] = pd.to_datetime({"year": monthly_df["year"], "month": monthly_df["month"], "day": 1})
        monthly_df = monthly_df[["province", "time", "avg", "var", "std"]].set_index(["province", "time"]).sort_index()

        outputs.append(monthly_df)

    monthly_births = pd.concat(outputs).sort_index()

    print("\nOverall result:")
    print(monthly_births.head())
    print(monthly_births.tail())

    monthly_births.to_pickle(OUT_MONTHLY)
    print(f"\nSaved monthly births to: {OUT_MONTHLY}")

    # ------------------------------------------------------------
    # Plot book
    # ------------------------------------------------------------
    with PdfPages(OUT_PDF) as book:
        print("\nMaking a book of plots...")

        for prov, sf in monthly_births.groupby("province"):
            ts = sf.loc[prov]
            ts = ts[ts.index >= pd.Timestamp(PLOT_START_YEAR, 1, 1)]

            fig, ax = plt.subplots(figsize=(12, 5))
            axes_setup(ax)
            ax.grid(color="grey", alpha=0.3)

            ax.fill_between(
                ts.index,
                (ts["avg"] - 2.0 * ts["std"]).values,
                (ts["avg"] + 2.0 * ts["std"]).values,
                edgecolor="None",
                alpha=0.4,
            )

            ax.plot(ts.index, ts["avg"].values, lw=3)

            ax.set_ylabel("Monthly births")
            fig.suptitle("Monthly births in " + prov.upper())
            fig.tight_layout(rect=[0, 0.0, 1, 0.9])
            book.savefig(fig)
            plt.close(fig)

    print(f"...done! Saved plot book to: {OUT_PDF}")