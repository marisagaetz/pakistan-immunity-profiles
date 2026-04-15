""" CohortPriors.py

Age-at-infection estimates by birth cohort, stratified at province level,
using the masked softmax smoother (masked_buckets).

The model estimates the full age-at-infection distribution for each birth
cohort, including ages that are not yet observable for recent cohorts.
It does this by fitting a penalized softmax over observable cells and
letting the smoothness prior (RW2 over cohorts + group-to-group over
ages) propagate information from older cohorts into younger ones.

The core logic (masked_buckets) is pulled directly from the Nigeria 
workflow.

Dependencies:
  - ../pickle_jar/combined_linelist_regressed.pkl

Outputs:
  - ../pickle_jar/cohort_age_priors.pkl
  - ../_plots/cohort_age_prior_fits.pdf
  - ../_plots/cohort_age_priors.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import masked_buckets as mb

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

LINELIST_PKL = "../pickle_jar/combined_linelist_regressed.pkl"
OUT_PKL = "../pickle_jar/cohort_age_priors.pkl"
OUT_PDF_PANELS = "../_plots/cohort_age_prior_fits.pdf"
OUT_PDF_GRIDS = "../_plots/cohort_age_priors.pdf"

COHORT_START = 2004
COHORT_END = 2025
AGE_MAX = 25

# Smoother parameters 
CORRELATION_TIME = 10.0
G2G_CORRELATION = 4.0

PROVINCES = ["punjab", "sindh", "kp", "balochistan", "ict", "ajk", "gb"]
REGIONS = PROVINCES + [None] # None = national

RENAME_MAP = {
    "Sindh": "sindh",
    "Islamabad": "ict",
    "Punjab": "punjab",
    "AJK": "ajk",
    "Khyber Pakhtunkhwa": "kp",
    "Gilgit Baltistan": "gb",
    "Balochistan": "balochistan",
}


# -------------------------------------------------------------------------
# Data preparation
# -------------------------------------------------------------------------

def clean_linelist(ll):
    """
    One-time cleanup applied to the raw linelist before any filtering.
    Standardises province names, parses dates, and computes case weights.
    """
    ll = ll.copy()
    ll["province"] = ll["province"].replace(RENAME_MAP).str.lower()
    ll["time"] = pd.to_datetime(ll["time"], errors="coerce")
    ll = ll.dropna(subset=["time"])

    # Expected-value weighting: confirmed = 1, compatible = conf_prob
    if "is_case" in ll.columns:
        ll["case_weight"] = ll["is_case"]
        m = ll["case_weight"].isna()
        ll.loc[m, "case_weight"] = ll.loc[m, "conf_prob"].fillna(0.0)
    else:
        ll["case_weight"] = ll["conf_prob"].fillna(0.0)

    ll = ll.loc[ll["case_weight"] > 0.0]
    return ll.reset_index(drop=True)


def build_weighted_cases(ll, province):
    """
    Filter the cleaned linelist to a province (or national if None),
    compute birth year and age at infection, and restrict to the
    cohort window.
    """
    df = ll if province is None else ll.loc[ll["province"] == province]
    df = df.copy()

    df["year"] = df["time"].dt.year
    if "age_months" in df.columns and "age_years" not in df.columns:
        df["age_years"] = df["age_months"] / 12.0
    df = df.dropna(subset=["age_years"])

    df["age_at_inf"] = np.clip(np.floor(df["age_years"]).astype(int), 0, AGE_MAX)
    df["birth_year"] = df["year"] - df["age_at_inf"]
    df = df.loc[
        (df["birth_year"] >= COHORT_START) & (df["birth_year"] <= COHORT_END)
    ]
    return df.reset_index(drop=True)


def build_count_table(df):
    """
    Weighted count table: rows = birth cohort, columns = age at infection.
    Reindexed to the full cohort × age grid.
    """
    counts = (
        df.groupby(["birth_year", "age_at_inf"])["case_weight"]
        .sum()
        .unstack("age_at_inf", fill_value=0.0)
    )
    full_index = np.arange(COHORT_START, COHORT_END + 1, dtype=int)
    full_columns = np.arange(0, AGE_MAX + 1, dtype=int)
    return counts.reindex(index=full_index, columns=full_columns, fill_value=0.0)


def build_obs_mask(cohort_index, age_columns, y_min, y_max):
    """
    Binary mask: 1 if birth_year + age falls within the surveillance
    window [y_min, y_max], else 0.
    """
    years = cohort_index.values[:, np.newaxis] + age_columns.values[np.newaxis, :]
    m = ((years >= y_min) & (years <= y_max)).astype(int)
    return pd.DataFrame(m, index=cohort_index, columns=age_columns)


# -------------------------------------------------------------------------
# Fitting
# -------------------------------------------------------------------------

def fit_all_regions(ll, mask):
    """
    Fit the masked_buckets model for each region. Returns a dict
    mapping region tag -> dict with keys: mid, var, low, high, counts, frac.
    """
    national_index = mask.index
    results = {}

    for province in REGIONS:
        tag = "national" if province is None else province
        print(f"\n  Fitting {tag}...")

        df = build_weighted_cases(ll, province)
        counts = build_count_table(df)
        this_mask = mask.loc[counts.index].values

        total_by_year = counts.sum(axis=1)
        frac = counts.div(total_by_year, axis=0).fillna(0.0)

        buckets = mb.BinomialPosterior(
            counts,
            correlation_time=CORRELATION_TIME,
            g2g_correlation=G2G_CORRELATION,
            mask=this_mask,
        )
        result = mb.FitModel(buckets)
        samples = mb.SampleBuckets(result, buckets)

        print(f"    success={result.success}, "
              f"{total_by_year.sum():.0f} weighted cases")

        mid = samples.mean(axis=0)
        var = samples.var(axis=0)
        low = np.percentile(samples, 2.5, axis=0)
        high = np.percentile(samples, 97.5, axis=0)

        # Build output DataFrames with forward-filled alignment
        mid_df = (
            pd.DataFrame(mid, columns=frac.columns, index=frac.index)
            .reindex(national_index)
            .fillna(method="ffill")
            .stack()
            .rename("avg")
        )
        var_df = (
            pd.DataFrame(var, columns=frac.columns, index=frac.index)
            .reindex(national_index)
            .fillna(method="ffill")
            .stack()
            .rename("var")
        )

        results[tag] = {
            "mid": mid,
            "var": var,
            "low": low,
            "high": high,
            "counts": counts,
            "frac": frac,
            "output": pd.concat([mid_df, var_df], axis=1),
        }

    return results


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

def axes_setup(ax):
    ax.spines["left"].set_position(("axes", -0.025))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def plot_age_panels(results, out_pdf):
    """
    One page per region: 7×4 grid of panels, one per age bin,
    showing observed fractions vs. smoothed model with CIs.
    """
    with PdfPages(out_pdf) as book:
        for tag, res in results.items():
            frac = res["frac"]
            mid, low, high = res["mid"], res["low"], res["high"]

            fig, axes = plt.subplots(
                7, 4, sharex=True, sharey=False, figsize=(15, 14),
            )
            axes = axes.reshape(-1)

            for i, ax in enumerate(axes):
                if i in {2, 3}:
                    ax.axis("off")
                    continue
                axes_setup(ax)
                ax.grid(color="grey", alpha=0.2)

            for i in frac.columns:
                j = i + 2 if i > 1 else i

                axes[j].fill_between(
                    frac.index, low[:, i], high[:, i],
                    facecolor="grey", edgecolor="None", alpha=0.4,
                )
                axes[j].plot(frac.index, mid[:, i], lw=2, color="grey")
                axes[j].plot(
                    frac[i], ls="None", lw=1, color="k", markersize=8,
                    marker="o", markeredgecolor="k", markerfacecolor="None",
                    markeredgewidth=2,
                )
                axes[j].text(
                    0.01, 0.99, f"{i} years old",
                    ha="left", va="top", fontsize=22, color="k",
                    transform=axes[j].transAxes,
                )
                if j % 4 == 0:
                    axes[j].set_ylabel("Probability")
                if j >= 8:
                    ticks = np.arange(2005, 2027, 5)
                    axes[j].set_xticks(ticks)
                    axes[j].set_xticklabels(
                        [f"'{y % 100:02d}" for y in ticks], rotation=45, ha="right",
                    )

            # Legend in the spare panel
            axes[2].plot(
                [], ls="None", color="k", markersize=8, marker="o",
                markeredgecolor="k", markerfacecolor="None",
                markeredgewidth=2, label="Observed fraction",
            )
            axes[2].plot([], lw=2, color="grey", label="Smoothed model")
            axes[2].legend(loc="center", frameon=False)

            fig.suptitle(f"Infection age by birth cohort: {tag.title()}")
            fig.tight_layout(rect=[0, 0.0, 1, 0.9])
            book.savefig(fig)
            plt.close(fig)

        d = book.infodict()
        d["Title"] = "Cohort age-at-infection priors"
        d["Author"] = "Marisa"


def plot_cohort_grids(results, mask, out_pdf):
    """
    One page per region: grid of small bar charts (one per cohort),
    comparing observed fractions with smoothed priors.
    """
    cohorts = np.arange(COHORT_START, COHORT_END + 1, dtype=int)
    ages = np.arange(0, AGE_MAX + 1, dtype=int)
    w = 0.30

    with PdfPages(out_pdf) as book:
        for tag, res in results.items():
            mid_unstacked = res["output"]["avg"].unstack()
            counts = res["counts"]
            total_by_cohort = counts.sum(axis=1)
            obs_frac = counts.div(total_by_cohort, axis=0).fillna(0.0)

            ncol = 4
            nrow = int(np.ceil(len(cohorts) / ncol))
            fig, axes = plt.subplots(
                nrow, ncol, figsize=(4 * ncol, 2.8 * nrow),
                sharex=True, sharey=True,
            )
            axes = np.array(axes).reshape(-1)

            # Compute shared y-axis limit
            ymax = 0.0
            for b in cohorts:
                if b in mid_unstacked.index:
                    ymax = max(ymax, mid_unstacked.loc[b].max())
                if b in obs_frac.index:
                    m = mask.loc[b].values.astype(bool) if b in mask.index else np.ones(AGE_MAX + 1, dtype=bool)
                    vals = obs_frac.loc[b].values.copy()
                    vals[~m] = 0.0
                    ymax = max(ymax, vals.max())
            ymax *= 1.15

            for i, b in enumerate(cohorts):
                ax = axes[i]
                axes_setup(ax)

                m = mask.loc[b].values.astype(bool) if b in mask.index else np.ones(AGE_MAX + 1, dtype=bool)
                n_b = int(total_by_cohort.get(b, 0))

                # Smoothed prior
                if b in mid_unstacked.index:
                    ax.bar(
                        ages + w / 2, mid_unstacked.loc[b].values,
                        width=w, alpha=0.85, color="C2",
                        label="Smoothed prior" if i == 0 else None,
                    )

                # Observed (observable ages only)
                if b in obs_frac.index:
                    obs = obs_frac.loc[b].values.copy()
                    obs[~m] = np.nan
                    ax.bar(
                        ages - w / 2, obs, width=w, alpha=0.55, color="C1",
                        label="Observed" if i == 0 else None,
                    )

                # Grey shading for unobservable ages
                if m.any() and not m.all():
                    first_obs = ages[m][0]
                    last_obs = ages[m][-1]
                    if first_obs > 0:
                        ax.axvspan(-0.5, first_obs - 0.5, alpha=0.08, color="grey")
                    if last_obs < AGE_MAX:
                        ax.axvspan(last_obs + 0.5, AGE_MAX + 0.5, alpha=0.08, color="grey")

                ax.set_title(f"{b} (n={n_b})")
                ax.set_xlim(-0.5, AGE_MAX + 0.5)
                ax.set_ylim(0, max(ymax, 0.01))
                ax.grid(alpha=0.2)

            # Hide unused axes
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            # Collect legend handles
            handles, labels = [], []
            for ax in axes[: i + 1]:
                for h, l in zip(*ax.get_legend_handles_labels()):
                    if l not in labels:
                        handles.append(h)
                        labels.append(l)
            fig.legend(handles, labels, loc="upper right", frameon=False)

            fig.suptitle(
                f"{tag.title()}: Cohort age-at-infection priors "
                f"({COHORT_START}\u2013{COHORT_END})",
                fontsize=14,
            )
            fig.tight_layout(rect=[0.04, 0.05, 0.98, 0.94])
            book.savefig(fig)
            plt.close(fig)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":

    # Load and clean
    ll = pd.read_pickle(LINELIST_PKL).reset_index(drop=True)
    ll = clean_linelist(ll)

    # Build observation mask
    y_min = ll["time"].dt.year.min()
    y_max = ll["time"].dt.year.max()
    full_index = pd.Index(np.arange(COHORT_START, COHORT_END + 1, dtype=int))
    full_columns = pd.Index(np.arange(0, AGE_MAX + 1, dtype=int))
    mask = build_obs_mask(full_index, full_columns, y_min, y_max)

    # Fit all regions
    results = fit_all_regions(ll, mask)

    # Plot
    plot_age_panels(results, OUT_PDF_PANELS)
    plot_cohort_grids(results, mask, OUT_PDF_GRIDS)

    # Save
    output = pd.concat(
        {tag: res["output"] for tag, res in results.items()},
    )
    output.index.set_names(["province", "birth_year", "age"], inplace=True)
    print("\nFinal output:")
    print(output)
    output.to_pickle(OUT_PKL)
    print(f"\nSaved: {OUT_PKL}")