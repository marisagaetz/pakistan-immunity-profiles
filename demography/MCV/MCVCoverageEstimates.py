""" MCVCoverageEstimates.py

Estimate province-level MCV1/MCV2 coverage by birth year using:
  - demographic weights from mom_distribution.pkl
  - predicted P(MCV1) from mcv1_prob_pred.pkl

Pipeline:
  (1) Province-level MCV1 coverage from cell-level predictions + DHS weights
  (2) National aggregate (excl. AJK/GB) for WHO comparison
  (3) Optional rescaling of province estimates to match WHO
  (4) Extend provinces forward/backward using WHO trend, with
     ratio-based extrapolation uncertainty
  (5) MCV2 point estimates via WHO MCV2/MCV1 ratios; MCV2 uncertainty
     set to a fixed coefficient of variation.
  (6) Monthly interpolation for province + national outputs

Dependencies:
  - extrapolate_trends
  - WHO MCV1 and MCV2 coverage estimate xlsx files in _data folder
  - ../../pickle_jar/mom_distribution_main.pkl
  - ../../pickle_jar/mom_distribution_ajk_gb.pkl
  - ../../pickle_jar/mcv1_prob_pred.pkl

Outputs:
  - ../../pickle_jar/mcv_prov_cov_monthly.pkl   (province-level monthly)
  - ../../pickle_jar/mcv_nat_cov_monthly.pkl    (national monthly, all provs incl. AJK/GB)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from demography.extrapolate_trends import (
    load_who_national, clip01,
    extend_with_ref_window_ratios,
    compute_scale_factors, apply_year_scale,
    compute_ratio_extrapolation_variance,
    compute_subregion_popshares, extend_popshares,
    aggregate_to_national,
    interpolate_yearly_to_monthly,
)

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
COUNTRY = "pakistan"

MAX_YEAR_DEFAULT = 2016
MAX_YEAR_AJK = 2017

EXTRAPOLATE_TO = 2025
EXTRAPOLATE_BACK_TO = 1987
EXTRAP_K = 3

EXCLUDE_FROM_NAT = {"ajk", "gb"}

# Cell-level variance column from MCVOneProbability
CELL_VAR_COL = "var_prob"

# If True, rescale main province estimates so national aggregate
# matches WHO, and save/plot the rescaled series.
RESCALE = False

# Extrapolation uncertainty parameters
MIN_RATIO_VAR = 0.001
EXTRAP_GROWTH_RATE = 0.1

# MCV2 uncertainty: fixed coefficient of variation, reflecting that
# MCV2 is derived from MCV1 via WHO ratio (a strong assumption whose
# uncertainty isn't captured by propagating MCV1 variance).
MCV2_CV = 0.10

PLOT_START = 2004
MCV2_PLOT_START = 2009

# Data file paths
DATA_DIR = "../../_data/"
WHO_MCV1_FILE = DATA_DIR + "mcv1_WHO.xlsx"
WHO_MCV2_FILE = DATA_DIR + "mcv2_WHO.xlsx"

# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------

def axes_setup(ax):
    ax.spines["left"].set_position(("axes", -0.025))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_coverage_series(
    ax, years, est, std, obs_min, obs_max,
    color="tab:blue", label_obs="", label_extrap="",
    marker="o", markersize=8,
):
    """
    Plot a coverage series with full-opacity observed portion and
    faded extrapolated portions, with error bars throughout.
    """
    s = pd.DataFrame({"year": years, "est": est, "std": std}).dropna(subset=["est"])
    if s.empty:
        return

    # Observed (full opacity)
    mask_obs = (s["year"] >= obs_min) & (s["year"] <= obs_max)
    if mask_obs.any():
        ax.errorbar(s.loc[mask_obs, "year"], s.loc[mask_obs, "est"],
                    yerr=1.96 * s.loc[mask_obs, "std"],
                    lw=1, fmt="none", color=color)
        ax.plot(s.loc[mask_obs, "year"], s.loc[mask_obs, "est"],
                lw=2, ls="dashed", marker=marker, color=color,
                label=label_obs, markersize=markersize)

    # Backward extrapolation (faded, bridged to first observed year)
    mask_back = s["year"] < obs_min
    if mask_back.any():
        bridge = s.loc[s["year"] == obs_min]
        ext = pd.concat([s.loc[mask_back], bridge], ignore_index=True).sort_values("year")
        ax.errorbar(ext["year"], ext["est"], yerr=1.96 * ext["std"],
                    lw=1, fmt="none", color=color, alpha=0.35)
        ax.plot(ext["year"], ext["est"], lw=2, ls="dashed", marker=marker,
                color=color, alpha=0.35, markersize=markersize)

    # Forward extrapolation (faded, bridged to last observed year)
    mask_fwd = s["year"] > obs_max
    if mask_fwd.any():
        bridge = s.loc[s["year"] == obs_max]
        ext = pd.concat([bridge, s.loc[mask_fwd]], ignore_index=True).sort_values("year")
        ax.errorbar(ext["year"], ext["est"], yerr=1.96 * ext["std"],
                    lw=1, fmt="none", color=color, alpha=0.35)
        ax.plot(ext["year"], ext["est"], lw=2, ls="dashed", marker=marker,
                color=color, alpha=0.35, markersize=markersize)

    # Legend entry for extrapolated portion
    if (mask_back.any() or mask_fwd.any()) and label_extrap:
        ax.plot([], [], lw=2, ls="dashed", marker=marker, color=color,
                alpha=0.35, markersize=markersize, label=label_extrap)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # -------------------------------------------------------
    # Load demographic distributions
    # -------------------------------------------------------
    dist = pd.read_pickle("../../pickle_jar/mom_distribution_main.pkl").copy()
    dist["year"] = dist["year"].astype(int)
    dist_ajk_gb = pd.read_pickle("../../pickle_jar/mom_distribution_ajk_gb.pkl").copy()
    dist_ajk_gb["year"] = dist_ajk_gb["year"].astype(int)

    dist_all = pd.concat([dist, dist_ajk_gb], ignore_index=True)
    mask = (
        ((dist_all["year"] <= MAX_YEAR_DEFAULT) |
         ((dist_all["province"] == "ajk") & (dist_all["year"] <= MAX_YEAR_AJK)))
    )
    dist_all = dist_all.loc[mask].copy()

    # -------------------------------------------------------
    # Load MCV probability predictions
    # -------------------------------------------------------
    mcv_df = pd.read_pickle("../../pickle_jar/mcv1_prob_pred.pkl").copy()
    mcv_df["year"] = mcv_df["birth_year"].astype(int)
    mask = (
        ((mcv_df["year"] <= MAX_YEAR_DEFAULT) |
         ((mcv_df["province"] == "ajk") & (mcv_df["year"] <= MAX_YEAR_AJK)))
    )
    mcv_df = mcv_df.loc[mask].copy()
    mcv_df = mcv_df[["province", "area", "mom_edu", "year", "prob_mcv1_pred", CELL_VAR_COL]].copy()

    # -------------------------------------------------------
    # Merge and compute province-level MCV1 coverage
    # -------------------------------------------------------
    df = pd.merge(dist_all, mcv_df, on=["province", "area", "mom_edu", "year"], how="inner")
    df[CELL_VAR_COL] = df[CELL_VAR_COL].fillna(0.0)

    df["weight_prov"] = df.groupby(["province", "year"])["weight"].transform(
        lambda w: w / w.sum() if w.sum() != 0 else np.nan
    )
    df["weighted_mcv_prov"] = df["prob_mcv1_pred"] * df["weight_prov"]
    df["weighted_var_prov"] = df[CELL_VAR_COL] * (df["weight_prov"] ** 2)

    province_cov = (
        df.groupby(["province", "year"], as_index=False)
        .agg(mcv1=("weighted_mcv_prov", "sum"), mcv1_var=("weighted_var_prov", "sum"))
    )
    province_cov["mcv1"] = clip01(province_cov["mcv1"].astype(float))
    province_cov["mcv1_std"] = np.sqrt(province_cov["mcv1_var"])

    # -------------------------------------------------------
    # Identify which (province, year) combos have survey data
    # -------------------------------------------------------
    mcv_full = pd.read_pickle("../../pickle_jar/mcv1_prob_pred.pkl").copy()
    mcv_full["year"] = mcv_full["birth_year"].astype(int)

    obs_counts = mcv_full.groupby(["province", "year"])["n"].sum().reset_index()
    obs_counts = obs_counts.loc[obs_counts["n"].notna() & (obs_counts["n"] > 0)]
    obs_by_prov_year = (
        obs_counts.groupby("province")["year"]
        .apply(lambda s: set(s.astype(int)))
        .to_dict()
    )
    print("Observed (province, year) combos:")
    for p in sorted(obs_by_prov_year):
        print(f"  {p}: {sorted(obs_by_prov_year[p])}")

    all_obs_years = sorted(set().union(*obs_by_prov_year.values()))
    nat_obs_min = int(min(all_obs_years))
    nat_obs_max = int(max(all_obs_years))

    # -------------------------------------------------------
    # National aggregate (excl. AJK/GB, for WHO comparison)
    # -------------------------------------------------------
    popshares = compute_subregion_popshares(dist, group_col="province", year_col="year")
    prov_main = province_cov.loc[~province_cov["province"].isin(EXCLUDE_FROM_NAT)].copy()

    nat_est = aggregate_to_national(
        prov_main, popshares, group_col="province", year_col="year",
        cov_col="mcv1", var_col="mcv1_var", popshare_col="popshare",
    )

    # -------------------------------------------------------
    # WHO national series
    # -------------------------------------------------------
    who_end = EXTRAPOLATE_TO if EXTRAPOLATE_TO is not None else MAX_YEAR_DEFAULT
    who_nat = load_who_national(WHO_MCV1_FILE, year_min=EXTRAPOLATE_BACK_TO, year_max=None, extrapolate_to=who_end)
    who_mcv2 = load_who_national(WHO_MCV2_FILE, year_min=EXTRAPOLATE_BACK_TO, year_max=None, extrapolate_to=who_end)

    # -------------------------------------------------------
    # Rescale (optional)
    # -------------------------------------------------------
    nat_for_scale = nat_est.set_index("year")["nat_est"]
    scale = compute_scale_factors(who_nat, nat_for_scale)

    province_cov_rescaled = province_cov.copy()
    prov_rescaled_main = province_cov_rescaled.loc[
        ~province_cov_rescaled["province"].isin(EXCLUDE_FROM_NAT)
    ].copy()
    prov_rescaled_main = apply_year_scale(
        prov_rescaled_main, scale, year_col="year",
        mean_col="mcv1", var_col="mcv1_var", clip=(0, 1),
    )
    prov_rescaled_ajkgb = province_cov_rescaled.loc[
        province_cov_rescaled["province"].isin(EXCLUDE_FROM_NAT)
    ].copy()
    province_cov_rescaled = pd.concat([prov_rescaled_main, prov_rescaled_ajkgb], ignore_index=True)
    province_cov_rescaled["mcv1_std"] = np.sqrt(province_cov_rescaled["mcv1_var"].clip(lower=0))

    start_year = EXTRAPOLATE_BACK_TO
    end_year = int(EXTRAPOLATE_TO) if EXTRAPOLATE_TO is not None else int(who_nat.index.max())

    # -------------------------------------------------------
    # WHO reference for extend_with_ref_window_ratios
    # -------------------------------------------------------
    who_ref_df = who_nat.rename("ref").reset_index().rename(columns={"index": "year"})
    who_ref_df["year"] = who_ref_df["year"].astype(int)

    # -------------------------------------------------------
    # Extend + ratio-based extrapolation uncertainty
    # -------------------------------------------------------

    # Raw (all provinces)
    province_cov_raw_ext = extend_with_ref_window_ratios(
        province_cov.copy(), who_ref_df,
        start_year=start_year, end_year=end_year, window=EXTRAP_K,
        group_col="province", year_col="year",
        value_col="mcv1", var_col="mcv1_var",
        ref_col="ref", clip=(0, 1),
    )
    province_cov_raw_ext = compute_ratio_extrapolation_variance(
        province_cov_raw_ext, who_nat, obs_by_prov_year,
        group_col="province", cov_col="mcv1", var_col="mcv1_var",
        window=EXTRAP_K,
        min_ratio_var=MIN_RATIO_VAR, growth_rate=EXTRAP_GROWTH_RATE,
    )
    province_cov_raw_ext["mcv1_std"] = np.sqrt(province_cov_raw_ext["mcv1_var"].clip(lower=0))

    # Rescaled (main provinces only)
    province_cov_rescaled_main = province_cov_rescaled.loc[
        ~province_cov_rescaled["province"].isin(EXCLUDE_FROM_NAT)
    ].copy()

    province_cov_rescaled_ext = extend_with_ref_window_ratios(
        province_cov_rescaled_main, who_ref_df,
        start_year=start_year, end_year=end_year, window=EXTRAP_K,
        group_col="province", year_col="year",
        value_col="mcv1", var_col="mcv1_var",
        ref_col="ref", clip=(0, 1),
    )
    province_cov_rescaled_ext = compute_ratio_extrapolation_variance(
        province_cov_rescaled_ext, who_nat, obs_by_prov_year,
        group_col="province", cov_col="mcv1", var_col="mcv1_var",
        window=EXTRAP_K,
        min_ratio_var=MIN_RATIO_VAR, growth_rate=EXTRAP_GROWTH_RATE,
    )
    province_cov_rescaled_ext["mcv1_std"] = np.sqrt(province_cov_rescaled_ext["mcv1_var"].clip(lower=0))
    province_cov_rescaled_ext = province_cov_rescaled_ext.sort_values(["province", "year"]).reset_index(drop=True)

    # -------------------------------------------------------
    # Choose which series to save based on RESCALE toggle
    # -------------------------------------------------------
    if RESCALE:
        ajkgb_raw = province_cov_raw_ext.loc[
            province_cov_raw_ext["province"].isin(EXCLUDE_FROM_NAT)
        ].copy()
        prov_yearly = pd.concat(
            [province_cov_rescaled_ext, ajkgb_raw], ignore_index=True
        )
    else:
        prov_yearly = province_cov_raw_ext.copy()

    prov_yearly = prov_yearly.sort_values(["province", "year"]).reset_index(drop=True)

    # -------------------------------------------------------
    # MCV2 via WHO ratios + CV-based uncertainty
    # -------------------------------------------------------
    prov_yearly["mcv1_who"] = prov_yearly["year"].map(who_nat).astype(float)
    prov_yearly["mcv2_who"] = prov_yearly["year"].map(who_mcv2).astype(float)

    ratio = (prov_yearly["mcv1"].astype(float) / prov_yearly["mcv1_who"]).replace([np.inf, -np.inf], np.nan)
    prov_yearly["mcv2"] = clip01(prov_yearly["mcv2_who"] * ratio)

    # Fixed CV for MCV2 uncertainty (ratio assumption is too coarse
    # for propagated MCV1 variance to be meaningful)
    prov_yearly["mcv2_var"] = (MCV2_CV * prov_yearly["mcv2"]) ** 2
    prov_yearly["mcv2_std"] = MCV2_CV * prov_yearly["mcv2"].abs()

    # Drop WHO helper columns
    prov_yearly = prov_yearly.drop(columns=["mcv1_who", "mcv2_who"])

    # -------------------------------------------------------
    # Monthly interpolation (province-level)
    # -------------------------------------------------------
    prov_monthly = interpolate_yearly_to_monthly(
        df_yearly=prov_yearly,
        group_col="province", year_col="year", time_col="time",
        cov_cols=("mcv1", "mcv2"), var_cols=("mcv1_var", "mcv2_var"),
        start_year=int(EXTRAPOLATE_BACK_TO),
        end_year=int(EXTRAPOLATE_TO) if EXTRAPOLATE_TO is not None else None,
        anchor_month=1,
    )
    prov_monthly = prov_monthly.rename(columns={
        "mcv1_var_std": "mcv1_std", "mcv2_var_std": "mcv2_std",
    })
    prov_monthly = prov_monthly[
        ["time", "province", "mcv1", "mcv1_var", "mcv1_std", "mcv2", "mcv2_var", "mcv2_std"]
    ].copy()

    prov_monthly.to_pickle("../../pickle_jar/mcv_prov_cov_monthly.pkl")
    print("Wrote: mcv_prov_cov_monthly.pkl")

    # -------------------------------------------------------
    # National trend from aggregated extrapolated provinces
    # (all provinces including AJK/GB)
    # -------------------------------------------------------
    dhs_popshares = compute_subregion_popshares(dist, group_col="province", year_col="year")
    dhs_popshares = extend_popshares(
        dhs_popshares, year_min=EXTRAPOLATE_BACK_TO, year_max=EXTRAPOLATE_TO,
    )

    # All-provinces aggregate (for output pickle, incl. AJK/GB)
    nat_mcv1 = aggregate_to_national(
        prov_yearly, dhs_popshares,
        group_col="province", year_col="year",
        cov_col="mcv1", var_col="mcv1_var", popshare_col="popshare",
    )
    nat_mcv1 = nat_mcv1.rename(columns={
        "nat_est": "mcv1", "nat_var": "mcv1_var", "nat_std": "mcv1_std",
    })

    nat_mcv2 = aggregate_to_national(
        prov_yearly, dhs_popshares,
        group_col="province", year_col="year",
        cov_col="mcv2", var_col="mcv2_var", popshare_col="popshare",
    )
    nat_mcv2 = nat_mcv2.rename(columns={
        "nat_est": "mcv2", "nat_var": "mcv2_var", "nat_std": "mcv2_std",
    })

    nat_yearly = nat_mcv1.merge(nat_mcv2, on="year", how="outer").sort_values("year")

    # Main-provinces-only aggregate (for WHO comparison plots, excl. AJK/GB)
    prov_yearly_main = prov_yearly.loc[~prov_yearly["province"].isin(EXCLUDE_FROM_NAT)].copy()

    nat_main_mcv1 = aggregate_to_national(
        prov_yearly_main, dhs_popshares,
        group_col="province", year_col="year",
        cov_col="mcv1", var_col="mcv1_var", popshare_col="popshare",
    )
    nat_main_mcv1 = nat_main_mcv1.rename(columns={
        "nat_est": "mcv1", "nat_var": "mcv1_var", "nat_std": "mcv1_std",
    })

    nat_main_mcv2 = aggregate_to_national(
        prov_yearly_main, dhs_popshares,
        group_col="province", year_col="year",
        cov_col="mcv2", var_col="mcv2_var", popshare_col="popshare",
    )
    nat_main_mcv2 = nat_main_mcv2.rename(columns={
        "nat_est": "mcv2", "nat_var": "mcv2_var", "nat_std": "mcv2_std",
    })

    nat_main_yearly = nat_main_mcv1.merge(nat_main_mcv2, on="year", how="outer").sort_values("year")

    # Interpolate all-provinces national to monthly (for output pickle)
    nat_yearly["_group"] = "national"
    nat_monthly = interpolate_yearly_to_monthly(
        df_yearly=nat_yearly,
        group_col="_group", year_col="year", time_col="time",
        cov_cols=("mcv1", "mcv2"), var_cols=("mcv1_var", "mcv2_var"),
        start_year=int(EXTRAPOLATE_BACK_TO),
        end_year=int(EXTRAPOLATE_TO) if EXTRAPOLATE_TO is not None else None,
        anchor_month=1,
    )
    nat_monthly = nat_monthly.rename(columns={
        "mcv1_var_std": "mcv1_std", "mcv2_var_std": "mcv2_std",
    })
    nat_monthly = nat_monthly[
        ["time", "mcv1", "mcv1_var", "mcv1_std", "mcv2", "mcv2_var", "mcv2_std"]
    ].copy()

    nat_monthly.to_pickle("../../pickle_jar/mcv_nat_cov_monthly.pkl")
    print("Wrote: mcv_nat_cov_monthly.pkl")

    # -------------------------------------------------------
    # National comparison plot (MCV1 + MCV2, excl. AJK/GB)
    # -------------------------------------------------------
    plot_end = int(EXTRAPOLATE_TO) if EXTRAPOLATE_TO is not None else int(who_nat.index.max())
    plot_years = pd.DataFrame({"year": np.arange(int(PLOT_START), plot_end + 1, dtype=int)})

    who_df = who_nat.rename("who_mcv1").reset_index().rename(columns={"index": "year"})
    who_df["year"] = who_df["year"].astype(int)
    who_mcv2_df = who_mcv2.rename("who_mcv2").reset_index().rename(columns={"index": "year"})
    who_mcv2_df["year"] = who_mcv2_df["year"].astype(int)

    nat_plot = (
        plot_years
        .merge(nat_main_yearly, on="year", how="left")
        .merge(who_df, on="year", how="left")
        .merge(who_mcv2_df, on="year", how="left")
    )

    # Crop MCV2 data before 2009
    nat_plot_mcv2 = nat_plot.copy()
    nat_plot_mcv2.loc[nat_plot_mcv2["year"] < MCV2_PLOT_START, ["mcv2", "mcv2_std", "who_mcv2"]] = np.nan

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # MCV1 (top)
    axes_setup(ax1)
    ax1.grid(alpha=0.3)
    ax1.plot(nat_plot["year"], nat_plot["who_mcv1"],
             lw=2, ls="dashed", marker="o", color="k",
             label="WHO/UNICEF national", markersize=7)
    plot_coverage_series(
        ax1, nat_plot["year"], nat_plot["mcv1"], nat_plot["mcv1_std"],
        obs_min=nat_obs_min, obs_max=nat_obs_max,
        label_obs="Model (excl. AJK/GB)",
        label_extrap="Model (extrapolated)",
        markersize=7,
    )
    ax1.set_ylabel("MCV1 coverage")
    ax1.set_ylim(0, 1)
    ax1.legend(frameon=False)

    # MCV2 (bottom)
    axes_setup(ax2)
    ax2.grid(alpha=0.3)
    ax2.plot(nat_plot_mcv2["year"], nat_plot_mcv2["who_mcv2"],
             lw=2, ls="dashed", marker="o", color="k",
             label="WHO/UNICEF national", markersize=7)
    plot_coverage_series(
        ax2, nat_plot_mcv2["year"], nat_plot_mcv2["mcv2"], nat_plot_mcv2["mcv2_std"],
        obs_min=nat_obs_min, obs_max=nat_obs_max,
        color="tab:green",
        label_obs="Model (excl. AJK/GB)",
        label_extrap="Model (extrapolated)",
        markersize=7,
    )
    ax2.set_ylabel("MCV2 coverage")
    ax2.set_xlabel("Birth year")
    ax2.set_ylim(0, 1)
    ax2.legend(frameon=False)

    fig.suptitle("National MCV coverage (excl. AJK/GB) vs WHO", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("../../_plots/mcv_national_compare.png", dpi=200)
    plt.show()

    # -------------------------------------------------------
    # Province plots (MCV1 + MCV2 in one figure per province)
    # -------------------------------------------------------
    for prov, df_p in prov_yearly.groupby("province", sort=True):
        df_p = df_p.sort_values("year")
        df_p = df_p.loc[df_p["year"] >= PLOT_START].copy()
        if df_p.empty:
            continue

        prov_obs_years = obs_by_prov_year.get(prov, set())
        prov_obs_min = min(prov_obs_years) if prov_obs_years else int(df_p["year"].min())
        prov_obs_max = max(prov_obs_years) if prov_obs_years else int(df_p["year"].max())

        y0, y1 = int(df_p["year"].min()), int(df_p["year"].max())
        who_s = who_df.loc[(who_df["year"] >= y0) & (who_df["year"] <= y1)]
        who_s2 = who_mcv2_df.loc[(who_mcv2_df["year"] >= y0) & (who_mcv2_df["year"] <= y1)].copy()

        # Crop MCV2 data before 2009
        df_p_mcv2 = df_p.copy()
        df_p_mcv2.loc[df_p_mcv2["year"] < MCV2_PLOT_START, ["mcv2", "mcv2_std"]] = np.nan
        who_s2.loc[who_s2["year"] < MCV2_PLOT_START, "who_mcv2"] = np.nan

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

        # MCV1 (top)
        axes_setup(ax1)
        ax1.grid(alpha=0.3)
        ax1.plot(who_s["year"], who_s["who_mcv1"],
                 lw=2, ls="dashed", marker="o", color="k",
                 label="WHO national reference", markersize=7)
        plot_coverage_series(
            ax1, df_p["year"], df_p["mcv1"], df_p["mcv1_std"],
            obs_min=prov_obs_min, obs_max=prov_obs_max,
            label_obs=f"{prov} (DHS/MICS-based estimate)",
            label_extrap=f"{prov} (extrapolated via WHO trend)",
            markersize=7,
        )

        # Rescaled overlay
        is_main = prov not in EXCLUDE_FROM_NAT
        if RESCALE and is_main:
            df_p_res = province_cov_rescaled_ext.loc[
                (province_cov_rescaled_ext["province"] == prov) &
                (province_cov_rescaled_ext["year"] >= PLOT_START)
            ].sort_values("year")
            if not df_p_res.empty:
                plot_coverage_series(
                    ax1, df_p_res["year"], df_p_res["mcv1"], df_p_res["mcv1_std"],
                    obs_min=prov_obs_min, obs_max=prov_obs_max,
                    color="tab:green", marker="s", markersize=6,
                    label_obs=f"{prov} rescaled",
                )

        ax1.set_ylabel("MCV1 coverage")
        ax1.set_ylim(0, 1)
        ax1.legend(frameon=False)

        # MCV2 (bottom)
        axes_setup(ax2)
        ax2.grid(alpha=0.3)
        ax2.plot(who_s2["year"], who_s2["who_mcv2"],
                 lw=2, ls="dashed", marker="o", color="k",
                 label="WHO national reference", markersize=7)
        plot_coverage_series(
            ax2, df_p_mcv2["year"], df_p_mcv2["mcv2"], df_p_mcv2["mcv2_std"],
            obs_min=prov_obs_min, obs_max=prov_obs_max,
            color="tab:green",
            label_obs=f"{prov} MCV2 estimate",
            label_extrap=f"{prov} MCV2 (extrapolated)",
            markersize=7,
        )
        ax2.set_ylabel("MCV2 coverage")
        ax2.set_xlabel("Birth year")
        ax2.set_ylim(0, 1)
        ax2.legend(frameon=False)

        fig.suptitle(prov.title(), fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # -------------------------------------------------------
    # Print config summary
    # -------------------------------------------------------
    print(f"\nConfig: RESCALE={RESCALE}, MCV2_CV={MCV2_CV}")
    print(f"Saved series uses: {'rescaled' if RESCALE else 'raw'} province estimates")
    print(f"National pickle contains model-implied aggregate (all provinces incl. AJK/GB)")