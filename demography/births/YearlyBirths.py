""" YearlyBirths.py

Uses outputs from AgeAtKthKid.py, ZeroInflatedNumKids.py, and
MomDistribution.py to estimate birth rate per year by province.

Core idea:
If U is a uniformly chosen person in a given (state, year),
set B=1 if U is a baby born in that year, and B=0 otherwise.

We estimate P(B=1 | state, year) via:
P(B=1) = sum_cells P(B=1 | mother's cell=c) * P(mother's cell=c)

Where:
- P(mother's cell=c) comes from DHS weights in mom_distribution.pkl
- P(B=1 | mother's cell=c) comes from age-at-k and num births models

National aggregation:
  - For WB comparison: main provinces only (DHS weights)
  - For final output: main provinces + AJK/GB, using DHS-based popshares
    for main provinces and census-based popshares for AJK/GB

Dependencies:
    - extrapolate_trends
    - World Bank birth rate CSV in _data folder
    - ../../pickle_jar/mom_distribution_main.pkl
    - ../../pickle_jar/mom_distribution_ajk_gb.pkl
    - ../../pickle_jar/age_at_kth_kid_by_province.pkl
    - ../../pickle_jar/num_brs_by_province.pkl

Outputs:
    - ../../pickle_jar/birth_rates_by_province.pkl
    - ../../_plots/br_compare.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import nbinom

from matplotlib.ticker import MaxNLocator, FuncFormatter
from demography.extrapolate_trends import (
    load_wb_cbr,
    extend_with_ref_window_ratios,
    compute_scale_factors, apply_year_scale,
    compute_ratio_extrapolation_variance,
    compute_subregion_popshares, extend_popshares,
    aggregate_to_national,
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

COUNTRY = "pakistan"
RESCALE_MAIN_TO_WB = True
EXTRAPOLATE_TO = 2025
EXTRAPOLATE_BACK_TO = 1987
PLOT_START = 2004
EXTRAP_WINDOW = 1

EXCLUDE_FROM_NAT = {"ajk", "gb"}

MIN_RATIO_VAR = 0.001
EXTRAP_GROWTH_RATE = 0.05

# Data file paths
DATA_DIR = "../../_data/"
WB_BIRTHS_FILE = DATA_DIR + "Births.csv"

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def axes_setup(ax):
    ax.spines["left"].set_position(("axes", -0.025))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def format_year_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}"))

def weighted_mean_var(df):
    df = df.dropna(subset=["weight", "pr_birth_last_year"]).copy()
    df = df.loc[df["weight"] > 0].copy()
    wsum = df["weight"].sum()
    if wsum <= 0:
        return pd.Series([np.nan, np.nan, np.nan], index=["avg", "var", "count_var"])

    avg = (df["weight"] * df["pr_birth_last_year"]).sum() / wsum
    var = (df["weight"] * df["pr_birth_last_year"] * (1.0 - df["pr_birth_last_year"])).sum() / wsum
    count_var = (df["weight"] * df["pr_birth_last_year"] * (1.0 - df["pr_birth_last_year"])).sum()

    return pd.Series([avg, var, count_var], index=["avg", "var", "count_var"])

def lognormal_pdf(x, mu_ln, sigma2_ln):
    x = np.asarray(x, dtype=float)
    sigma_ln = np.sqrt(max(float(sigma2_ln), 1e-12))
    x_safe = np.maximum(x, 1e-6)
    coef = 1.0 / (x_safe * sigma_ln * np.sqrt(2.0 * np.pi))
    z = (np.log(x_safe) - float(mu_ln)) / sigma_ln
    return coef * np.exp(-0.5 * z * z)

def nb_pmf_from_mu_alpha(k, mu, alpha):
    mu = float(max(mu, 1e-12))
    alpha = float(max(alpha, 1e-12))
    r = 1.0 / alpha
    p = r / (r + mu)
    return nbinom.pmf(k, r, p)

def choose_k_support(mu, alpha, k_cap=60, k_min=15):
    mu = float(mu); alpha = float(alpha)
    var = mu + alpha * mu * mu
    sd = np.sqrt(max(var, 1e-12))
    k_max = int(np.ceil(mu + 6.0 * sd)) + 5
    k_max = max(k_min, min(k_cap, k_max))
    return np.arange(0, k_max + 1)

def compute_pr_birth_last_year(dist, age_k_idx, numkids_idx, use_hurdle_lookup):
    """
    Compute pr_birth_last_year per demographic cell in dist.
    
    For each cell, checks whether the (province, mom_age) combination
    uses a hurdle model or plain NB, and applies the appropriate PMF.
    """
    dist = dist.copy()

    ages = np.arange(1, 12 * 100 + 1) / 12.0
    pr = np.zeros(len(dist), dtype=float)

    for i, r in dist.iterrows():
        province = str(r["province"])
        mom_age_bin = str(r["mom_age"])
        area = str(r["area"])
        edu = str(r["mom_edu"])
        year = int(r["year"])

        # Look up parameters
        try:
            row_nk = numkids_idx.loc[(province, area, edu, mom_age_bin, year)]
        except KeyError:
            pr[i] = 0.0
            continue

        mu = float(row_nk["mean_num_brs_pos"])
        alpha = float(row_nk["alpha_prov_age"])
        pi0 = float(row_nk["pi_zero"])

        # Check if this (province, age_bin) uses hurdle
        is_hurdle = use_hurdle_lookup.get((province, mom_age_bin), False)

        # Compute PMF over k
        k_vals = choose_k_support(mu, alpha)

        if is_hurdle:
            # Hurdle: P(k>0) = (1-pi0) * NB(k; mu_pos, alpha)
            nb_k = nb_pmf_from_mu_alpha(k_vals, mu=mu, alpha=alpha)
            probs_k = np.zeros_like(nb_k)
            probs_k[k_vals == 0] = pi0
            probs_k[k_vals > 0] = (1.0 - pi0) * nb_k[k_vals > 0]
        else:
            # Plain NB: P(k) = NB(k; mu, alpha)
            probs_k = nb_pmf_from_mu_alpha(k_vals, mu=mu, alpha=alpha)

        # Only k >= 1 contributes to births
        mask = (k_vals >= 1) & (probs_k > 0)
        k_vals = k_vals[mask]
        probs_k = probs_k[mask]

        # For each k, get the log-normal age-at-kth-birth distribution
        age_dists = []
        for k in k_vals:
            try:
                row = age_k_idx.loc[(province, area, edu, year, int(k))]
                mu_ln = float(row["mean_ln_mom_age"])
                sigma2 = float(row["var_ln_mom_age"])
            except KeyError:
                age_dists.append(np.zeros_like(ages))
                continue

            pdf = lognormal_pdf(ages, mu_ln, sigma2)
            s = pdf.sum()
            pdf = pdf / s if s > 0 else pdf
            age_dists.append(pdf)

        age_dists = np.vstack(age_dists)
        mixture = age_dists * probs_k[:, None]

        # Integrate over the mom's age bin
        L, H = map(int, mom_age_bin.split("-"))
        start = max((L - 1) * 12, 0)
        end = max((H - 1) * 12, start)

        total_pr = mixture[:, start:end].sum()
        total_pr *= (1.0 / 5.0)
        pr[i] = float(total_pr)

    dist["pr_birth_last_year"] = pr
    dist["year"] = dist["year"].astype(int) - 1
    return dist

def aggregate_birthrate(dist_with_pr, group_cols, rate_per=1000.0):
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    cols = list(group_cols) + ["weight", "pr_birth_last_year"]
    br = dist_with_pr[cols].copy()

    out = (
        br.groupby(group_cols, sort=True)
          .apply(weighted_mean_var)
          .reset_index()
          .rename(columns={"avg": "br_est_prob", "var": "br_var_prob"})
    )

    out["br_est"] = rate_per * out["br_est_prob"]
    out["br_var"] = rate_per * out["br_var_prob"]
    out["br_std"] = np.sqrt(out["br_var"].clip(lower=0))

    out = out.drop(columns=["br_est_prob", "br_var_prob"])
    return out


# --------------------------------------------------------------------------
# Plotting helper
# --------------------------------------------------------------------------

def plot_birth_rate_series(df_series, wb_df, title, year_col="year",
                           wb_col="wb_br", est_col="br_est", std_col="br_std",
                           rescaled_col="br_final", 
                           show_rescaled=RESCALE_MAIN_TO_WB,
                           rescaled_label="Rescaled", est_label="Estimate", 
                           wb_label="World Bank",
                           savepath=None, show=True):
    dfp = df_series.dropna(subset=[est_col]).copy()
    if dfp.empty:
        return None

    y0, y1 = int(dfp[year_col].min()), int(dfp[year_col].max())
    wb_s = wb_df.loc[(wb_df[year_col] >= y0) & (wb_df[year_col] <= y1)].copy()

    fig, ax = plt.subplots(figsize=(11, 5))
    axes_setup(ax)
    ax.grid(alpha=0.3)
    ax.plot(wb_s[year_col], wb_s[wb_col], lw=2, ls="dashed", marker="o", 
            color="k", label=wb_label, markersize=8)
    if std_col in dfp.columns:
        ax.errorbar(dfp[year_col], dfp[est_col], yerr=dfp[std_col], lw=1, 
            fmt="none", color="tab:orange")
    ax.plot(dfp[year_col], dfp[est_col], lw=2, ls="dashed", marker="o", 
            color="tab:orange", label=est_label, markersize=8)
    if show_rescaled and (rescaled_col in dfp.columns):
        ax.plot(dfp[year_col], dfp[rescaled_col], lw=2, ls="solid", marker="s", 
                color="tab:red", label=rescaled_label, markersize=6)

    ax.set_ylabel("Crude birth rate (per 1000)")
    ax.set_xlabel("Year")
    format_year_axis(ax)
    ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == "__main__":

    # Load inputs
    main = pd.read_pickle("../../pickle_jar/mom_distribution_main.pkl")
    ajk_gb = pd.read_pickle("../../pickle_jar/mom_distribution_ajk_gb.pkl")

    age_k = pd.read_pickle("../../pickle_jar/age_at_kth_kid_by_province.pkl")
    age_k_idx = age_k.set_index(["province", "area", "mom_edu", "year", "bord"]).sort_index()

    numkids = pd.read_pickle("../../pickle_jar/num_brs_by_province.pkl")
    numkids_idx = numkids.set_index(["province", "area", "mom_edu", "mom_age", "year"]).sort_index()

    # Build use_hurdle lookup from numkids
    hurdle_info = numkids[["province", "mom_age", "use_hurdle"]].drop_duplicates()
    use_hurdle_lookup = {
        (str(row["province"]), str(row["mom_age"])): bool(row["use_hurdle"])
        for _, row in hurdle_info.iterrows()
    }

    # Cell-level pr_birth_last_year
    main2 = compute_pr_birth_last_year(main, age_k_idx, numkids_idx, use_hurdle_lookup)    
    ajk2 = compute_pr_birth_last_year(ajk_gb, age_k_idx, numkids_idx, use_hurdle_lookup)

    # Aggregate to province-year
    prov_main = aggregate_birthrate(main2, ["province", "year"])
    prov_main["group"] = "main"

    prov_ajk = aggregate_birthrate(ajk2, ["province", "year"])
    prov_ajk["group"] = "ajk_gb"

    # -------------------------------------------------------
    # Identify observed (province, year) combos for extrapolation uncertainty
    # -------------------------------------------------------
    prov_all_raw = pd.concat([prov_main, prov_ajk], ignore_index=True)
    obs_by_prov_year = (
        prov_all_raw.dropna(subset=["br_est"])
        .loc[prov_all_raw["br_est"].abs() > 1e-12]
        .groupby("province")["year"]
        .apply(lambda s: set(s.astype(int)))
        .to_dict()
    )

    # Load WB
    wb = load_wb_cbr(WB_BIRTHS_FILE, EXTRAPOLATE_BACK_TO, EXTRAPOLATE_TO, [COUNTRY])
    wb["year"] = wb["year"].astype(int)

    # Build WB reference DataFrame for extend_with_ref_window_ratios
    wb_ref = wb[["year", "wb_br"]].copy()

    # -------------------------------------------------------
    # Extend provinces outside observed range using WB trend
    # -------------------------------------------------------
    prov_main_ext = extend_with_ref_window_ratios(
        prov_main.copy(), wb_ref,
        start_year=EXTRAPOLATE_BACK_TO, end_year=EXTRAPOLATE_TO,
        window=EXTRAP_WINDOW, group_col="province",
        value_col="br_est", var_col="br_var", ref_col="wb_br",
    )
    prov_main_ext["group"] = "main"

    prov_ajk_ext = extend_with_ref_window_ratios(
        prov_ajk.copy(), wb_ref,
        start_year=EXTRAPOLATE_BACK_TO, end_year=EXTRAPOLATE_TO,
        window=EXTRAP_WINDOW, group_col="province",
        value_col="br_est", var_col="br_var", ref_col="wb_br",
    )
    prov_ajk_ext["group"] = "ajk_gb"

    # -------------------------------------------------------
    # Ratio-based extrapolation uncertainty
    # -------------------------------------------------------
    wb_ref_series = wb.set_index("year")["wb_br"]

    prov_main_ext = compute_ratio_extrapolation_variance(
        prov_main_ext, wb_ref_series, obs_by_prov_year,
        group_col="province", cov_col="br_est", var_col="br_var",
        window=EXTRAP_WINDOW, min_ratio_var=MIN_RATIO_VAR,
        growth_rate=EXTRAP_GROWTH_RATE,
    )
    prov_main_ext["br_std"] = np.sqrt(prov_main_ext["br_var"].clip(lower=0))

    prov_ajk_ext = compute_ratio_extrapolation_variance(
        prov_ajk_ext, wb_ref_series, obs_by_prov_year,
        group_col="province", cov_col="br_est", var_col="br_var",
        window=EXTRAP_WINDOW, min_ratio_var=MIN_RATIO_VAR,
        growth_rate=EXTRAP_GROWTH_RATE,
    )
    prov_ajk_ext["br_std"] = np.sqrt(prov_ajk_ext["br_var"].clip(lower=0))

    # -------------------------------------------------------
    # Rescale main provinces to WB (optional)
    # -------------------------------------------------------
    # Aggregate extrapolated main provinces for WB comparison
    dhs_popshares = compute_subregion_popshares(main, group_col="province", year_col="year")
    dhs_popshares = extend_popshares(
        dhs_popshares, year_min=EXTRAPOLATE_BACK_TO, year_max=EXTRAPOLATE_TO,
    )

    nat_main_agg = aggregate_to_national(
        prov_main_ext, dhs_popshares,
        group_col="province", year_col="year",
        cov_col="br_est", var_col="br_var",
    )

    if RESCALE_MAIN_TO_WB:
        nat_for_scale = nat_main_agg.set_index("year")["nat_est"]
        scale = compute_scale_factors(wb_ref_series, nat_for_scale)

        prov_main_ext = apply_year_scale(
            prov_main_ext, scale, year_col="year",
            mean_col="br_est", var_col="br_var",
            out_mean_col="br_final", out_var_col="br_var_final",
        )
        prov_main_ext["br_std_final"] = np.sqrt(prov_main_ext["br_var_final"].clip(lower=0))
        if "count_var" in prov_main_ext.columns:
            prov_main_ext["count_var"] = prov_main_ext["count_var"] * (
                prov_main_ext["year"].map(scale).fillna(1.0) ** 2
            )
    else:
        prov_main_ext["scale_factor"] = 1.0
        prov_main_ext["br_final"] = prov_main_ext["br_est"]
        prov_main_ext["br_std_final"] = prov_main_ext["br_std"]
        if "count_var" in prov_main_ext.columns:
            mask = prov_main_ext["count_var"].isna() & prov_main_ext["br_est"].notna()
            p_approx = prov_main_ext.loc[mask, "br_est"] / 1000.0
            prov_main_ext.loc[mask, "count_var"] = p_approx * (1.0 - p_approx)

    # AJK/GB: never rescaled
    prov_ajk_ext["scale_factor"] = np.nan
    prov_ajk_ext["br_final"] = prov_ajk_ext["br_est"]
    prov_ajk_ext["br_std_final"] = prov_ajk_ext["br_std"]
    if "count_var" in prov_ajk_ext.columns:
        mask = prov_ajk_ext["count_var"].isna() & prov_ajk_ext["br_est"].notna()
        p_approx = prov_ajk_ext.loc[mask, "br_est"] / 1000.0
        prov_ajk_ext.loc[mask, "count_var"] = p_approx * (1.0 - p_approx)

    # -------------------------------------------------------
    # Combine province outputs and save
    # -------------------------------------------------------
    prov_out = pd.concat([prov_main_ext, prov_ajk_ext], ignore_index=True)
    prov_out = prov_out.sort_values(["group", "province", "year"]).reset_index(drop=True)
    prov_out.to_pickle("../../pickle_jar/birth_rates_by_province.pkl")
    print(f"RESCALE_MAIN_TO_WB = {RESCALE_MAIN_TO_WB}")
    print(prov_out.head())


    # -------------------------------------------------------
    # Plot: national (main provinces aggregate) vs WB
    # -------------------------------------------------------
    nat_main_agg = nat_main_agg.merge(wb, on="year", how="left")
    df_n = nat_main_agg.dropna(subset=["nat_est"]).copy()
    df_n = df_n.loc[df_n["year"] >= PLOT_START]
    plot_birth_rate_series(df_n, wb, "Pakistan national crude birth rate: DHS-based vs World Bank",
                          est_col="nat_est", std_col="nat_std",
                          wb_label="World Bank (Pakistan excl. AJK/GB)",
                          est_label="DHS-based national (province aggregate)",
                          rescaled_label="Rescaled target (WB)",
                          savepath="../../_plots/br_compare.png")

    # Plot: per province
    for prov, df_p in prov_out.groupby("province"):
        df_p = df_p.dropna(subset=["br_est"]).copy()
        df_p = df_p.loc[df_p["year"] >= PLOT_START]
        show_rescaled = (df_p["group"].iloc[0] == "main") and RESCALE_MAIN_TO_WB
        plot_birth_rate_series(df_p, wb, f"Crude birth rate in {prov.title()}",
                              show_rescaled=show_rescaled,
                              wb_label="WB national reference",
                              est_label=f"{prov} raw",
                              rescaled_label=f"{prov} rescaled")