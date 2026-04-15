""" SurvivalPrior.py

This script:
  - Loads inputs (CompileDatasets outputs, age-at-infection distributions)
  - For each region, calls a reusable core function

Core logic now lives in methods/survival_prior_core.py

Dependencies:
  - survival_prior_core
  - pickle_jar/semimonthly_prov_dataset.pkl
  - pickle_jar/national_dataset.pkl
  - pickle_jar/sia_cal.pkl
  - pickle_jar/cohort_age_priors.pkl

Outputs:
  - _plots/immunity_profiles_by_province.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages

from survival_prior_core import (
    SurvivalPriorParams,
    run_region_survival_prior,
    get_epi_semimonth_and_births,
    build_precomputed_timeline,
)

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

PROFILE_START_PLOT = 2004          # only for plotting immunity profiles
MODEL_START_YEAR   = 2012          # reporting rate model start
MODEL_END_YEAR     = 2025          # reporting rate model end

PROVINCES = ["punjab", "sindh", "kp", "balochistan", "ict", "ajk", "gb"]
REGIONS   = PROVINCES + [None]  # None -> national

DATA_CUTOFF = pd.Timestamp("2025-09-15")

# Efficacies
MCV1_EFFIC = 0.85
MCV2_EFFIC = 0.95
SIA_VAX_EFFIC = 0.85

# Inputs
PROV_PKL = "pickle_jar/semimonthly_prov_dataset.pkl"
NAT_PKL  = "pickle_jar/national_dataset.pkl"
SIA_CAL_PKL = "pickle_jar/sia_cal.pkl"
AGE_AT_INF_PKL = "pickle_jar/cohort_age_priors.pkl"

# Outputs
OUT_PDF = "_plots/immunity_profiles_by_province.pdf"

# ---------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------

colors = ["#FF420E", "#00ff07", "#0078ff", "#BF00BA"]

def axes_setup(axes, inplace=True):
    axes.spines["left"].set_position(("axes", -0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    return None if inplace else axes

def model_overview(
    province,
    imm_profile,
    sia_cal,
    prA_given_I,
    sf,
    result,
    cov,
    phi,
    r_ls,
    PROFILE_START_PLOT=2004,
    title_prefix="Coarse regression in",
):
    """
    Overview plot like Appendix 2, Fig 1 in the Nigeria paper.
    """ 
    sf = sf.copy()
    sf.index = sf.index.astype(int)
 
    if isinstance(phi, (pd.Series, pd.DataFrame)):
        phi_aligned = pd.Series(phi.squeeze(), index=getattr(phi, "index", sf.index)).reindex(sf.index)
    else:
        phi_aligned = pd.Series(np.asarray(phi).reshape(-1), index=sf.index)
 
    phi_aligned = pd.to_numeric(phi_aligned, errors="coerce").fillna(0.0)
 
    max_birth_for_plot = int(sf.index.max())
    new_index = np.arange(int(PROFILE_START_PLOT), max_birth_for_plot + 1, dtype=int)
 
    imm_aligned = imm_profile.reindex(new_index).copy()
    imm_aligned = imm_aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
 
    sia_names_sorted = []
    if sia_cal is not None and len(sia_cal) > 0 and ("name" in sia_cal.columns):
        tmp = sia_cal.copy()
        if "time" in tmp.columns:
            tmp["time"] = pd.to_datetime(tmp["time"], errors="coerce")
            tmp = tmp.sort_values("time")
        sia_names_sorted = [n for n in tmp["name"].tolist() if n in imm_aligned.columns]
 
    comp_list = []
    for c in ["mcv1", "mcv2"]:
        if c in imm_aligned.columns:
            comp_list.append(c)
    comp_list += [c for c in sia_names_sorted if c not in comp_list]
    if "infected" in imm_aligned.columns:
        comp_list.append("infected")
 
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(6, 3, figure=fig)
 
    vax_ax  = axes_setup(fig.add_subplot(gs[:4, 0]), inplace=False)
    vax_leg = fig.add_subplot(gs[-1, 0])
    fit_ax  = axes_setup(fig.add_subplot(gs[:3, 1:3]), inplace=False)
    rr_ax   = axes_setup(fig.add_subplot(gs[3:, 1:3], sharex=fit_ax), inplace=False)
 
    profile_colors = {}
    profile_colors["mcv1"] = "k"
    profile_colors["mcv2"] = "grey"
 
    if len(sia_names_sorted) > 0:
        sia_cmap = plt.get_cmap("Blues")
        sia_cols = [sia_cmap(i) for i in np.linspace(0.15, 0.90, len(sia_names_sorted))]
        for name, col in zip(sia_names_sorted, sia_cols):
            profile_colors[name] = col
 
    profile_colors["infected"] = "xkcd:light red"
 
    bottom = pd.Series(0.0, index=new_index)
 
    mid_sia_name = None
    if len(sia_names_sorted) > 0:
        mid_sia_name = sia_names_sorted[len(sia_names_sorted) // 2]
 
    for comp in comp_list:
        vals = imm_aligned[comp].reindex(new_index).fillna(0.0).astype(float).clip(lower=0.0)
        if comp == "mcv1":
            label = "MCV1"
        elif comp == "mcv2":
            label = "MCV2"
        elif comp == "infected":
            label = "Infection-derived immunity"
        elif (mid_sia_name is not None) and (comp == mid_sia_name):
            label = "Catch-up vaccine"
        else:
            label = None
 
        col = profile_colors.get(comp, "grey")
 
        vax_ax.bar(
            new_index,
            vals.values,
            width=0.666,
            bottom=bottom.values,
            color=col,
            edgecolor="none",
        )
 
        if label is not None:
            vax_leg.fill_between([], [], color=col, label=label)
 
        bottom = bottom + vals
 
    vax_ax.set_ylim((0, 1))
    vax_ax.set_xticks(list(new_index)[0::4])
    vax_ax.set_xticklabels(["`" + str(i)[-2:] for i in list(new_index)[0::4]])
    vax_ax.set_xlabel("Birth cohort")
    vax_ax.set_ylabel("Immune fraction")
    vax_ax.grid(alpha=0.2)
 
    vax_leg.axis("off")
    vax_leg.legend(frameon=False, loc="center", fontsize=14, bbox_to_anchor=(0.5, 1.5))
 
    theta_hat = np.asarray(result["x"]).reshape(-1)
    cov = np.asarray(cov)
 
    fit_low = None
    fit_high = None
    samples = np.random.multivariate_normal(theta_hat, cov, size=1000)
    rr_samps = 1.0 / (1.0 + np.exp(-samples))
    fit_samps = rr_samps * (phi_aligned.values[None, :])
    fit_low = np.percentile(fit_samps, 2.5, axis=0)
    fit_high = np.percentile(fit_samps, 97.5, axis=0)

 
    if (fit_low is not None) and (fit_high is not None):
        fit_ax.fill_between(
            sf.index,
            fit_low,
            fit_high,
            facecolor="k",
            edgecolor="none",
            alpha=0.25,
            zorder=2,
        )
 
    fit_ax.plot(sf["fit"], color="k", lw=4, label="Survival model", zorder=3)
    fit_ax.plot(sf["cases"], color=colors[2], lw=4, label="Observed cases", zorder=4)
    fit_ax.plot(
        r_ls * phi_aligned,
        color="xkcd:saffron",
        lw=6,
        zorder=3,
        ls="dashed",
        label=r"With const. $r_t$",
    )
 
    fit_ax.set_ylabel("Cases (per year)")
    fit_ax.set_ylim((0, None))
    fit_ax.legend(frameon=False, fontsize=14, loc="upper left")
    fit_ax.grid(alpha=0.2)
 
    rr = pd.to_numeric(sf["rr"], errors="coerce").fillna(0.0)
    rr_std = pd.to_numeric(sf["rr_std"], errors="coerce").fillna(0.0)
 
    rr_ax.fill_between(
        sf.index,
        100.0 * (rr - 2.0 * rr_std),
        100.0 * (rr + 2.0 * rr_std),
        facecolor=colors[3],
        edgecolor="none",
        alpha=0.2,
        zorder=0,
    )
    rr_ax.fill_between(
        sf.index,
        100.0 * (rr - 1.0 * rr_std),
        100.0 * (rr + 1.0 * rr_std),
        facecolor=colors[3],
        edgecolor="none",
        alpha=0.4,
        zorder=1,
    )
    rr_ax.plot(
        sf.index,
        100.0 * rr,
        color=colors[3],
        lw=6,
        label="Estimated reporting rate",
        zorder=2,
    )
 
    if ("lab_rej" in sf.columns) and (pd.to_numeric(sf["lab_rej"], errors="coerce").fillna(0.0).mean() > 0):
        lab_rej = pd.to_numeric(sf["lab_rej"], errors="coerce").fillna(0.0)
        nmfr = 100.0 * (rr.mean()) * lab_rej / (lab_rej.mean())
        rr_ax.plot(
            sf.index,
            nmfr,
            color="k",
            lw=2,
            ls="dashed",
            marker="o",
            markersize=10,
            label="Trend in rejected cases",
        )
 
    rr_ax.set_ylabel("Reporting rate (%)")
    rr_ax.set_ylim((0, None))
    rr_ax.legend(frameon=False, fontsize=14, loc="upper left")
    rr_ax.grid(alpha=0.2)
 
    rr_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
 
    region_name = province.title() if province is not None else "Pakistan"
    fig.suptitle(f"{title_prefix} {region_name}", fontsize=22)
    fig.subplots_adjust(top=0.92, bottom=0.07, left=0.06, right=0.97,
                        hspace=0.6, wspace=0.35)
 
    return fig
 

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    # Load datasets
    prov_df = pd.read_pickle(PROV_PKL)
    nat_df  = pd.read_pickle(NAT_PKL)

    # Truncate to case data availability
    prov_df = prov_df.loc[prov_df["time"] <= DATA_CUTOFF].copy()
    nat_df = nat_df.loc[nat_df["time"] <= DATA_CUTOFF].copy()

    # SIA calendar
    sia_cal = pd.read_pickle(SIA_CAL_PKL).copy()
    sia_cal["time"] = pd.to_datetime(sia_cal["time"], errors="coerce")
    sia_cal = sia_cal.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Cohort axis: prefer earliest start_elig year
    if "start_elig" in sia_cal.columns:
        sia_cal["start_elig"] = pd.to_datetime(sia_cal["start_elig"], errors="coerce")
        cohort_start = int(sia_cal["start_elig"].dt.year.min())
    else:
        cohort_start = int(min(prov_df["time"].dt.year.min(), nat_df["time"].dt.year.min()))

    cohort_start = min(cohort_start, PROFILE_START_PLOT)

    cohort_years_full = np.arange(cohort_start, MODEL_END_YEAR + 1, dtype=int)
    model_years = np.arange(MODEL_START_YEAR, MODEL_END_YEAR + 1, dtype=int)

    # Age-at-infection distributions (from StratifiedDistribution outputs)
    age_dists_inf = pd.read_pickle(AGE_AT_INF_PKL)
    age_dists_inf.index.set_names(["province", "birth_year", "age"], inplace=True)

    # Parameters for the callable core
    params = SurvivalPriorParams(
        mcv1_effic=MCV1_EFFIC,
        mcv2_effic=MCV2_EFFIC,
        sia_vax_effic=SIA_VAX_EFFIC,
        model_start_year=MODEL_START_YEAR,
        pI_right=0.1,
    )

    with PdfPages(OUT_PDF) as pdf:
        for province in REGIONS:
            region_tag = "national" if province is None else str(province)
            print(f"\nRunning {region_tag}...")
    
            # Build prA_given_I (cohort x age-years)
            if province is not None:
                prA_given_I = age_dists_inf.loc[province, "avg"].unstack()
            else:
                prA_given_I = age_dists_inf.loc["national", "avg"].unstack()
    
            prA_given_I = prA_given_I.reindex(cohort_years_full)
            prA_given_I.columns = prA_given_I.columns.astype(int)
    
            # Build epi_sm and precomputed timeline once per region
            epi_sm, births_y = get_epi_semimonth_and_births(province, prov_df, nat_df)
            age_columns = prA_given_I.columns.values.astype(int)
    
            timeline = build_precomputed_timeline(
                epi_sm=epi_sm,
                sia_cal=sia_cal,
                province=province,
                cohort_years_full=cohort_years_full,
                age_columns=age_columns,
            )
    
            # Call core
            out = run_region_survival_prior(
                province=province,
                epi_sm=epi_sm,
                births_y=births_y,
                sia_cal=sia_cal,
                prA_given_I=prA_given_I,
                cohort_years_full=cohort_years_full,
                model_years=model_years,
                params=params,
                return_rr=True,
                precomputed_timeline=timeline,
            )
    
            imm_profile = out["imm_profile"]
            prov_cases = out["prov_cases"]
    
            # Add "infected" column so plots show the red segment
            cohorts = out["cohorts"]
            ages    = out["ages"]
            prA_and_I = out["prA_and_I"]
            
            end_year = int(prov_cases.index.max() + 1) if prov_cases is not None else int(MODEL_END_YEAR + 1)
            max_age_by_cohort = np.clip(end_year - cohorts, 0, int(ages.max())).astype(int)
            cum = np.cumsum(prA_and_I, axis=1)
            infected_by_end = np.array([cum[i, max_age_by_cohort[i]] for i in range(len(cohorts))], dtype=float)
            imm_profile["infected"] = pd.Series(infected_by_end, index=cohorts).reindex(imm_profile.index).fillna(0.0).clip(0.0, 1.0)
    
            rrm = out["rr_model"]
            fig = model_overview(
                province,
                imm_profile,
                sia_cal,
                prA_given_I,
                prov_cases,
                result=rrm["result"],
                cov=rrm["cov"],
                phi=rrm["phi"],
                r_ls=rrm["r_ls"],
                PROFILE_START_PLOT=PROFILE_START_PLOT
            )
            pdf.savefig(fig)
            plt.show()
            plt.close(fig)
