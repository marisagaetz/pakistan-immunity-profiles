""" ZeroInflatedNumKids.py

Zero-inflated negative binomial approach to estimating number of kids by 
mother's age bin in each province.

Two-part model per province:
  For age bins with excess zeros (typically older women):
    (A) logit model for P(Y>0)
    (B) NB2 model for Y | Y>0, with truncation normalization in the pmf

  For age bins without excess zeros (typically younger women):
    Plain NB2 fitted on all data (including zeros), which naturally
    accommodates zeros.

Dependencies: survey_io, model_utils
Outputs:
  - ../../pickle_jar/num_brs_by_province.pkl
  - ../../_plots/num_brs_by_province.pdf

Design choices:
    - Zero inflation (hurdle) is applied only to age bins >= 30. For younger 
        women, plain NB appears to fit better, perhaps because "women who
        will never have children" is not yet distinguishable from "women
        who haven't had children yet." The cutoff can be adjusted via 
        HURDLE_AGE_CUTOFF.
    - Year is modeled as a centered linear term for the same reasons as in 
        AgeAtKthKid.py (only 3 national survey rounds).
"""

# For filepaths
import os

# Input/output functionality is built on top of pandas
import numpy as np
import pandas as pd

# For model fitting
import statsmodels.api as sm
from scipy.stats import nbinom

# For plotting and making PDFs
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# For loading, renaming, and unifying DHS and MICS data
from demography.survey_io import load_survey, infer_survey_folder

# For building design matrices and prediction grids
from demography.births.model_utils import (
    build_design_matrix, build_prediction_grid
)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

# Should align with YEAR_MAX in AgeAtKthKid.py
YEAR_MAX = 2018

# Use DHS data only vs DHS + MICS
DHS_ONLY = True

# Age bins at or above this lower bound use the hurdle model.
HURDLE_AGE_CUTOFF = 30

# List of province plots you want shown (all will be included in the 
# output PDF regardless of this list)
PROVS_TO_SHOW = ["punjab", "sindh", "kp", "balochistan", "ict", "ajk", "gb"]

BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SURVEYS = os.path.join(BASE, "_data", "_surveys")

# ------------------------------------------------------------------
# Load DHS and MICS data
# ------------------------------------------------------------------

dhs_ir_paths = [
    os.path.join(SURVEYS, "DHS5_2006", "PKIR52DT", "pkir52fl.dta"),
    os.path.join(SURVEYS, "DHS6_2012", "PKIR61DT", "PKIR61FL.DTA"),
    os.path.join(SURVEYS, "DHS7_2017", "PKIR71DT", "PKIR71FL.DTA"),
]

ir_columns = ["year", "mom_age", "province", "mom_edu", "num_brs", "area", "weight"]

irs = {
    infer_survey_folder(path): load_survey(path, True, True, ir_columns)
    for path in dhs_ir_paths
}

mics_wm_paths = [
    os.path.join(SURVEYS,"MICS4_Balochistan_2010","wm.sav"),
    os.path.join(SURVEYS,"MICS4_Punjab_2011","wm.sav"),
    os.path.join(SURVEYS,"MICS5_GB_2016","wm.sav"),
    os.path.join(SURVEYS,"MICS5_KP_2016","wm.sav"),
    os.path.join(SURVEYS,"MICS5_Punjab_2014","wm.sav"),
    os.path.join(SURVEYS,"MICS5_Sindh_2014","wm.sav"),
    os.path.join(SURVEYS,"MICS6_Balochistan_2019","wm.sav"),
    os.path.join(SURVEYS,"MICS6_KP_2019","wm.sav"),
    os.path.join(SURVEYS,"MICS6_Punjab_2017","wm.sav"),
    os.path.join(SURVEYS,"MICS6_Sindh_2018","wm.sav"),
]

wm_columns = ["year", "mom_age", "province", "mom_edu", "num_brs", "area", "weight"]

wms = {
    infer_survey_folder(path): load_survey(path, True, True, wm_columns)
    for path in mics_wm_paths
}

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

def nb_pmf_from_mu_alpha(k, mu, alpha):
    """
    NB2 parameterization Var = mu + alpha*mu^2.
    scipy nbinom uses (r, p): r=1/alpha, p=r/(r+mu).
    """
    mu = np.asarray(mu, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    mu = np.clip(mu, 1e-12, None)
    alpha = np.clip(alpha, 1e-12, None)
    r = 1.0 / alpha
    p = r / (r + mu)
    return nbinom.pmf(k, r, p)

def hurdle_nb_pmf(k, mu_pos, alpha, pi0):
    k = np.asarray(k, dtype=int)
    nb = nb_pmf_from_mu_alpha(k, mu=mu_pos, alpha=alpha)
    nb0 = nb_pmf_from_mu_alpha(0, mu=mu_pos, alpha=alpha)
    
    out = np.zeros_like(nb, dtype=float)
    out[k == 0] = pi0
    mask = (k > 0)
    out[mask] = (1.0 - pi0) * nb[mask] / (1.0 - nb0)
    return out

def estimate_alpha_aggregate(y, mu_hat, min_n=10, floor=0.01):
    """
    Aggregate Pearson alpha estimate from NB2 variance identity:
      Var(Y) = mu + alpha * mu^2
      => alpha = mean((y - mu)^2 - mu) / mean(mu^2)
    """
    if len(y) < min_n:
        return None
    numer = float(np.mean((y - mu_hat)**2 - mu_hat))
    denom = float(np.mean(mu_hat**2))
    if denom < 1e-12:
        return None
    return max(numer / denom, floor)

def fit_nb_two_pass(y, X, estimate_alpha_fn):
    """
    Two-pass NB fitting: first with default alpha=1.0, then refit 
    with Pearson-estimated alpha.
    """
    mod = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=1.0))
    res = mod.fit()
    mu_hat = np.clip(res.predict(X), 1e-8, None)
    alpha = estimate_alpha_fn(y, mu_hat) or 0.01
    
    mod = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha))
    res = mod.fit()
    mu_hat = np.clip(res.predict(X), 1e-8, None)
    
    return res, mu_hat, alpha

# ------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Assemble DHS and MICS data
    # ------------------------------------------------------------------
    dhs = pd.concat(irs.values(), axis=0).reset_index(drop=True)
    if "sweight" in dhs.columns:
        use_special = (dhs["weight"] == 0) & (dhs["sweight"] > 0)
        dhs.loc[use_special, "weight"] = dhs.loc[use_special, "sweight"]
    
    mics = pd.concat(wms.values(), axis=0)
    mics = mics.loc[mics["weight"] > 0].reset_index(drop=True)

    variables = ["survey","mom_age","province","area","mom_edu","year","num_brs","weight"]
    
    if DHS_ONLY:
        df = dhs[variables].copy()
    else:
        df = pd.concat([dhs[variables], mics[variables]], axis=0)

    df = df.loc[(df.notnull().all(axis=1)) & (df["year"] <= YEAR_MAX)].reset_index(drop=True)
    df["weight"] = df.groupby(["survey", "province"])["weight"].transform(lambda w: w / w.sum())

    print("\nCleaned and compiled dataset...")
    print(df)

    outputs = []
    alpha_rows = []

    cat_cols = ["area", "mom_edu", "mom_age"]
    grid_names = ["province", "area", "mom_edu", "mom_age", "year"]

    for province, sf in df.groupby("province"):
        sf = sf.copy()

        # Define categorical covariates and center year
        for c in cat_cols:
            sf[c] = sf[c].astype("category")
        sf["year_c"] = sf["year"].astype(float) - sf["year"].astype(float).mean()

        # Build training design matrix
        X = build_design_matrix(sf, cat_cols)

        # Hurdle component: logit for I(Y>0)
        y_binary = (sf["num_brs"].to_numpy() > 0).astype(int)
        logit_mod = sm.GLM(y_binary, X, family=sm.families.Binomial())
        logit_res = logit_mod.fit()

        # Hurdle component: NB2 for Y | Y>0
        sf_pos = sf.loc[sf["num_brs"] > 0].copy()
        sf_pos["year_c"] = sf_pos["year"].astype(float) - sf["year"].astype(float).mean()
        Xp = build_design_matrix(sf_pos, cat_cols, X_template=X)

        y_pos = sf_pos["num_brs"].to_numpy()
        hurdle_nb_res, mu_hat_pos, alpha_hurdle_prov = fit_nb_two_pass(
            y_pos, Xp, estimate_alpha_aggregate
        )

        # Plain NB component: NB2 for Y (all data including zeros)
        y_all = sf["num_brs"].to_numpy().astype(float)
        plain_nb_res, mu_hat_all, alpha_plain_prov = fit_nb_two_pass(
            y_all, X, estimate_alpha_aggregate
        )

        # Hurdle selection: use hurdle for age bins >= HURDLE_AGE_CUTOFF
        age_cats = sf["mom_age"].cat.categories
        use_hurdle = {
            ab: int(str(ab).split("-")[0]) >= HURDLE_AGE_CUTOFF
            for ab in age_cats
        }

        alpha_grid = pd.DataFrame({
            "mom_age": age_cats,
            "alpha_prov_age": [
                alpha_hurdle_prov if use_hurdle[ab] else alpha_plain_prov
                for ab in age_cats
            ],
            "use_hurdle": [use_hurdle[ab] for ab in age_cats],
            "province": province,
        })
        alpha_rows.append(alpha_grid)

        # Prediction grid
        grid = build_prediction_grid(
            province, sf, YEAR_MAX,
            cat_cols=cat_cols,
            grid_names=grid_names
        )
        Xg = build_design_matrix(grid, cat_cols, X_template=X)

        # Predictions from all three models
        logit_pred = logit_res.get_prediction(Xg)
        hurdle_pi0 = 1.0 - logit_pred.predicted_mean
        hurdle_pi0_se = logit_pred.se_mean

        hurdle_nb_pred = hurdle_nb_res.get_prediction(Xg)
        hurdle_mu_pos = np.clip(np.asarray(hurdle_nb_pred.predicted_mean, float), 1e-8, 30.0)
        hurdle_mu_pos_se = hurdle_nb_pred.se_mean

        plain_nb_pred = plain_nb_res.get_prediction(Xg)
        plain_mu = np.clip(np.asarray(plain_nb_pred.predicted_mean, float), 1e-8, 30.0)
        plain_mu_se = plain_nb_pred.se_mean

        # Assign per age bin
        is_hurdle = grid["mom_age"].map(use_hurdle).values

        grid["pi_zero"] = np.where(is_hurdle, hurdle_pi0, 0.0)
        grid["mean_num_brs_pos"] = np.where(is_hurdle, hurdle_mu_pos, plain_mu)
        grid["mean_num_brs"] = np.where(
            is_hurdle,
            (1.0 - hurdle_pi0) * hurdle_mu_pos,
            plain_mu,
        )
        grid["se_pi_zero"] = np.where(is_hurdle, hurdle_pi0_se, 0.0)
        grid["se_mean_num_brs_pos"] = np.where(is_hurdle, hurdle_mu_pos_se, plain_mu_se)

        outputs.append(grid[grid_names + [
            "pi_zero", "mean_num_brs_pos", "mean_num_brs",
            "se_pi_zero", "se_mean_num_brs_pos"
        ]])

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------
    out_mu = pd.concat(outputs, ignore_index=True)
    out_alpha = pd.concat(alpha_rows, ignore_index=True)
    out = out_mu.merge(out_alpha, on=["province", "mom_age"], how="left")
    out.to_pickle("../../pickle_jar/num_brs_by_province.pkl")

    print("\nOutput columns:", out.columns.tolist())
    print(out.head())

    # ---------------------------------------------------------
    # Plots
    # ---------------------------------------------------------
    pred = out.set_index(["province","area","mom_edu","mom_age","year"]).sort_index()

    age_bins = sorted(df["mom_age"].unique(), key=lambda s: int(str(s).split("-")[0]))
    k = np.arange(0, 15)

    cmap = plt.get_cmap("magma")
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(age_bins))]

    with PdfPages("../../_plots/num_brs_by_province.pdf") as book:
        print("\nMaking a book of plots...")

        for province, sf in df.groupby("province"):
            fig, axes = plt.subplots(2, 4, sharex=True, sharey=False, figsize=(16, 6))
            axes = axes.reshape(-1)
            for ax in axes:
                ax.spines["left"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            axes[-1].axis("off")

            for i, ab in enumerate(age_bins[:7]):
                ax = axes[i]
                af = sf.loc[sf["mom_age"] == ab, ["num_brs", "area", "mom_edu", "year", "weight"]].copy()
                
                if af.empty:
                    ax.text(0.5, 0.5, f"No data for {ab}",
                            ha="center", va="center", fontsize=16)
                    ax.set_yticks([])
                    continue

                hist = af["num_brs"].value_counts().sort_index()
                hist = hist / hist.sum()

                w = (
                    af.groupby(["area", "mom_edu", "year"])["weight"]
                      .sum()
                      .rename("w")
                      .reset_index()
                )
                w["weight"] = w["w"] / w["w"].sum()

                alpha_vals = out.loc[
                    (out["province"] == province) & (out["mom_age"] == ab),
                    "alpha_prov_age"
                ]
                if alpha_vals.empty:
                    ax.set_yticks([])
                    continue
                alpha = float(alpha_vals.iloc[0])

                hurdle_flag = out.loc[
                    (out["province"] == province) & (out["mom_age"] == ab),
                    "use_hurdle"
                ]
                is_hurdle_bin = bool(hurdle_flag.iloc[0]) if not hurdle_flag.empty else False

                mix_pmf = np.zeros_like(k, dtype=float)
                for row in w.itertuples(index=False):
                    area = row.area
                    edu = row.mom_edu
                    yr = int(row.year)
                    wt = float(row.weight)

                    mu = float(pred.loc[(province, area, edu, ab, yr), "mean_num_brs_pos"])
                    pi0 = float(pred.loc[(province, area, edu, ab, yr), "pi_zero"])

                    if is_hurdle_bin:
                        mix_pmf += wt * hurdle_nb_pmf(k, mu_pos=mu, alpha=alpha, pi0=pi0)
                    else:
                        mix_pmf += wt * nb_pmf_from_mu_alpha(k, mu=mu, alpha=alpha)

                if mix_pmf.sum() > 0:
                    mix_pmf = mix_pmf / mix_pmf.sum()

                ax.plot(k, mix_pmf, color=colors[i], lw=6, zorder=4)
                ax.bar(hist.index, hist.values, width=0.666, color="grey", lw=2, zorder=1)

                label = f"{ab} years"
                if is_hurdle_bin:
                    label += " (H)"
                ax.text(0.99, 0.99, label, fontsize=20, color="k",
                        ha="right", va="top", transform=ax.transAxes)

                ax.set_yticks([])
                ax.set_ylim((0, None))
                if i >= 4:
                    ax.set_xlabel("Number of kids")

            axes[-1].plot([], lw=6, color=colors[min(3, len(colors)-1)], label="Model fit")
            axes[-1].bar([], [], color="grey", label="Survey data")
            axes[-1].legend(loc="center", frameon=False)

            fig.suptitle("Number of kids for moms by age in " + str(province).title())
            fig.tight_layout(rect=[0, 0.0, 1, 0.9])
            book.savefig(fig)
            
            if province in PROVS_TO_SHOW:
                plt.show()
            plt.close(fig)

    print("...done!")

