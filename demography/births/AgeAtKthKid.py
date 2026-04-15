""" AgeAtKthKid.py

Log-normal regression to estimate a mom's age at the time of their
children's births. 

Dependencies: survey_io, model_utils
Outputs: 
    - ../../pickle_jar/age_at_kth_kid_by_province.pkl
    - ../../_plots/age_at_kth_kid_by_province.pdf

Note (Pakistan): The MICS4 and MICS5 surveys do not have bh.sav files (which
have info about full birth history). Similarly, the 2019 DHS special survey
does not have a br.dta file. We've therefore omitted these surveys from 
this analysis. 

Design choices:
    - Year is modeled as a linear term because we only have 3 national survey 
        rounds (2006, 2012, 2017). With additional national surveys, consider 
        switching to something with more flexibility in the year term.
    - Birth order is modeled with a B-spline smooth (degree 3, 6 df) because 
        we observe birth orders from 1 to 19 and the relationship with 
        log(mom's age) is nonlinear: early birth orders are more tightly 
        spaced in age than later ones.
"""

# For filepaths
import os

# Input/output functionality is built on top of pandas
import pandas as pd
import numpy as np

# For model fitting
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines

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

# # YEAR_MAX should be the latest year with survey data. Resulting birth 
# rate estimates can be extrapolated later (see YearlyBirths.py).
YEAR_MAX = 2018

# Omits MICS if set to True. Can toggle this for debugging and 
# experimentation with World Bank comparison in YearlyBirths.py estimates
DHS_ONLY = True

# List of province plots you want shown (all will be included in the 
# output PDF regardless of this list)
PROVS_TO_SHOW = ["punjab", "sindh", "kp", "balochistan", "ict", "ajk", "gb"]

BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SURVEYS = os.path.join(BASE, "_data", "_surveys")

# ------------------------------------------------------------------
# Load DHS and MICS data
# ------------------------------------------------------------------

# DHS births recode file paths (births history)
dhs_br_paths = [
    os.path.join(SURVEYS,"DHS5_2006","PKBR52DT","pkbr52fl.dta"),
    os.path.join(SURVEYS,"DHS6_2012","PKBR61DT","PKBR61FL.DTA"),
    os.path.join(SURVEYS,"DHS7_2017","PKBR71DT","PKBR71FL.DTA")
]

# See survey_io file for a list of column names and their meanings
br_columns = ["caseid", "bord", "child_DoB", "mom_DoB", "year"]
    
# Load and process the surveys (via survey_io)
brs = {
    infer_survey_folder(path): load_survey(path, True, True, br_columns)
    for path in dhs_br_paths
}

# DHS individual recode file paths (individual woman's survey)
dhs_ir_paths = [
    os.path.join(SURVEYS, "DHS5_2006", "PKIR52DT", "pkir52fl.dta"),
    os.path.join(SURVEYS, "DHS6_2012", "PKIR61DT", "PKIR61FL.DTA"),
    os.path.join(SURVEYS, "DHS7_2017", "PKIR71DT", "PKIR71FL.DTA"),
]

# See survey_io file for a list of column names and their meanings
ir_columns = ["caseid", "mom_edu", "province", "area", "weight"]

# Load and process the surveys (via survey_io)
irs = {
    infer_survey_folder(path): load_survey(path, False, True, ir_columns)
    for path in dhs_ir_paths
}

# MICS births recode file paths (births history)
mics_bh_paths = [
    os.path.join(SURVEYS,"MICS6_Balochistan_2019","bh.sav"),
    os.path.join(SURVEYS,"MICS6_KP_2019","bh.sav"),
    os.path.join(SURVEYS,"MICS6_Punjab_2017","bh.sav"),
    os.path.join(SURVEYS,"MICS6_Sindh_2018","bh.sav")
]

# See survey_io file for a list of column names and their meanings
bh_columns = ["cluster", "hh", "line_num", "bord", "child_birth_mon", 
              "child_birth_year"]

# Load and process the surveys (via survey_io)
bhs = {
    infer_survey_folder(path): load_survey(path, True, True, bh_columns)
    for path in mics_bh_paths
}

# MICS individual recode file paths (individual woman's survey)
mics_wm_paths = [
    os.path.join(SURVEYS,"MICS6_Balochistan_2019","wm.sav"),
    os.path.join(SURVEYS,"MICS6_KP_2019","wm.sav"),
    os.path.join(SURVEYS,"MICS6_Punjab_2017","wm.sav"),
    os.path.join(SURVEYS,"MICS6_Sindh_2018","wm.sav")   
]

# See survey_io file for a list of column names and their meanings
wm_columns = ["cluster", "hh", "line_num", "year", "mom_birth_mon", 
              "mom_birth_year", "mom_edu", "province", "area", "weight"]

# Load and process the surveys (via survey_io)
wms = {
    infer_survey_folder(path): load_survey(path, False, True, wm_columns)
    for path in mics_wm_paths
}
    
# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------

# Colors for plotting
colors = ["#375E97","#FB6542","#FFBB00","#3F681C"]

def lognorm_pdf(x, mu_ln, var_ln):
    """
    PDF of LogNormal where ln(X) ~ Normal(mu_ln, var_ln).
    x can be array; returns array of same shape.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    xm = x[mask]
    out[mask] = (1.0 / (xm * np.sqrt(2*np.pi*var_ln))) * np.exp(-(np.log(xm) - mu_ln)**2 / (2*var_ln))
    return out

# ------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Assemble DHS and MICS data
    # ------------------------------------------------------------------
    # Merge DHS br and ir on "caseid"
    dhs = []
    for k in brs.keys():
        this_dhs = brs[k].merge(irs[k],
                    on="caseid",
                    how="left",
                    validate="m:1",
                    )
        dhs.append(this_dhs)
    dhs = pd.concat(dhs,axis=0)
    dhs = dhs.sort_values(["survey","caseid","bord"]).reset_index(drop=True)
    
    # Fill in special weights for AJK and GB
    if "sweight" in dhs.columns:
        use_special = (dhs["weight"] == 0) & (dhs["sweight"] > 0)
        dhs.loc[use_special, "weight"] = dhs.loc[use_special, "sweight"]
    
    print("\nThe DHS data for this analysis:")
    print(dhs)

    # Merge MICS bh and wm on "cluster", "hh", and "line_num"
    mics = []
    for k in bhs.keys():
        this_mics = bhs[k].merge(wms[k],
                    on=["cluster","hh","line_num"],
                    how="left",
                    validate="m:1",
                    )
        mics.append(this_mics)
    mics = pd.concat(mics,axis=0).reset_index(drop=True)
    print("\nThe MICS data for this analysis:")
    print(mics)
    
    # Correct for twins, keeping the one with the lower birth order
    mics = mics.loc[~mics[
                ["survey","cluster","hh","line_num",
                "child_birth_mon","child_birth_year"]].duplicated(keep="first")]
    dhs = dhs.loc[~dhs[["survey","caseid","child_DoB"]].duplicated(keep="first")]
    
    # Make a mom's age covariate    
    mics["mom_DoB"] = pd.to_datetime({"month":mics["mom_birth_mon"],
                                      "year":mics["mom_birth_year"],
                                      "day":1})
    mics["child_DoB"] = pd.to_datetime({"month":mics["child_birth_mon"],
                                      "year":mics["child_birth_year"],
                                      "day":1})
    mics["mom_age"] = (mics["child_DoB"]-mics["mom_DoB"]).dt.days/365.
    mics["ln_mom_age"] = np.log(mics["mom_age"])
    dhs["mom_age"] = (dhs["child_DoB"]-dhs["mom_DoB"])/12.
    dhs["ln_mom_age"] = np.log(dhs["mom_age"])

    # ------------------------------------------------------------------
    # Select variables and set up a dataframe for regression
    # ------------------------------------------------------------------
    variables = ["survey", "area", "province", "year", "mom_edu", 
                 "bord", "mom_age", "ln_mom_age", "weight"]

    # Assemble the dataframe based on DHS_ONLY value
    if DHS_ONLY:
        df = dhs[variables].copy() 
    else:
        df = pd.concat([dhs[variables],mics[variables]],axis=0)
    
    # Drop missing covariates and implausibly young moms; subset to date range
    df = df.loc[
        (df.notnull().all(axis=1)) &
        (df["mom_age"] > 5) & 
        (df["year"] <= YEAR_MAX)
    ].reset_index(drop=True)
    
    # Normalize weights by survey and province
    df["weight"] = df.groupby(["survey", "province"])["weight"].transform(lambda w: w / w.sum())

    print("\nCleaned and compiled dataset...")
    print(df.head())

    # Inspect value ranges across surveys
    print("\nVariable values...")
    for c in variables[:-3]:
        values = sorted(df[c].unique())
        print(f"{c} values ({len(values)} of them) = {values}")

    # ------------------------------------------------------------------
    # Fit one model per province
    # ------------------------------------------------------------------
    outputs = []
    
    for province, sf in df.groupby("province"):
        sf = sf.copy()
    
        # Define categorical covariates
        for c in ["area", "mom_edu"]:
            sf[c] = sf[c].astype("category")
    
        # Center year
        sf["year_c"] = sf["year"].astype(float) - sf["year"].astype(float).mean()
    
        # Build training design matrix
        y_s = sf["ln_mom_age"].to_numpy()
        Xs_lin = build_design_matrix(sf, ["area", "mom_edu"])
    
        # Smooth parts: bord only
        bs_s = BSplines(sf[["bord"]], df=[6], degree=[3])
        mod_s = GLMGam(y_s, exog=Xs_lin, smoother=bs_s, family=sm.families.Gaussian())
        res_s = mod_s.fit()
    
        # Build prediction grid and its design matrix
        bmax = int(sf["bord"].max())
        grid_s = build_prediction_grid(
            province, sf, YEAR_MAX,
            cat_cols=["area", "mom_edu"],
            extra_dims={"bord": np.arange(1, bmax + 1)},
            grid_names=["province", "area", "mom_edu", "year", "bord"]
        )
        grid_s["bord_clipped"] = grid_s["bord"].clip(
            lower=int(sf["bord"].min()), upper=bmax
        )
        Xg_s = build_design_matrix(grid_s, ["area", "mom_edu"], X_template=Xs_lin)
    
        # Predict
        pred_result = res_s.get_prediction(exog=Xg_s, exog_smooth=grid_s[["bord_clipped"]])
    
        # Save province output
        out_s = grid_s[["province", "area", "mom_edu", "year", "bord"]].copy()
        out_s["mean_ln_mom_age"] = pred_result.predicted_mean
        out_s["se_mean_ln_mom_age"] = pred_result.se_mean
        out_s["var_ln_mom_age"] = res_s.scale
    
        outputs.append(out_s)
    
    # Print and save all outputs
    out = pd.concat(outputs, ignore_index=True)
    out.to_pickle("../../pickle_jar/age_at_kth_kid_by_province.pkl")
    print("\nFinal output...")
    print(out.head())

    # -------------------------------------------------------------
    # Make a book of plots
    # -------------------------------------------------------------

    # Plotting setup
    cmap = plt.get_cmap("magma")
    bord_values = np.arange(1, 11)
    colors = [cmap(i) for i in np.linspace(0.4, 0.95, len(bord_values))]
    
    df = df.copy()
    out = out.copy()
    pred = out.set_index(["province", "area", "mom_edu", "year", "bord"]).sort_index()
    
    with PdfPages("../../_plots/age_at_kth_kid_by_province.pdf") as book:
        print("\nMaking a book of plots...")
        for province, sf in df.groupby("province"):
            # Histogram data per province
            by_ord = sf[["bord", "mom_age"]].copy()
            by_ord["bord"] = pd.to_numeric(by_ord["bord"], errors="coerce").astype("Int64")
            by_ord = by_ord[by_ord["bord"].between(1, 10)].copy()
    
            # Count exact ages then later bucket to monthly
            by_ord["freq"] = 1
            by_ord = (
                by_ord
                .groupby(["bord", "mom_age"])["freq"]
                .sum()
                .sort_index()
            )
    
            fig, axes = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(15, 6))
            axes = axes.reshape(-1)
    
            for ax in axes:
                ax.spines["left"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(color="grey", alpha=0.2)
    
            for i in bord_values:
                ax = axes[i - 1]
    
                if i not in by_ord.index.get_level_values("bord"):
                    ax.text(0.5, 0.5, f"No data for child {i}",
                            ha="center", va="center", fontsize=16)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
    
                # Histogram series for this birth order
                hist = by_ord.loc[i].copy()
                total = hist.sum()
    
                # Bucket monthly 
                hist.index = (12 * hist.index).astype(int) / 12.0
                hist = hist.groupby(level=0).sum()
    
                # Choose x-grid for pdf evaluation: match the histogram index
                x = hist.index.to_numpy(dtype=float)
                    
                # Weight (area, edu, year) combinations by DHS/MICS weights
                kf = sf.loc[sf["bord"] == i, ["area", "mom_edu", "year", "weight"]].copy()
    
                w = (
                    kf.groupby(["area", "mom_edu", "year"])["weight"]
                      .sum()
                      .rename("w")
                      .reset_index()
                )
                w["weight"] = w["w"] / w["w"].sum()
    
                # Build mixture pdf over (area, edu, year)
                mix_pdf = np.zeros_like(x, dtype=float)
                        
                # Iterate over combinations; pull predicted (mu_ln, var_ln) from 'pred'
                for row in w.itertuples(index=False):
                    area = row.area
                    edu = row.mom_edu
                    yr = int(row.year)
                    wt = float(row.weight)
    
                    mu_ln = float(pred.loc[(province, area, edu, yr, i), "mean_ln_mom_age"])
                    var_ln = float(pred.loc[(province, area, edu, yr, i), "var_ln_mom_age"])
                    mix_pdf += wt * lognorm_pdf(x, mu_ln, var_ln)
    
                # Scale mixture pdf to match histogram total mass (counts)
                if mix_pdf.sum() > 0:
                    mix_pdf = mix_pdf * (total / mix_pdf.sum())
    
                # Plot
                ax.plot(hist.index, hist.values, lw=2, color="k")
                ax.plot(x, mix_pdf, lw=4, color=colors[i - 1])
    
                ax.text(0.01, 0.99, f"Child {i}",
                        fontsize=22, color="k",
                        ha="left", va="top",
                        transform=ax.transAxes)
    
                ax.set_ylim((0, None))
                ax.set_xlim((-1, 51))
                ax.set_yticks([])
    
                if i >= 6:
                    ax.set_xlabel("Mom's age at birth")
                    ax.set_xticks(np.arange(0, 6) * 10)
    
            fig.suptitle("Mom's age at kth childbirth in " + str(province).title())
            fig.tight_layout(rect=[0, 0.0, 1, 0.9])
            book.savefig(fig)
            
            if province in PROVS_TO_SHOW:
                plt.show()
            plt.close(fig)
    
        d = book.infodict()
        d["Title"] = "Age at kth childbirth in Pakistan"
    
    print("...done!")
