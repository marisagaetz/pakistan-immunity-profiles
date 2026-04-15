"""
MCVOneProbability.py

Estimate P(MCV1 received) by child's birth year, stratified by:
  - province
  - area (urban/rural)
  - mom_edu

Approach:
  - Pool DHS (KR+IR merge) and MICS (CH+WM merge)
  - Restrict to children aged 12-23 months at interview
  - Fit, per province, a statsmodels Binomial GLM:
      logit(p) = FE(area, mom_edu) + bs(year)
    where bs(year) is a B-spline basis providing smooth year effects.
  - Uncertainty via the GLM coefficient covariance.

Dependencies: survey_io

Outputs:
  - ../../pickle_jar/mcv1_prob_pred.pkl
"""

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm
import patsy

from demography.survey_io import load_survey, infer_survey_folder

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

MIN_YEAR = 2005
MAX_YEAR = 2018

# B-spline degrees of freedom for the year effect (lower = smoother)
YEAR_SPLINE_DF = 4

BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SURVEYS = os.path.join(BASE, "_data", "_surveys")

# ------------------------------------------------------------
# Load DHS and MICS data
# ------------------------------------------------------------

dhs_kr_paths = [
    os.path.join(SURVEYS, "DHS5_2006", "PKKR52DT", "pkkr52fl.dta"),
    os.path.join(SURVEYS, "DHS6_2012", "PKKR61DT", "PKKR61FL.DTA"),
    os.path.join(SURVEYS, "DHS7_2017", "PKKR71DT", "PKKR71FL.DTA"),
]

kr_columns = ["caseid", "bord", "child_DoB", "live_child", "mom_DoB", "interview_date", "mcv1"]

krs = {
    infer_survey_folder(path): load_survey(path, False, True, kr_columns)
    for path in dhs_kr_paths
}

dhs_ir_paths = [
    os.path.join(SURVEYS, "DHS5_2006", "PKIR52DT", "pkir52fl.dta"),
    os.path.join(SURVEYS, "DHS6_2012", "PKIR61DT", "PKIR61FL.DTA"),
    os.path.join(SURVEYS, "DHS7_2017", "PKIR71DT", "PKIR71FL.DTA"),
]

ir_columns = ["caseid", "mom_edu", "province", "area", "num_brs", "mom_age"]

irs = {
    infer_survey_folder(path): load_survey(path, True, True, ir_columns)
    for path in dhs_ir_paths
}

mics_ch_paths = [
    os.path.join(SURVEYS, "MICS4_Balochistan_2010", "ch.sav"),
    os.path.join(SURVEYS, "MICS4_Punjab_2011", "ch.sav"),
    os.path.join(SURVEYS, "MICS5_GB_2016", "ch.sav"),
    os.path.join(SURVEYS, "MICS5_KP_2016", "ch.sav"),
    os.path.join(SURVEYS, "MICS5_Punjab_2014", "ch.sav"),
    os.path.join(SURVEYS, "MICS5_Sindh_2014", "ch.sav"),
    os.path.join(SURVEYS, "MICS6_Balochistan_2019", "ch.sav"),
    os.path.join(SURVEYS, "MICS6_KP_2019", "ch.sav"),
    os.path.join(SURVEYS, "MICS6_Punjab_2017", "ch.sav"),
    os.path.join(SURVEYS, "MICS6_Sindh_2018", "ch.sav"),
]

ch_columns = ["cluster", "hh", "line_num", "child_birth_day", "child_birth_mon", "child_birth_year", "child_age", "mcv1"]

chs = {
    infer_survey_folder(path): load_survey(path, False, True, ch_columns)
    for path in mics_ch_paths
}

mics_wm_paths = [
    os.path.join(SURVEYS, "MICS4_Balochistan_2010", "wm.sav"),
    os.path.join(SURVEYS, "MICS4_Punjab_2011", "wm.sav"),
    os.path.join(SURVEYS, "MICS5_GB_2016", "wm.sav"),
    os.path.join(SURVEYS, "MICS5_KP_2016", "wm.sav"),
    os.path.join(SURVEYS, "MICS5_Punjab_2014", "wm.sav"),
    os.path.join(SURVEYS, "MICS5_Sindh_2014", "wm.sav"),
    os.path.join(SURVEYS, "MICS6_Balochistan_2019", "wm.sav"),
    os.path.join(SURVEYS, "MICS6_KP_2019", "wm.sav"),
    os.path.join(SURVEYS, "MICS6_Punjab_2017", "wm.sav"),
    os.path.join(SURVEYS, "MICS6_Sindh_2018", "wm.sav"),
]

wm_columns = ["cluster", "hh", "line_num", "interview_day", "interview_mon", "year", "mom_edu", "mom_age", "area", "province"]

wms = {
    infer_survey_folder(path): load_survey(path, True, True, wm_columns)
    for path in mics_wm_paths
}

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def cms_to_datetime(cmc, d=15, min_cmc=1, max_cmc=2000):
    """Convert DHS century-month codes to datetime."""
    s = pd.to_numeric(pd.Series(cmc), errors="coerce").astype(float)
    s = s.where((s >= min_cmc) & (s <= max_cmc), np.nan)
    years = 1900 + np.floor((s - 1) / 12.0)
    months = ((s - 1) % 12) + 1
    return pd.to_datetime({"year": years, "month": months, "day": d}, errors="coerce")


def _build_design_matrix(data, year_col, edu_col, area_col, year_df,
                         design_info=None, year_lower=None, year_upper=None):
    """
    Build a patsy design matrix with:
      - B-spline basis for year (smooth temporal trend)
      - Categorical dummies for area and education (fixed effects)
    """
    if design_info is not None:
        return patsy.dmatrix(design_info, data, return_type="dataframe")

    n_unique_years = data[year_col].nunique()
    if n_unique_years > year_df + 1:
        # Pin the B-spline boundaries to the full year window so the
        # basis spans the same range at fit time and prediction time.
        lo = year_lower if year_lower is not None else data[year_col].min()
        hi = year_upper if year_upper is not None else data[year_col].max()
        year_term = (
            f"bs({year_col}, df={year_df}, degree=3, "
            f"lower_bound={lo}, upper_bound={hi})"
        )
    else:
        year_term = year_col

    formula = f"{year_term} + C({edu_col}) + C({area_col})"
    X = patsy.dmatrix(formula, data, return_type="dataframe")
    return X


def fit_glm_one_province(
    df_prov,
    y_col="mcv1",
    year_col="birth_year",
    edu_col="mom_edu",
    area_col="area",
    year_df=YEAR_SPLINE_DF,
    year_min=MIN_YEAR,
    year_max=MAX_YEAR,
):
    """
    Fit a Binomial GLM with logit link for one province:
        logit(P(mcv1=1)) = intercept + bs(year) + C(area) + C(edu)
    """
    dfp = df_prov.copy()

    # Build design matrix with B-spline knots spanning the full window
    X = _build_design_matrix(
        dfp, year_col, edu_col, area_col, year_df,
        year_lower=year_min, year_upper=year_max,
    )

    # Fit the GLM
    y = dfp[y_col].astype(float).values
    glm = sm.GLM(y, X, family=sm.families.Binomial())
    glm_result = glm.fit()

    # Full year grid (every year in the window, not just observed)
    all_years = list(range(year_min, year_max + 1))

    return {
        "success": True,
        "glm_result": glm_result,
        "design_info": X.design_info,
        "year_col": year_col,
        "edu_col": edu_col,
        "area_col": area_col,
        "year_df": year_df,
        "years": all_years,
        "edu_levels": sorted(dfp[edu_col].dropna().unique().tolist()),
        "area_levels": sorted(dfp[area_col].dropna().unique().tolist()),
    }


def predict_with_uncertainty(
    fit,
    grid,
    year_col="birth_year",
    edu_col="mom_edu",
    area_col="area",
):
    """
    Predict P(MCV1) and model uncertainty for a prediction grid.
    Uses the GLM coefficient covariance to compute Var(eta) for 
    each prediction row, then applies the delta method.
    """
    glm_result = fit["glm_result"]
    design_info = fit["design_info"]

    # Build design matrix for the prediction grid using the same
    # column structure as the training data
    Xg = _build_design_matrix(
        grid, fit["year_col"], fit["edu_col"], fit["area_col"],
        fit["year_df"], design_info=design_info,
    )

    # Predicted probabilities
    prob = glm_result.predict(Xg)
    prob = np.asarray(prob)

    # Model uncertainty via delta method:
    #   Var(eta_i) = x_i' @ cov(beta) @ x_i
    #   Var(p_i) = (p_i * (1 - p_i))^2 * Var(eta_i)
    Xg_arr = np.asarray(Xg)
    beta_cov = np.asarray(glm_result.cov_params())
    var_eta = np.sum((Xg_arr @ beta_cov) * Xg_arr, axis=1)

    dp_deta = prob * (1.0 - prob)
    var_prob = dp_deta ** 2 * var_eta

    return prob, var_prob


def build_mcv1_predictions_with_obs(
    df,
    y_col="mcv1",
    year_col="birth_year",
    edu_col="mom_edu",
    area_col="area",
    prov_col="province",
    year_df=YEAR_SPLINE_DF,
    min_unique_years=1,
):
    """
    Fit one GLM per province, return output with predictions,
    observed counts, and model uncertainty.

    Variance has two components:
      - var_model: from the GLM coefficient covariance (wider for
        years far from observed data, via the spline basis).
      - var_sampling: p(1-p)/n for cells with observations.  For
        unobserved cells, uses the maximum observed sampling variance
        from the same (province, area, mom_edu) group as a floor.
    """
    # Observed counts for comparison
    group_cols = [prov_col, area_col, edu_col, year_col]
    mcv_yearly = (
        df.groupby(group_cols)[y_col]
        .agg(["sum", "count"])
        .reset_index()
        .rename(columns={"sum": "y", "count": "n"})
    )
    mcv_yearly["y"] = mcv_yearly["y"].astype(int)
    mcv_yearly["n"] = mcv_yearly["n"].astype(int)
    mcv_yearly["prob_mcv1_obs"] = mcv_yearly["y"] / mcv_yearly["n"].replace(0, np.nan)

    out_rows = []

    for prov, df_p in df.groupby(prov_col):
        n_years = df_p[year_col].nunique()
        if n_years < min_unique_years:
            continue

        fit = fit_glm_one_province(
            df_p,
            y_col=y_col,
            year_col=year_col,
            edu_col=edu_col,
            area_col=area_col,
            year_df=year_df,
        )

        print(f"  {prov}: converged={fit['success']}")

        # Build prediction grid over all (area, edu, year) combos
        grid = pd.MultiIndex.from_product(
            [fit["area_levels"], fit["edu_levels"], fit["years"]],
            names=[area_col, edu_col, year_col],
        ).to_frame(index=False)
        grid[prov_col] = prov

        prob, var_model = predict_with_uncertainty(
            fit, grid,
            year_col=year_col, edu_col=edu_col, area_col=area_col,
        )

        grid["prob_mcv1_pred"] = prob
        grid["var_model"] = var_model

        out_rows.append(grid)

    pred_df = (
        pd.concat(out_rows, ignore_index=True) if out_rows
        else pd.DataFrame(
            columns=[prov_col, area_col, edu_col, year_col, "prob_mcv1_pred", "var_model"]
        )
    )

    # Merge observed counts
    out = pred_df.merge(
        mcv_yearly[[prov_col, area_col, edu_col, year_col, "prob_mcv1_obs", "y", "n"]],
        on=[prov_col, area_col, edu_col, year_col],
        how="left",
    )

    # Sampling variance: p(1-p)/n for observed cells
    p = out["prob_mcv1_pred"].astype(float)
    n = pd.to_numeric(out["n"], errors="coerce")
    out["var_sampling"] = (p * (1.0 - p)) / n.replace(0, np.nan)

    # For unobserved cells, floor at max observed sampling variance
    # from the same (province, area, mom_edu) group
    cell_cols = [prov_col, area_col, edu_col]
    max_sampling = (
        out.loc[out["var_sampling"].notna()]
        .groupby(cell_cols)["var_sampling"]
        .max()
        .rename("var_sampling_floor")
    )
    out = out.merge(max_sampling, on=cell_cols, how="left")
    out["var_sampling"] = out["var_sampling"].fillna(out["var_sampling_floor"])
    out.drop(columns=["var_sampling_floor"], inplace=True)

    # Combined variance
    out["var_prob"] = out["var_model"].fillna(0.0) + out["var_sampling"].fillna(0.0)

    out = out[
        [prov_col, area_col, edu_col, year_col,
         "prob_mcv1_pred", "var_model", "var_sampling", "var_prob",
         "prob_mcv1_obs", "y", "n"]
    ].sort_values(
        [prov_col, area_col, edu_col, year_col]
    ).reset_index(drop=True)

    return out


# ------------------------------------------------------------
# Main script
# ------------------------------------------------------------
if __name__ == "__main__":

    # Merge DHS KR + IR
    dhs_parts = []
    for k in krs.keys():
        this_dhs = krs[k].merge(irs[k], on="caseid", how="left", validate="m:1")
        dhs_parts.append(this_dhs)
    dhs = pd.concat(dhs_parts, axis=0).sort_values(["survey", "caseid", "bord"]).reset_index(drop=True)

    # Restrict to alive children aged 12-23 months
    dhs["child_age"] = dhs["interview_date"] - dhs["child_DoB"]
    dhs = dhs.loc[(dhs["live_child"] == "yes") & (dhs["child_age"] >= 12) & (dhs["child_age"] < 24)].copy()
    dhs["birth_year"] = cms_to_datetime(dhs["child_DoB"]).dt.year

    # Merge MICS CH + WM
    mics_parts = []
    for k in chs.keys():
        this_mics = chs[k].merge(wms[k], on=["cluster", "hh", "line_num"], how="left", validate="m:1")
        mics_parts.append(this_mics)
    mics = pd.concat(mics_parts, axis=0).reset_index(drop=True)

    mics["child_DoB"] = pd.to_datetime(
        {"month": mics["child_birth_mon"], "year": mics["child_birth_year"], "day": mics["child_birth_day"]},
        errors="coerce",
    )
    mics["interview_date"] = pd.to_datetime(
        {"month": mics["interview_mon"], "year": mics["year"], "day": mics["interview_day"]},
        errors="coerce",
    )
    mics["child_age_m"] = 12.0 * ((mics["interview_date"] - mics["child_DoB"]).dt.days / 365.0)
    mics["child_age_m"] = mics["child_age_m"].fillna(mics["child_age"])
    mics = mics.loc[(mics["child_age_m"] >= 12) & (mics["child_age_m"] < 24)].copy()
    mics = mics.loc[~mics["mom_age"].isna()].copy()
    mics["birth_year"] = mics["child_birth_year"]

    # Combine and fit
    variables = ["area", "province", "mom_edu", "birth_year", "mcv1"]
    df = pd.concat([dhs[variables], mics[variables]], axis=0).reset_index(drop=True)
    df = df.loc[df.notnull().all(axis=1)].reset_index(drop=True)
    df = df.loc[(df["birth_year"] >= MIN_YEAR) & (df["birth_year"] <= MAX_YEAR)].copy()

    out = build_mcv1_predictions_with_obs(
        df,
        y_col="mcv1",
        year_col="birth_year",
        edu_col="mom_edu",
        area_col="area",
        prov_col="province",
        year_df=YEAR_SPLINE_DF,
    )

    # Standard error on probability scale
    out["std_prob"] = np.sqrt(out["var_prob"].clip(lower=0))

    print("\n", out.head(15))
    print("\nVariance breakdown (means across cells):")
    print(f"  var_model:    {out['var_model'].mean():.8f}")
    print(f"  var_sampling: {out['var_sampling'].mean():.8f}")
    print(f"  var_prob:     {out['var_prob'].mean():.8f}")
    print(f"  std_prob:     {out['std_prob'].mean():.6f}")

    out.to_pickle("../../pickle_jar/mcv1_prob_pred.pkl")