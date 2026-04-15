""" survival_prior_core.py

Callable core utilities extracted from SurvivalPrior.py.

Core functionalities:
  - Precomputes event timelines: vaccination event lists and eligible ages
    are built once and reused across calls to compute_profile_from_semimonth.
  - Runs the two-point solve for the lifetime probability of infection per
    birth cohort based on vaccination opportunities and age-at-infection.
  - Solves for the immunity profile-implied reporting rate with RW2 smoothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

def _safe_col(s: str) -> str:
    """Sanitize a string for use as a column-name fragment."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def campaign_colname(campaign_idx: int, sia_row: pd.Series) -> str:
    """
    Build the SIA eligibility column name:
      f"sia__{YYYYMMDD}__{idx:03d}__{safe_name}"
    """
    stamp = pd.to_datetime(sia_row["time"]).strftime("%Y%m%d")
    safe_name = _safe_col(sia_row.get("name", f"campaign_{campaign_idx}"))
    return f"sia__{stamp}__{int(campaign_idx):03d}__{safe_name}"


def get_epi_semimonth_and_births(
    province: Optional[str],
    prov_df: pd.DataFrame,
    nat_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select the semimonthly epi data for one province (or national) and
    compute annual births + variance by cohort year.
    """
    if province is None:
        epi_sm = nat_df.copy()
        epi_sm["province"] = "national"
    else:
        epi_sm = prov_df.loc[prov_df["province"] == province].copy()

    epi_sm["time"] = pd.to_datetime(epi_sm["time"], errors="coerce")
    epi_sm = epi_sm.dropna(subset=["time"]).copy()
    epi_sm["birth_year"] = epi_sm["time"].dt.year.astype(int)

    births_y = (
        epi_sm.groupby("birth_year", as_index=False)
              .agg(births=("births", "sum"), births_var=("births_var", "sum"))
    )
    births_y["births_std"] = np.sqrt(
        pd.to_numeric(births_y["births_var"], errors="coerce")
          .fillna(0.0).clip(lower=0.0)
    )
    births_y = births_y.set_index("birth_year").sort_index()

    return epi_sm, births_y

# ---------------------------------------------------------------------
# Reporting rate smoothing (RW2 penalty)
# ---------------------------------------------------------------------

def smoothing_cost(theta, expI, cases, lam, w=None):
    """Penalized (weighted) least-squares objective for the reporting rate."""
    beta = 1.0 / (1.0 + np.exp(-theta))
    f = beta * expI
    resid2 = (cases - f) ** 2
    ll = float(np.sum(w * resid2)) if w is not None else float(np.sum(resid2))
    return ll + float(theta.T @ (lam @ theta))

def smoothing_grad(theta, expI, cases, lam, w=None):
    """Gradient of smoothing_cost w.r.t. theta."""
    beta = 1.0 / (1.0 + np.exp(-theta))
    resid = beta * expI - cases
    dBeta = beta * expI * (1.0 - beta)
    if w is not None:
        return 2.0 * (w * resid * dBeta + lam @ theta)
    return 2.0 * (resid * dBeta + lam @ theta)

def smoothing_hessian(theta, expI, cases, lam, w=None):
    """Hessian of smoothing_cost w.r.t. theta."""
    beta = 1.0 / (1.0 + np.exp(-theta))
    diag_vals = (
        beta * (1.0 - beta) * expI *
        ((1.0 - 2.0 * beta) * (beta * expI - cases) + beta * (1.0 - beta) * expI)
    )
    if w is not None:
        diag_vals = w * diag_vals
    return 2.0 * (np.diag(diag_vals) + lam)

def fit_reporting_rate_rw2(
    epi_sm: pd.DataFrame,
    phi_year: pd.Series,
    model_years: np.ndarray,
    model_start_year: int,
    cases_col: str = "cases",
    phi_var: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit a year-varying reporting rate r(y) via logit-transformed RW2-penalized
    least squares: cases(y) ≈ r(y) * phi(y).
    """
    prov_cases = epi_sm.copy()
    prov_cases = prov_cases.loc[prov_cases["time"].dt.year >= int(model_start_year)].copy()

    if "discarded" in prov_cases.columns and "lab_rej" not in prov_cases.columns:
        prov_cases = prov_cases.rename(columns={"discarded": "lab_rej"})

    prov_cases = prov_cases.set_index("time").resample("YS").sum()
    prov_cases.index = prov_cases.index.year.astype(int)
    prov_cases = prov_cases.reindex(model_years)
    phi_year = pd.Series(phi_year).reindex(model_years)

    cases_vec = pd.to_numeric(prov_cases[cases_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    phi_vec = pd.to_numeric(phi_year, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Inverse-variance weights from birth-uncertainty-propagated phi variance
    w_vec = None
    if phi_var is not None:
        pv = pd.Series(phi_var).reindex(model_years).fillna(0.0).to_numpy(dtype=float)
        # w_y = 1 / (var_phi_y + eps); normalize so mean weight = 1
        w_raw = 1.0 / np.clip(pv, 1e-6, np.inf)
        w_vec = w_raw / w_raw.mean() if w_raw.mean() > 0 else None

    # Initial guess: constant reporting rate from (weighted) least-squares
    if w_vec is not None:
        denom = float(np.sum(w_vec * phi_vec ** 2))
        r_ls = float(np.clip(np.sum(w_vec * phi_vec * cases_vec) / denom, 1e-6, 1.0 - 1e-6)) if denom > 0 else 0.01
    else:
        denom = float(np.sum(phi_vec ** 2))
        r_ls = float(np.clip(np.sum(phi_vec * cases_vec) / denom, 1e-6, 1.0 - 1e-6)) if denom > 0 else 0.01

    # RW2 penalty matrix
    TT = len(prov_cases)
    D2 = np.diag(TT * [-2.0]) + np.diag((TT - 1) * [1.0], k=1) + np.diag((TT - 1) * [1.0], k=-1)
    if TT >= 3:
        D2[0, 2] = 1.0
        D2[-1, -3] = 1.0

    lam = (D2.T @ D2) * ((3.0 ** 4) / 8.0) * float(np.nan_to_num(prov_cases[cases_col].var(), nan=0.0))
    x0 = np.full(TT, np.log(r_ls / (1.0 - r_ls)))

    result = minimize(
        lambda x: smoothing_cost(x, phi_vec, cases_vec, lam, w_vec),
        x0=x0,
        jac=lambda x: smoothing_grad(x, phi_vec, cases_vec, lam, w_vec),
        method="BFGS",
    )

    theta_hat = np.asarray(result["x"], dtype=float).reshape(-1)
    rr_hat = 1.0 / (1.0 + np.exp(-theta_hat))
    prov_cases["rr"] = rr_hat
    prov_cases["fit"] = rr_hat * phi_vec

    # Posterior variance of rr via delta method on the Hessian
    resid = cases_vec - rr_hat * phi_vec
    if w_vec is not None:
        sig_nu2 = max(float(np.sum(w_vec * resid ** 2) / TT), 1e-12) if TT > 0 else 1.0
    else:
        sig_nu2 = max(float(np.sum(resid ** 2) / TT), 1e-12) if TT > 0 else 1.0
    hess = smoothing_hessian(theta_hat, phi_vec, cases_vec, lam, w_vec) / sig_nu2
    cov = np.linalg.pinv(hess)

    prov_cases["rr_var"] = np.diag(cov) * ((rr_hat * (1.0 - rr_hat)) ** 2)
    prov_cases["rr_std"] = np.sqrt(
        pd.to_numeric(prov_cases["rr_var"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )

    # Store phi_var on prov_cases for diagnostics
    if phi_var is not None:
        prov_cases["phi_var"] = pd.Series(phi_var).reindex(model_years).values
        prov_cases["phi_std"] = np.sqrt(prov_cases["phi_var"].clip(lower=0.0))

    rr_model = {
        "phi": pd.Series(phi_vec, index=prov_cases.index, name="phi"),
        "r_ls": r_ls,
        "result": result,
        "theta_hat": theta_hat,
        "cov": cov,
        "lam": lam,
    }

    return prov_cases, rr_model

# ---------------------------------------------------------------------
# Precomputed event timeline structure
# ---------------------------------------------------------------------

# Event type constants
EVT_MCV1 = 0
EVT_MCV2 = 1
EVT_SIA  = 2

@dataclass
class PrecomputedTimeline:
    """
    Precomputed per-semimonth event timelines and interpolation data.

    Built once from epi_sm + sia_cal. Reused across calls to
    compute_profile_from_semimonth (where only pI and prA_given_I change).
    """
    n_rows: int = 0

    # Per-row data (length n_rows each)
    birth_years: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    cohort_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    birth_weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    mcv1_coverages: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    mcv2_coverages: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    # Event lists: one list per semimonth row.
    # Each event is a tuple: (evt_type, age_years, age_lo_idx, age_hi_idx, alpha, extra)
    #   - extra: dict with sia_name, sia_coverage for SIA events; None otherwise
    event_lists: List[List[Tuple]] = field(default_factory=list)

    # SIA metadata
    sia_names: List[str] = field(default_factory=list)
    all_source_names: List[str] = field(default_factory=list)

    # Age axis info for F interpolation
    n_ages: int = 0
    age_min: int = 0
    age_max: int = 0

    # Cohort axis
    cohort_years_full: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))


def _compute_interp_indices(age_years: float, age_min: int, age_max: int, n_ages: int):
    """
    Compute floor/ceil column indices and interpolation fraction for a
    fractional age, relative to integer age columns [age_min, ..., age_max].
    """
    age_lo = int(np.floor(age_years))
    alpha = age_years - age_lo

    # Map to column indices (columns are age_min .. age_max)
    if age_lo < age_min:
        lo_idx = -1  # sentinel: F = 0 at this age
    elif age_lo > age_max:
        lo_idx = n_ages - 1
    else:
        lo_idx = age_lo - age_min

    age_hi = age_lo + 1
    if age_hi > age_max:
        hi_idx = n_ages - 1 if age_lo <= age_max else -1
    else:
        hi_idx = age_hi - age_min

    return lo_idx, hi_idx, alpha


def build_precomputed_timeline(
    epi_sm: pd.DataFrame,
    sia_cal: pd.DataFrame,
    province: Optional[str],
    cohort_years_full: np.ndarray,
    age_columns: np.ndarray,
) -> PrecomputedTimeline:
    """
    Build a PrecomputedTimeline from semimonthly data and SIA calendar.
    """
    age_min = int(age_columns.min())
    age_max = int(age_columns.max())
    n_ages = len(age_columns)

    cohort_idx_map = {int(b): i for i, b in enumerate(cohort_years_full)}

    # Prepare semimonth data
    sm = epi_sm.copy()
    sm["time"] = pd.to_datetime(sm["time"], errors="coerce")
    sm = sm.dropna(subset=["time"]).copy()
    sm["birth_year"] = sm["time"].dt.year.astype(int)
    sm["births"] = pd.to_numeric(sm["births"], errors="coerce").fillna(0.0)
    sm["mcv1"] = pd.to_numeric(sm.get("mcv1", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    sm["mcv2"] = pd.to_numeric(sm.get("mcv2", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # Precompute SIA metadata
    sia_info = []
    for i, md in sia_cal.iterrows():
        sia_col = campaign_colname(i, md)
        if sia_col not in sm.columns:
            raise KeyError(f"Missing SIA eligibility column in epi_sm: {sia_col}")
        coverage = float(md.get(
            "coverage_pakistan" if province is None else "coverage_targeted_area", 0.0
        ))
        sia_info.append({
            "idx": i, "col": sia_col, "time": pd.to_datetime(md["time"]),
            "coverage": coverage, "name": str(md.get("name", f"SIA_{i}")),
        })

    sia_names = [s["name"] for s in sia_info]
    all_source_names = ["mcv1", "mcv2"] + sia_names

    # Build per-row arrays and event lists
    n_rows_total = len(sm)
    birth_years_arr = np.empty(n_rows_total, dtype=int)
    cohort_indices_arr = np.empty(n_rows_total, dtype=int)
    birth_weights_arr = np.empty(n_rows_total, dtype=float)
    mcv1_cov_arr = np.empty(n_rows_total, dtype=float)
    mcv2_cov_arr = np.empty(n_rows_total, dtype=float)
    event_lists_all = []
    valid_mask = np.ones(n_rows_total, dtype=bool)

    for row_i, (_, row) in enumerate(sm.iterrows()):
        t_birth = row["time"]
        b = int(row["birth_year"])
        w = float(row["births"])

        if b not in cohort_idx_map or w <= 0:
            valid_mask[row_i] = False
            birth_years_arr[row_i] = b
            cohort_indices_arr[row_i] = -1
            birth_weights_arr[row_i] = 0.0
            mcv1_cov_arr[row_i] = 0.0
            mcv2_cov_arr[row_i] = 0.0
            event_lists_all.append([])
            continue

        birth_years_arr[row_i] = b
        cohort_indices_arr[row_i] = cohort_idx_map[b]
        birth_weights_arr[row_i] = w
        mcv1_cov_arr[row_i] = float(row["mcv1"])
        mcv2_cov_arr[row_i] = float(row["mcv2"])

        # Build chronological event list for this semimonth
        raw_events = []

        # MCV1 at t_birth + 9 months
        mcv1_time = t_birth + pd.DateOffset(months=9)
        lo, hi, al = _compute_interp_indices(9.0 / 12.0, age_min, age_max, n_ages)
        raw_events.append((mcv1_time, EVT_MCV1, 9.0 / 12.0, lo, hi, al, None))

        # MCV2 at t_birth + 15 months
        mcv2_time = t_birth + pd.DateOffset(months=15)
        lo, hi, al = _compute_interp_indices(15.0 / 12.0, age_min, age_max, n_ages)
        raw_events.append((mcv2_time, EVT_MCV2, 15.0 / 12.0, lo, hi, al, None))

        # SIAs: include only if this semimonth's cohort was eligible
        for si in sia_info:
            elig_val = pd.to_numeric(row.get(si["col"], 0), errors="coerce")
            if pd.isna(elig_val) or int(elig_val) != 1:
                continue
            age_at_campaign = (si["time"] - t_birth).days / 365.25
            if age_at_campaign < 0:
                continue
            lo, hi, al = _compute_interp_indices(age_at_campaign, age_min, age_max, n_ages)
            extra = {"sia_name": si["name"], "sia_coverage": si["coverage"]}
            raw_events.append((si["time"], EVT_SIA, age_at_campaign, lo, hi, al, extra))

        # Sort by time, strip the time key
        raw_events.sort(key=lambda e: e[0])
        event_lists_all.append([
            (evt[1], evt[2], evt[3], evt[4], evt[5], evt[6])
            for evt in raw_events
        ])

    # Filter to valid rows
    keep = valid_mask
    return PrecomputedTimeline(
        n_rows=int(keep.sum()),
        birth_years=birth_years_arr[keep],
        cohort_indices=cohort_indices_arr[keep],
        birth_weights=birth_weights_arr[keep],
        mcv1_coverages=mcv1_cov_arr[keep],
        mcv2_coverages=mcv2_cov_arr[keep],
        event_lists=[event_lists_all[i] for i in range(n_rows_total) if keep[i]],
        sia_names=sia_names,
        all_source_names=all_source_names,
        n_ages=n_ages,
        age_min=age_min,
        age_max=age_max,
        cohort_years_full=cohort_years_full,
    )


# ---------------------------------------------------------------------
# Interpolation using precomputed indices
# ---------------------------------------------------------------------

def _interp_F_fast(
    pI_val: float,
    cum_prA: np.ndarray,
    row_idx: int,
    lo_idx: int,
    hi_idx: int,
    alpha: float,
) -> float:
    """
    Cumulative infection lookup using precomputed indices.

    Returns pI * F(age), where F is linearly interpolated from the
    cumulative prA_given_I array.
    """
    f_lo = 0.0 if lo_idx < 0 else cum_prA[row_idx, lo_idx]
    f_hi = (0.0 if hi_idx < 0
            else f_lo if hi_idx == lo_idx
            else cum_prA[row_idx, hi_idx])
    return pI_val * (f_lo + alpha * (f_hi - f_lo))


# ---------------------------------------------------------------------
# Immunity profile computation (using precomputed timeline)
# ---------------------------------------------------------------------

def compute_profile_from_semimonth(
    timeline: PrecomputedTimeline,
    pI,
    prA_given_I: pd.DataFrame,
    cohort_years_full: np.ndarray,
    mcv1_effic: float,
    mcv2_effic: float,
    sia_vax_effic: float,
) -> pd.DataFrame:
    """
    Build cohort-indexed immunity profile with unified chronological processing,
    using PrecomputedTimeline.

    Walks through each semimonth's event list in order, accumulating
    vaccine-derived immunity from MCV1, MCV2, and SIAs while accounting
    for prior infection (via pI * cumulative prA_given_I).

    The PrecomputedTimeline contains all event lists, ages, interpolation indices,
    birth weights, and coverages. Only pI and prA_given_I change between calls.
    """
    n_cohorts = len(cohort_years_full)
    all_source_names = timeline.all_source_names

    # Build cumulative infection array 
    prA = prA_given_I.reindex(cohort_years_full).bfill()
    cum_prA = np.cumsum(prA.values, axis=1)  

    if np.isscalar(pI):
        pI_arr = np.full(n_cohorts, float(pI), dtype=float)
    else:
        pI_arr = pd.Series(pI).reindex(cohort_years_full).astype(float).bfill().fillna(0.0).values

    # Accumulators (birth-weighted sums per cohort)
    imm_accum = {src: np.zeros(n_cohorts) for src in all_source_names}
    births_accum = np.zeros(n_cohorts)

    # Process each semimonth row 
    for row_i in range(timeline.n_rows):
        bi = timeline.cohort_indices[row_i]
        w = timeline.birth_weights[row_i]
        mcv1_cov = timeline.mcv1_coverages[row_i]
        mcv2_cov = timeline.mcv2_coverages[row_i]
        pI_val = pI_arr[bi]

        cum_vax_imm = 0.0
        source_imm = {}

        for (evt_type, age_years, lo_idx, hi_idx, alpha, extra) in timeline.event_lists[row_i]:

            # Fraction already immune via infection at this age
            f_inf = min(max(
                _interp_F_fast(pI_val, cum_prA, bi, lo_idx, hi_idx, alpha),
            0.0), 1.0)
            susceptible = max(1.0 - cum_vax_imm - f_inf, 0.0)

            if evt_type == EVT_MCV1:
                new_imm = max(mcv1_effic * mcv1_cov * susceptible, 0.0)
                source_imm["mcv1"] = source_imm.get("mcv1", 0.0) + new_imm
                cum_vax_imm += new_imm

            elif evt_type == EVT_MCV2:
                # MCV2 targets those who received MCV1 but weren't immunized
                sus_given_mcv2 = (1.0 - mcv1_effic) * (1.0 - f_inf)
                sia_imm_so_far = cum_vax_imm - source_imm.get("mcv1", 0.0)
                sus_given_mcv2 = max(sus_given_mcv2 - sia_imm_so_far, 0.0)
                new_imm = max(mcv2_effic * mcv2_cov * sus_given_mcv2, 0.0)
                source_imm["mcv2"] = source_imm.get("mcv2", 0.0) + new_imm
                cum_vax_imm += new_imm

            elif evt_type == EVT_SIA:
                new_imm = max(sia_vax_effic * extra["sia_coverage"] * susceptible, 0.0)
                sn = extra["sia_name"]
                source_imm[sn] = source_imm.get(sn, 0.0) + new_imm
                cum_vax_imm += new_imm

        births_accum[bi] += w
        for src in all_source_names:
            imm_accum[src][bi] += w * source_imm.get(src, 0.0)

    # Normalize by total births per cohort
    births_total = np.maximum(births_accum, 1e-12)
    imm_profile = pd.DataFrame(
        {src: imm_accum[src] / births_total for src in all_source_names},
        index=cohort_years_full,
    )
    imm_profile.index.name = "birth_year"
    return imm_profile.clip(0.0, 1.0)


# ---------------------------------------------------------------------
# Yearly infection timing machinery
# ---------------------------------------------------------------------

def build_prA_and_I(
    prA_given_I: pd.DataFrame,
    pr_inf2: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the joint cohort × age infection pmf: P_b(I) * P_b(a|I).
    """
    cohorts = pr_inf2.index.values.astype(int)
    ages = prA_given_I.columns.values.astype(int)
    prA_mat = prA_given_I.reindex(cohorts).bfill().to_numpy(dtype=float)
    return pr_inf2.values[:, None] * prA_mat, cohorts, ages


def build_PST_from_prA_and_I(prA_and_I: np.ndarray, cohorts: np.ndarray, ages: np.ndarray) -> pd.DataFrame:
    """
    Build the population-scaled transmission (PST) matrix: rows = birth cohort,
    columns = calendar year, values = P_b(I) * P_b(a|I) placed at year = b + a.
    """
    n_cohorts = len(cohorts)
    n_ages = len(ages)
    T = int(cohorts[-1] + ages[-1] - cohorts[0] + 1)
    cols = np.arange(cohorts[0], cohorts[0] + T, dtype=np.int32)

    PST = np.zeros((n_cohorts, T), dtype=float)
    for i in range(n_cohorts):
        PST[i, i:i + n_ages] = prA_and_I[i]

    PST = pd.DataFrame(PST, index=cohorts, columns=cols)
    PST.index.name = "birth_year"
    return PST


# ---------------------------------------------------------------------
# Callable "region run" core
# ---------------------------------------------------------------------

@dataclass
class SurvivalPriorParams:
    """Parameters for the survival-prior immunity model."""
    mcv1_effic: float = 0.825
    mcv2_effic: float = 0.95
    sia_vax_effic: float = 0.8

    model_start_year: int = 2012
    pI_right: float = 0.1


def run_region_survival_prior(
    *,
    province: Optional[str],
    epi_sm: pd.DataFrame,
    births_y: pd.DataFrame,
    sia_cal: pd.DataFrame,
    prA_given_I: pd.DataFrame,
    cohort_years_full: np.ndarray,
    model_years: np.ndarray,
    params: SurvivalPriorParams = SurvivalPriorParams(),
    return_rr: bool = False,
    rr_cases_col: str = "cases",
    precomputed_timeline: Optional[PrecomputedTimeline] = None,
) -> Dict[str, Any]:
    """
    Run the SurvivalPrior "core" for one region and return key intermediate objects.

    Steps:
      1. Two-point interpolation to solve for P_b(I) per cohort.
      2. Final immunity profile.
      3. Joint cohort-age infection pmf → PST → expected infections phi(y).
      4. (Optional) Reporting-rate regression: cases(y) ≈ r(y) * phi(y).

    Accepts epi_sm and births_y directly (from get_epi_semimonth_and_births)
    to avoid redundant recomputation.

    Accepts an optional precomputed timeline to avoid rebuilding event lists
    on repeated calls. Build it once via build_precomputed_timeline().
    """
    # Align prA_given_I
    prA_given_I = prA_given_I.reindex(cohort_years_full).copy()
    prA_given_I.columns = prA_given_I.columns.astype(int)

    # Build or reuse precomputed timeline
    if precomputed_timeline is None:
        precomputed_timeline = build_precomputed_timeline(
            epi_sm=epi_sm,
            sia_cal=sia_cal,
            province=province,
            cohort_years_full=cohort_years_full,
            age_columns=prA_given_I.columns.values.astype(int),
        )

    # Shared kwargs for compute_profile_from_semimonth
    profile_kwargs = dict(
        timeline=precomputed_timeline,
        prA_given_I=prA_given_I,
        cohort_years_full=cohort_years_full,
        mcv1_effic=params.mcv1_effic,
        mcv2_effic=params.mcv2_effic,
        sia_vax_effic=params.sia_vax_effic,
    )

    # Solve for pr_inf via two-point interpolation
    # Evaluate immunity at pI=0 (left) and pI=pI_right (right), then
    # solve f(pI) = V(pI) + pI - 1 = 0.
    left_profile = compute_profile_from_semimonth(pI=0.0, **profile_kwargs)
    right_profile = compute_profile_from_semimonth(pI=float(params.pI_right), **profile_kwargs)

    fL = -1.0 + left_profile.sum(axis=1)
    fR = float(params.pI_right) - 1.0 + right_profile.sum(axis=1)
    pr_inf = (-fL * float(params.pI_right)) / (fR - fL)
    pr_inf = pr_inf.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)

    # Final immunity profile
    imm_profile = compute_profile_from_semimonth(pI=pr_inf, **profile_kwargs)

    # Residual infection probability (1 - total vaccine immunity)
    pr_inf2 = (1.0 - imm_profile.sum(axis=1)).clip(0.0, 1.0)

    # Joint cohort-age infection pmf
    prA_and_I, cohorts, ages = build_prA_and_I(prA_given_I, pr_inf2)

    # PST and expected infections by calendar year
    PST = build_PST_from_prA_and_I(prA_and_I, cohorts, ages)
    births_by_cohort = births_y["births"].reindex(PST.index).bfill().fillna(0.0).astype(float)
    expI = births_by_cohort.values[:, None] * PST
    phi = expI.sum(axis=0)

    # Propagate birth-count uncertainty through expected infections:
    # Var(phi(y)) = sum_b Var(births_b) * PST_b(y)^2
    births_var_by_cohort = births_y["births_var"].reindex(PST.index).bfill().fillna(0.0).astype(float)
    phi_var = (births_var_by_cohort.values[:, None] * (PST.values ** 2)).sum(axis=0)
    phi_var = pd.Series(phi_var, index=PST.columns, name="phi_var")

    # Reporting-rate regression (if requested)
    prov_cases = None
    rr_model = None
    if return_rr:
        prov_cases, rr_model = fit_reporting_rate_rw2(
            epi_sm=epi_sm,
            phi_year=phi,
            model_years=model_years,
            model_start_year=params.model_start_year,
            cases_col=rr_cases_col,
            phi_var=phi_var,
        )

    return {
        "province": province,
        "imm_profile": imm_profile,
        "pr_inf2": pr_inf2,
        "prA_and_I": prA_and_I,
        "cohorts": cohorts,
        "ages": ages,
        "phi": phi,
        "phi_var": phi_var,
        "prov_cases": prov_cases,
        "rr_model": rr_model,
    }