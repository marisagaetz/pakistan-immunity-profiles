""" BirthSeasonality.py

Province-level estimates of birth seasonality over time using Fourier 
harmonics for month (periodic) and B-spline basis for year.  Uncertainty 
via Laplace approximation (draws from the GLM coefficient covariance).

Dependencies: survey_io

Outputs:
  - ../../_plots/birth_seasonality_by_province.pdf
  - ../../pickle_jar/birth_seasonality_by_province.pkl

Surveys used (current):
  - DHS BR (2006, 2012, 2017)
  - MICS BH (Punjab 2017, Sindh 2018, Balochistan 2019, KP 2019)

Notes:
  - Output index is MultiIndex: (province, year, month) with columns ['avg','var'].
"""

# For filepaths
import os

# I/O
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Smoothing
import statsmodels.api as sm
import patsy

# Survey I/O
from demography.survey_io import load_survey, infer_survey_folder

# -------------------------------------------------------------
# Config
# -------------------------------------------------------------

COUNTRY = "pakistan"

MIN_YEAR = 2005
MAX_YEAR = 2016

# Output paths
OUT_PDF = "../../_plots/birth_seasonality_by_province.pdf"
OUT_PKL = "../../pickle_jar/birth_seasonality_by_province.pkl"

# Smoother params
N_HARMONICS = 2       # Fourier harmonics for month (higher = less smooth)
YEAR_SPLINE_DF = 5    # B-spline df for year (higher = more flexible year trend)

# Number of coefficient draws for uncertainty bands
N_DRAWS = 1000        

BASE = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SURVEYS = os.path.join(BASE, "_data", "_surveys")

# -------------------------------------------------------------
# Load surveys
# -------------------------------------------------------------

# DHS births recode files (birth history)
dhs_br_paths = [
    os.path.join(SURVEYS, "DHS5_2006", "PKBR52DT", "pkbr52fl.dta"),
    os.path.join(SURVEYS, "DHS6_2012", "PKBR61DT", "PKBR61FL.DTA"),
    os.path.join(SURVEYS, "DHS7_2017", "PKBR71DT", "PKBR71FL.DTA"),
]

br_columns = ["child_DoB", "mom_DoB", "province"]

brs = {
    infer_survey_folder(path): load_survey(path, True, True, br_columns)
    for path in dhs_br_paths
}

# MICS births history files (BH)
mics_bh_paths = [
    os.path.join(SURVEYS, "MICS6_Balochistan_2019", "bh.sav"),
    os.path.join(SURVEYS, "MICS6_KP_2019", "bh.sav"),
    os.path.join(SURVEYS, "MICS6_Punjab_2017", "bh.sav"),
    os.path.join(SURVEYS, "MICS6_Sindh_2018", "bh.sav"),
]

bh_columns = ["child_birth_mon", "child_birth_year", "province"]

bhs = {
    infer_survey_folder(path): load_survey(path, True, True, bh_columns)
    for path in mics_bh_paths
}

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def axes_setup(ax):
    ax.spines["left"].set_position(("axes", -0.025))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def cms_to_datetime(a, d=15):
    years = 1900 + ((a - 1) // 12)
    months = (a - 1) % 12 + 1
    return pd.to_datetime({"year": years, "month": months, "day": d})

def softmax_rows(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def _build_design_matrix(years, months, n_harmonics=N_HARMONICS, year_df=YEAR_SPLINE_DF):
    """
    Build a design matrix with:
      - B-spline basis over year (smooth year-to-year trend)
      - Fourier pairs over month (periodic month-to-month shape)
      - Interactions between the two (lets the seasonal shape evolve over years)
    """
    data = pd.DataFrame({"year": years, "month": months})

    # Build the Fourier terms as explicit columns so patsy can see them.
    for h in range(1, n_harmonics + 1):
        data[f"sin{h}"] = np.sin(2 * np.pi * h * data["month"] / 12)
        data[f"cos{h}"] = np.cos(2 * np.pi * h * data["month"] / 12)

    # Formula: B-spline year trend + Fourier month terms
    # + interactions (year spline × Fourier) so the seasonal
    # profile can shift over time.
    fourier_terms = " + ".join(
        [f"sin{h} + cos{h}" for h in range(1, n_harmonics + 1)]
    )

    if year_df > 0 and len(np.unique(years)) > year_df:
        # Full model with year splines and interactions
        formula = (
            f"bs(year, df={year_df}, degree=3) + {fourier_terms}"
            f" + bs(year, df={year_df}, degree=3):({fourier_terms})"
        )
    else:
        # Not enough years for splines; just use Fourier + linear year
        formula = f"year + {fourier_terms} + year:({fourier_terms})"

    X = patsy.dmatrix(formula, data, return_type="dataframe")
    return X


def smooth_birth_seasonality_glm(counts_ym, n_harmonics=N_HARMONICS, year_df=YEAR_SPLINE_DF):
    """
    Fit per-year month probabilities using a statsmodels Binomial GLM
    with Fourier month harmonics and B-spline year terms.
    """
    C = np.asarray(counts_ym, dtype=float)
    T, M = C.shape
    assert M == 12

    n = C.sum(axis=1)
    mask = n > 0

    if mask.sum() == 0:
        return np.ones((T, 12)) / 12.0, None

    # Flatten to long form: one row per (year_index, month)
    year_indices = np.arange(T)
    years_long = np.repeat(year_indices, 12)
    months_long = np.tile(np.arange(1, 13), T)
    counts_long = C.reshape(-1)
    totals_long = np.repeat(n, 12)

    # Build the design matrix
    X = _build_design_matrix(years_long, months_long, n_harmonics, year_df)

    # Fit a binomial GLM: count / total ~ X
    # Using the logit link (default for Binomial).
    endog = np.column_stack([counts_long, totals_long - counts_long])
    glm = sm.GLM(endog, X, family=sm.families.Binomial())
    glm_result = glm.fit()

    # Predict fitted probabilities on the same grid
    mu = glm_result.predict(X)  # P(month | year)

    # Reshape to (T, 12) and renormalise to sum to 1 per year
    # (the GLM fits each month independently, so we softmax / normalise)
    P_hat = mu.values.reshape((T, 12)) if hasattr(mu, 'values') else mu.reshape((T, 12))
    row_sums = P_hat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P_hat = P_hat / row_sums

    # Zero-count years get uniform
    P_hat[~mask, :] = 1.0 / 12.0

    return P_hat, glm_result


def seasonality_laplace_uncertainty(
    counts_ym,
    n_harmonics=N_HARMONICS,
    year_df=YEAR_SPLINE_DF,
    n_draws=N_DRAWS,
    seed=0,
):
    """
    Uncertainty via Laplace approximation on the GLM.
    """
    C = np.asarray(counts_ym, dtype=float)
    T, M = C.shape
    assert M == 12

    n = C.sum(axis=1)
    mask = n > 0

    # Fit the GLM
    P_hat, glm_result = smooth_birth_seasonality_glm(C, n_harmonics, year_df)

    # Extract coefficient estimate and covariance
    beta_hat = np.asarray(glm_result.params)
    beta_cov = np.asarray(glm_result.cov_params())

    # The design matrix (same grid used for fitting)
    year_indices = np.arange(T)
    years_long = np.repeat(year_indices, 12)
    months_long = np.tile(np.arange(1, 13), T)
    X = _build_design_matrix(years_long, months_long, n_harmonics, year_df)
    X_arr = np.asarray(X)

    # Draw coefficient vectors from the Gaussian approximation
    rng = np.random.default_rng(seed)
    beta_draws = rng.multivariate_normal(beta_hat, beta_cov, size=n_draws)

    # Push each draw through the logit link → probabilities → (T, 12)
    draws = np.zeros((n_draws, T, M), dtype=float)
    for i, beta in enumerate(beta_draws):
        eta = X_arr @ beta                     # linear predictor
        mu = 1.0 / (1.0 + np.exp(-eta))       # inverse logit
        P = mu.reshape((T, 12))
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P = P / row_sums
        P[~mask, :] = 1.0 / 12.0
        draws[i, :, :] = P

    low = np.percentile(draws, 2.5, axis=0)
    high = np.percentile(draws, 97.5, axis=0)
    mid = draws.mean(axis=0)
    var = draws.var(axis=0)
    return mid, var, low, high


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

if __name__ == "__main__":

    dhs = pd.concat(brs.values(), axis=0).reset_index(drop=True)
    mics = pd.concat(bhs.values(), axis=0).reset_index(drop=True)

    # DHS: birth_date from CMC
    dhs["birth_date"] = cms_to_datetime(dhs["child_DoB"])
    dhs["birth_year"] = dhs["birth_date"].dt.year.astype(int)
    dhs["birth_month"] = dhs["birth_date"].dt.month.astype(int)

    # MICS: birth_date from month/year
    mics = mics.loc[(~mics["child_birth_year"].isna()) & (~mics["child_birth_mon"].isna())]
    mics["birth_year"] = mics["child_birth_year"].astype(int)
    mics["birth_month"] = mics["child_birth_mon"].astype(int)
    mics["birth_date"] = pd.to_datetime(
        {"month": mics["birth_month"], "year": mics["birth_year"], "day": 1}
    )

    # Restrict window
    dhs = dhs.loc[
        (dhs["birth_year"] >= MIN_YEAR) & (dhs["birth_year"] <= MAX_YEAR)
    ].reset_index(drop=True)
    mics = mics.loc[
        (mics["birth_year"] >= MIN_YEAR) & (mics["birth_year"] <= MAX_YEAR)
    ].reset_index(drop=True)

    # Combine
    columns = ["survey", "province", "birth_date", "birth_year", "birth_month"]
    df = pd.concat([dhs[columns], mics[columns]], axis=0).reset_index(drop=True)

    print("\nFull dataset:")
    print(df.head())
    print(df.tail())

    output = {}
    with PdfPages(OUT_PDF) as pdf:

        print("\nStarting the loop through provinces...")
        for prov, sf in df.groupby("province"):

            # counts: years x months
            monthly = sf.groupby(["birth_year", "birth_month"]).size().rename("count")
            monthly = monthly.unstack(level=1).fillna(0).sort_index()

            for m in range(1, 13):
                if m not in monthly.columns:
                    monthly[m] = 0
            monthly = monthly[sorted(monthly.columns)]  # 1..12

            counts_ym = monthly.values  # (T, 12)

            mid, var, low, high = seasonality_laplace_uncertainty(
                counts_ym,
                n_harmonics=N_HARMONICS,
                year_df=YEAR_SPLINE_DF,
                n_draws=N_DRAWS,
                seed=abs(hash(prov)) % (2**32),
            )

            # Output frames (province, year, month) -> avg/var
            mid_df = (
                pd.DataFrame(mid, columns=monthly.columns, index=monthly.index)
                .stack()
                .rename("avg")
            )
            var_df = (
                pd.DataFrame(var, columns=monthly.columns, index=monthly.index)
                .stack()
                .rename("var")
            )
            output[prov] = pd.concat([mid_df, var_df], axis=1)

            # Empirical fractions for plotting
            total_by_year = monthly.sum(axis=1).replace(0, np.nan)
            monthly_frac = monthly.div(total_by_year, axis=0).fillna(0.0)

            num_to_month = {
                1: "January", 2: "February", 3: "March", 4: "April",
                5: "May", 6: "June", 7: "July", 8: "August",
                9: "September", 10: "October", 11: "November", 12: "December",
            }

            fig, axes = plt.subplots(3, 4, sharex=True, sharey=True, figsize=(16, 9))
            axes = axes.reshape(-1)

            for ax in axes:
                axes_setup(ax)
                ax.grid(color="grey", alpha=0.2)

            for i in monthly_frac.columns:
                axes[i - 1].fill_between(
                    monthly_frac.index,
                    low[:, i - 1],
                    high[:, i - 1],
                    facecolor="grey",
                    edgecolor="None",
                    alpha=0.35,
                    label="Model" if i == 1 else None,
                )
                axes[i - 1].plot(
                    monthly_frac.index, mid[:, i - 1], lw=2, color="grey"
                )

                axes[i - 1].plot(
                    monthly_frac[i],
                    ls="dashed",
                    lw=3,
                    color="k",
                    markersize=8,
                    marker="o",
                    label="Survey" if i == 4 else None,
                )

                axes[i - 1].text(
                    0.01, 0.99, num_to_month[i],
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=18,
                    color="xkcd:red wine",
                    transform=axes[i - 1].transAxes,
                )

                if (i - 1) % 4 == 0:
                    axes[i - 1].set_ylabel("Probability")
                if i == 4:
                    axes[i - 1].legend(frameon=False, loc=1, fontsize=14)

            fig.suptitle("Seasonality in " + str(prov).title())
            fig.tight_layout(rect=[0, 0.0, 1, 0.92])
            pdf.savefig(fig)
            plt.close(fig)

        d = pdf.infodict()
        d["Title"] = "Birth seasonality in Pakistan"

    print("...finished.")

    output = pd.concat(output.values(), keys=output.keys())
    output.index = output.index.set_names(["province", "year", "month"])

    print("\nFinal output:")
    print(output.head())
    print(output.tail())

    output.to_pickle(OUT_PKL)
    print(f"\nSaved seasonality pickle to: {OUT_PKL}")