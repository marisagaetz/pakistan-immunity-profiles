""" MomDistribution.py

Uses DHS weights to coarsely estimate the fraction of individuals in each
demographic cell (i.e., mom-characteristic combination). 

Dependancies: survey_io
Outputs:
    - ../pickle_jar/mom_distribution_main.pkl
    - ../pickle_jar/mom_distribution_ajk_gb.pkl
    
Note: If adapting this code to a country other than Pakistan, the special
treatment of AJK, GB, and ICT (including the separate dataframe, 
reindex_prov_years, and subtract_ict_from_punjab) can be removed. This
treatment accounts for special DHS weights assigned to AJK and GB that
require different normalization, as well as discrepancies across DHS
surveys regarding whether ICT is included as part of Punjab.

Debugging tips:
    - If you encounter issues with data types, missing data, or if
        you want to add covariates, check the survey_io file, which
        sets up the DHS data.
    - Inspect the output(s) for NaN weight values, which should only
        occur in years prior to available data for that province, or 
        for uncommon age bins (e.g., 15-19).
    - Check the printed normalizations; sum of weights should equal 1.0
        across all provinces each year in the main distribution.
"""

# For filepaths
import os

# Input/output functionality is built on top of pandas
import pandas as pd
import numpy as np

# For loading, renaming, and unifying DHS and MICS data
from demography.survey_io import load_survey, infer_survey_folder

# --------------------------------------------------------------------
# Config
# --------------------------------------------------------------------

# YEAR_MAX should be the latest year with survey data
YEAR_MAX = 2018

# Only relevant for Pakistan
ICT_SEED_YEAR = 2012

# Window (in years) for rolling mean to smooth interpolated weights
SMOOTH_WEIGTHS_WINDOW = 2

BASE = os.path.dirname(os.path.dirname(__file__))
SURVEYS = os.path.join(BASE, "_data", "_surveys")

# --------------------------------------------------------------------
# Load DHS files
# --------------------------------------------------------------------

dhs_ir_paths = [
    os.path.join(SURVEYS,"DHS5_2006","PKIR52DT","pkir52fl.dta"),
    os.path.join(SURVEYS,"DHS6_2012","PKIR61DT","PKIR61FL.DTA"),
    os.path.join(SURVEYS,"DHS7_2017","PKIR71DT","PKIR71FL.DTA"),
]

ir_columns = ["mom_DoB", "interview_date", "mom_age", "province",
              "area", "mom_edu", "num_brs", "weight", "year"]

irs = {
    infer_survey_folder(path): load_survey(path, True, True, ir_columns)
    for path in dhs_ir_paths
}

irs = pd.concat(irs,axis=0).reset_index(drop=True)
    
# --------------------------------------------------------------------
# Pull AJK and GB into a separate dataframe, since they are omitted or
# given special weights in some of the DHS dataframes
# --------------------------------------------------------------------

# Separate AJK and GB and normalize their weights by province and survey
special_provs = ["ajk", "gb"]
irs_main = irs.loc[~irs["province"].isin(special_provs)].copy()
irs_ajk_gb = irs.loc[irs["province"].isin(special_provs)].copy()

# Normalize weights
group_sums = irs_ajk_gb.groupby(["survey", "province"])["weight"].transform("sum")
irs_ajk_gb["weight"] = irs_ajk_gb["weight"] / group_sums

# Normalize sweights
s_group_sums = irs_ajk_gb.groupby(["survey", "province"])["sweight"].transform("sum")
irs_ajk_gb["sweight"] = irs_ajk_gb["sweight"] / s_group_sums

# After normalizing, combine "weight" and "sweight" into a single "weight" column
irs_ajk_gb["weight"] = irs_ajk_gb["weight"].where(
    irs_ajk_gb["weight"].notna() & (irs_ajk_gb["weight"] != 0),
    irs_ajk_gb["sweight"]
)

# Renormalize irs_main by survey now that AJK and GB are omitted
irs_main["weight"] = irs_main["weight"] / irs_main.groupby(["survey"])["weight"].transform("sum")    

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------

def normalize_within_survey(df,provinces=None):
    """
    Average weights across years included in the same survey
    to account for surveys primarily done in one year but 
    spilling over to a second.
    """
    if provinces is not None:
        df = df.loc[df["province"].isin(provinces)].copy()
    
    # Compute total weight per (survey, province, year)
    totals = (
        df.groupby(["survey", "province", "year"])["weight"]
          .sum()
          .reset_index()
          .rename(columns={"weight": "year_sum"})
    )
    
    # Compute average of the year sums for each (survey, province)
    avg_by_group = (
        totals.groupby(["survey", "province"])["year_sum"]
              .mean()
              .reset_index()
              .rename(columns={"year_sum": "avg_sum"})
    )
    
    # Merge back onto totals
    totals = totals.merge(avg_by_group, on=["survey", "province"], how="left")
    
    # Compute scaling factor
    totals["scale_factor"] = totals["avg_sum"] / totals["year_sum"]
    
    # Merge scale factors back onto original dataframe
    df = df.merge(
        totals[["survey", "province", "year", "scale_factor"]],
        on=["survey", "province", "year"],
        how="left"
    )
    
    # Scale the weights
    df["weight"] = df["weight"] * df["scale_factor"]
    
    return df

def interp_inside_support(g):
    """
    For one (province, area, mom_edu, mom_age) group across years:
      - treat zeros as missing
      - interpolate linearly between first and last observed year
      - forward-fill after last observed year (through YEAR_MAX grid)
      - do not backfill before first observed year
    """
    g = g.sort_values("year").copy()
    g["weight"] = g["weight"].replace(0, np.nan)

    s = g["weight"]

    # Identify observed support (in terms of years)
    obs = g.loc[s.notna(), "year"]
    if obs.empty:
        g["weight"] = s
        return g

    y_first = int(obs.min())
    y_last  = int(obs.max())

    # Interpolate only within [y_first, y_last]
    inside = (g["year"] >= y_first) & (g["year"] <= y_last)
    s.loc[inside] = s.loc[inside].interpolate(limit_area="inside")

    # Forward-fill after last observed year
    after = g["year"] > y_last
    if after.any():
        # value at y_last (after interpolation) is the anchor
        anchor = s.loc[g["year"] == y_last]
        if not anchor.empty and pd.notna(anchor.iloc[0]):
            s.loc[after] = anchor.iloc[0]

    # Before first observed year remains NaN
    g["weight"] = s
    return g

def smooth_weights(g, window=SMOOTH_WEIGTHS_WINDOW):
    """
    For one (province, area, mom_edu, mom_age) group across years:
      - apply a centered rolling mean to non-NaN weights
      - leaves NaN values (years before first observation) untouched
    """
    g = g.sort_values("year").copy()
    s = g["weight"]
    obs = s.notna()
    if obs.sum() < 2:
        return g
    s.loc[obs] = s.loc[obs].rolling(window, center=True, min_periods=1).mean()
    g["weight"] = s
    return g

def reindex_prov_years(g, prov_year_bounds):
    """
    For one (province, area, mom_edu, mom_age) group:
      - expand to a complete year grid from the province's first 
        observed year through YEAR_MAX
      - missing years get NaN weights (to be filled by 
        interp_inside_support)
    """
    province = g["province"].iloc[0]
    area = g["area"].iloc[0]
    edu = g["mom_edu"].iloc[0]
    age = g["mom_age"].iloc[0]

    y0 = int(prov_year_bounds[province]["year_min"])
    y1 = int(YEAR_MAX)

    years = pd.Index(np.arange(y0, y1 + 1, dtype="int64"), name="year")

    # Reindex on year only 
    s = (
        g[["year", "weight"]]
        .assign(year=pd.to_numeric(g["year"], errors="coerce").astype("int64"))
        .set_index("year")
        .sort_index()
    )
    s = s.reindex(years)

    # Put keys back as columns
    s = s.reset_index()
    s["province"] = province
    s["area"] = area
    s["mom_edu"] = edu
    s["mom_age"] = age
    return s[["province", "area", "mom_edu", "mom_age", "year", "weight"]]

def subtract_ict_from_punjab(
    df,
    years=(2006, 2007),
    seed_year=ICT_SEED_YEAR,
    clip_lower=0.0,
):
    """
    DHS 2006/2007: ICT is lumped into Punjab.
    This function:
      1) Seeds ICT weights for 'years' from ICT 'seed_year' (cell-by-cell),
      2) Subtracts those ICT weights from Punjab in the same years (cell-by-cell),
      3) Optionally clips Punjab at 'clip_lower' (default 0.0).

    Assumes df is one row per (province, area, mom_edu, mom_age, year) and
    that ICT already has rows for 'years' (e.g., via reindex_prov_years after
    forcing ICT year_min = 2006).
    """
    df = df.copy()

    # Ensure numeric weights; treat 0 as missing if you want the same semantics
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

    group_cols = ["area", "mom_edu", "mom_age"]
    key_cols = group_cols + ["year"]

    # Build a lookup: ICT weights at seed_year for each demographic cell
    ict_seed = (
        df[(df["province"] == "ict") & (df["year"] == seed_year)]
        .set_index(group_cols)["weight"]
    )

    # Seed ICT for the target years from the seed_year, cell-by-cell
    for y in years:
        mask_ict_y = (df["province"] == "ict") & (df["year"] == y)
        if mask_ict_y.any():
            # Fill (or overwrite) ICT weights in year y using seed mapping
            df.loc[mask_ict_y, "weight"] = df.loc[mask_ict_y, group_cols].apply(
                lambda r: float(ict_seed.get(tuple(r.values), np.nan)),
                axis=1
            )

    # Compute ICT weights in target years (after seeding) as a Series keyed by 
    # (area, edu, age, year) 
    ict_weights = (
        df[(df["province"] == "ict") & (df["year"].isin(years))]
        .set_index(key_cols)["weight"]
    )

    # Subtract from Punjab in those years, aligned by cell
    mask_punjab = (df["province"] == "punjab") & (df["year"].isin(years))
    if mask_punjab.any():
        keys = df.loc[mask_punjab, key_cols].apply(tuple, axis=1)
        subtr = keys.map(lambda k: float(ict_weights.get(k, 0.0)))
        df.loc[mask_punjab, "weight"] = df.loc[mask_punjab, "weight"].values - subtr.values

        if clip_lower is not None:
            df.loc[mask_punjab, "weight"] = df.loc[mask_punjab, "weight"].clip(lower=clip_lower)

    return df

# --------------------------------------------------------------------
# Main script
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Normalize weights within surveys
    irs_main = normalize_within_survey(irs_main)
    irs_ajk_gb = normalize_within_survey(irs_ajk_gb) 

    for label, irs in [("main", irs_main), ("ajk_gb", irs_ajk_gb)]:
        # Print the dataset and summarize values
        print("\nFull dataset:")
        print(irs)

        print("\nVariable values...")
        for c in ["mom_age","province","area","mom_edu","year"]:
            values = sorted(irs[c].dropna().unique())
            print("{} values = {}".format(c,values))
        
        # Create a table of demographic cells
        irs = irs.copy()
        irs["weight"] = irs["weight"].replace(0, np.nan)
        irs["year"] = pd.to_numeric(irs["year"], errors="coerce").astype("int64")
        
        prov_year_bounds = (
            irs.groupby("province")["year"]
               .agg(year_min="min", year_max="max")
               .to_dict(orient="index")
        )
        
        # Force ICT to have a 2006 start so it gets a 2006..YEAR_MAX grid
        if label == "main":        
            prov_year_bounds["ict"]["year_min"] = 2006

        df = irs[["province", "area", "mom_edu", "mom_age", "weight", "year"]].copy()
        df = (df.groupby(["province", "area", "mom_edu", "mom_age", "year"], 
                       as_index=False)["weight"].sum())
        df = (df.groupby(["province", "area", "mom_edu", "mom_age"], group_keys=False)
              .apply(lambda g: reindex_prov_years(g, prov_year_bounds))).reset_index(drop=True)
        
        if label == "main":
            df = subtract_ict_from_punjab(df, years=(2006, 2007), 
                    seed_year=ICT_SEED_YEAR, clip_lower=0.0)

        df = df.reset_index(drop=True)

        # Interpolate weights and renormalize by year
        df = df.groupby(["province", "area", "mom_edu", "mom_age"], group_keys=False).apply(interp_inside_support)
        df = df.groupby(["province", "area", "mom_edu", "mom_age"], group_keys=False).apply(smooth_weights)
        df = df.reset_index(drop=True)
        
        # Renormalize weights within each year
        if label == "main":
            df["weight"] = df.groupby("year")["weight"].transform(lambda x: x / x.sum())
            df = df.reset_index(drop=True)
        
        # For AJK and GB, renormalize weights within each province x year
        else:
            df["weight"] = df.groupby(["province", "year"])["weight"].transform(lambda x: x / x.sum())
            df = df.reset_index(drop=True)

        # --------------------------------------------------------
        # Print, check normalization, and save to pickle
        # --------------------------------------------------------
        print("\nInterpolated and normalized mom distribution...")
        print(df)
        
        # Check the normalization
        print("\nNormalization check:")
        print(df[["year","weight"]].groupby("year").sum())
        
        # Save to pickle
        df.to_pickle(f"../pickle_jar/mom_distribution_{label}.pkl")
