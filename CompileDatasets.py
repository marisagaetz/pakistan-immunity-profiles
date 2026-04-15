""" CompileDatasets.py  

Create semimonthly province + national datasets for SurvivalPrior,
with SIA eligibility encoded directly on the semimonth birth grid.

Outputs
-------
1) pickle_jar/semimonthly_prov_dataset.pkl
   Semimonthly, by (province, time). Includes:
     - births (count per semimonth), births_var, births_std
     - mcv1/mcv2 (rates, interpolated to semimonth), mcv1_var/mcv2_var, stds
     - cases counts (confirmed/discarded/clinical/cases) per semimonth (2012-2025)
     - One 0/1 column per SIA campaign, indicating whether that semimonth of births
         in that province is in the SIA eligibility window AND the province is targeted.

2) pickle_jar/national_dataset.pkl
   Semimonthly, by time (single province="national"). Includes:
     - births (sum), births_var (sum), births_std
     - mcv1/mcv2 from model national estimates (interpolated to semimonth)
     - mcv1_var/mcv2_var propagated from model estimates
     - cases summed
     - One 0/1 column per SIA campaign, indicating whether that semimonth of births
            is age-eligible nationally (province targeting ignored, by design).

3) pickle_jar/sia_cal.pkl
   Campaign calendar with:
     - age_start, age_end (months; inclusive end convention)
     - start_elig, end_elig (DOB window endpoints)
     - provinces list
     - elig_* (by province) and elig_pakistan (counts), computed from the semimonthly datasets
     - coverage_targeted_area, coverage_pakistan computed as doses / eligibility
"""

import os
import re
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

# --------------------------------------------------------------------
# Config / IO
# --------------------------------------------------------------------

PICKLE_DIR = "pickle_jar"
OUT_PROV_PKL = os.path.join(PICKLE_DIR, "semimonthly_prov_dataset.pkl")
OUT_NAT_PKL  = os.path.join(PICKLE_DIR, "national_dataset.pkl")
OUT_SIA_CAL_PKL = os.path.join(PICKLE_DIR, "sia_cal.pkl")

PROVINCES = ["punjab", "balochistan", "sindh", "kp", "ict", "ajk", "gb"]

# SIA calendar CSV
SIA_CAL_FILE = "V_SIA_MAIN_MR.csv"

# Inputs
IN_MONTHLY_BIRTHS = os.path.join(PICKLE_DIR, "monthly_births_by_province.pkl")
IN_MCV_MONTHLY    = os.path.join(PICKLE_DIR, "mcv_prov_cov_monthly.pkl")
IN_MCV_NAT        = os.path.join(PICKLE_DIR, "mcv_nat_cov_monthly.pkl")
IN_LINELIST       = os.path.join(PICKLE_DIR, "combined_linelist_regressed.pkl")

# --------------------------------------------------------------------
# Semimonth utilities
# --------------------------------------------------------------------

def _to_semimonth_stamp(dt: pd.Timestamp) -> pd.Timestamp:
    """
    Round an arbitrary datetime to a semimonth stamp:
      - days 1..15  -> YYYY-MM-01
      - days 16..end-> YYYY-MM-15
    """
    dt = pd.to_datetime(dt)
    day = int(dt.day)
    if day <= 15:
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return dt.replace(day=15, hour=0, minute=0, second=0, microsecond=0)


def _semimonth_grid(t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DatetimeIndex:
    """
    Semimonth grid from month-start t0..t1 (inclusive), with stamps at 1st and 15th.
    """
    t0 = pd.to_datetime(t0).to_period("M").to_timestamp()
    t1 = pd.to_datetime(t1).to_period("M").to_timestamp()
    months = pd.date_range(start=t0, end=t1, freq="MS")
    semi = months.union(months + pd.Timedelta(days=14)).sort_values()
    return semi


def _safe_col(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s

# --------------------------------------------------------------------
# Births and MCV semimonth conversion
# --------------------------------------------------------------------

def monthly_to_semimonthly_births(df, time_col="time", group_col="province", cols=("avg", "var")):
    """
    Monthly totals at month-start -> semimonth births by splitting each month into two halves.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values([group_col, time_col])

    first = df.copy()
    fifteenth = df.copy()
    fifteenth[time_col] = fifteenth[time_col] + pd.offsets.Day(14)

    out = pd.concat([first, fifteenth], ignore_index=True).sort_values([group_col, time_col])

    # Split totals (and variances) in half
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce") / 2.0

    return out


def monthly_rates_to_semimonthly(
    df_monthly: pd.DataFrame,
    time_col: str = "time",
    group_col: str = "province",
    rate_cols=("mcv1", "mcv2"),
    var_cols=("mcv1_var", "mcv2_var"),
    make_std: bool = True,
) -> pd.DataFrame:
    """
    Monthly (month-start) -> semimonthly (1st + 15th), with time interpolation.
    Designed for rate-like columns (0..1) + variances (>=0).
    """
    df = df_monthly.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values([group_col, time_col])

    # Normalize to month-start safely
    df[time_col] = df[time_col].values.astype("datetime64[M]")

    t0 = df[time_col].min()
    t1 = df[time_col].max()
    semi = _semimonth_grid(t0, t1)

    numeric_cols = [c for c in list(rate_cols) + list(var_cols) if c in df.columns]
    keep_cols = [c for c in df.columns if c not in (numeric_cols + [time_col])]

    out_parts = []
    for g, gdf in df.groupby(group_col, sort=False):
        gdf = gdf.copy().sort_values(time_col).set_index(time_col)
        gdf = gdf[~gdf.index.duplicated(keep="last")]
        gdf = gdf.reindex(semi)
        gdf[group_col] = g

        if numeric_cols:
            gdf[numeric_cols] = gdf[numeric_cols].astype(float).interpolate(method="time")

        for c in rate_cols:
            if c in gdf.columns:
                gdf[c] = gdf[c].clip(0.0, 1.0)

        for c in var_cols:
            if c in gdf.columns:
                gdf[c] = gdf[c].clip(lower=0.0)

        for c in keep_cols:
            if c == group_col:
                continue
            gdf[c] = gdf[c].ffill().bfill()

        if make_std:
            for v in var_cols:
                if v in gdf.columns:
                    gdf[v.replace("_var", "_std")] = np.sqrt(gdf[v].astype(float).clip(lower=0.0))

        gdf.index.name = time_col
        out_parts.append(gdf.reset_index())

    out = pd.concat(out_parts, ignore_index=True).sort_values([group_col, time_col]).reset_index(drop=True)
    return out

# --------------------------------------------------------------------
# Linelist -> semimonth case counts
# --------------------------------------------------------------------

def build_semimonth_cases(linelist_path: str) -> pd.DataFrame:
    """
    Returns semimonth case counts by (province, time) with columns:
      confirmed, discarded, clinical, cases

    `cases` is computed as:
      - confirmed (is_case == 1): contributes 1.0
      - discarded (is_case == 0): contributes 0.0
      - clinical/suspected (is_case is NaN): contributes conf_prob
    """
    ll = pd.read_pickle(linelist_path).copy()
    ll["time"] = pd.to_datetime(ll["time"], errors="coerce")
    ll = ll.dropna(subset=["time"]).copy()

    # Expected case value: confirmed=1, discarded=0, suspected=conf_prob
    ll["case"] = ll["is_case"]
    m = ll["case"].isna()
    ll.loc[m, "case"] = ll.loc[m, "conf_prob"].fillna(0.0)

    # Semimonth bin
    ll["time"] = ll["time"].dt.to_period("M").astype(str) + np.where(ll["time"].dt.day <= 15, "-01", "-15")
    ll["time"] = pd.to_datetime(ll["time"], errors="coerce")

    cases = (
        ll.groupby(["province", "time"])
          .apply(lambda g: pd.Series({
              "confirmed": int((g["is_case"] == 1).sum()),
              "discarded": int((g["is_case"] == 0).sum()),
              "clinical": int(g["is_case"].isna().sum()),
              "cases": float(pd.to_numeric(g["case"], errors="coerce").fillna(0.0).sum()),
          }))
          .reset_index()
    )

    prov_map = {
        "Punjab": "punjab",
        "Balochistan": "balochistan",
        "Sindh": "sindh",
        "Khyber Pakhtunkhwa": "kp",
        "Islamabad": "ict",
        "AJK": "ajk",
        "Gilgit Baltistan": "gb",
    }
    cases["province"] = cases["province"].replace(prov_map).astype(str).str.lower()
    return cases


# --------------------------------------------------------------------
# SIA calendar parsing and eligibility columns
# --------------------------------------------------------------------

def _parse_sia_calendar(csv_path: str) -> pd.DataFrame:
    """
    Read SIA calendar CSV and return cleaned df with:
      time, doses, AGEGROUP, age_start, age_end, name, provinces, start_elig, end_elig
    """
    sia_cal = pd.read_csv(
        csv_path,
        usecols=[
            "COUNTRY_NAME", "START_DATE", "END_DATE",
            "TARGET", "DOSES", "AGEGROUP", "ACTIVITY_AREAS_COMMENT",
        ],
    )
    sia_cal = sia_cal.loc[sia_cal["COUNTRY_NAME"] == "Pakistan"].copy()

    sia_cal["doses"] = pd.to_numeric(sia_cal["DOSES"], errors="coerce")
    sia_cal["doses"] = sia_cal["doses"].fillna(pd.to_numeric(sia_cal["TARGET"], errors="coerce")).fillna(0.0)

    sia_cal["START_DATE"] = pd.to_datetime(sia_cal["START_DATE"], errors="coerce")
    sia_cal["END_DATE"] = pd.to_datetime(sia_cal["END_DATE"], errors="coerce")

    # Median campaign date
    sia_cal["time"] = sia_cal.apply(
        lambda r: r["START_DATE"] if pd.isna(r["END_DATE"])
        else r["START_DATE"] + pd.to_timedelta((r["END_DATE"] - r["START_DATE"]).days // 2, unit="d"),
        axis=1,
    )
    sia_cal = sia_cal.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    age_range = {
        "12-59 M": (12, 59),
        "9 M-15 Y": (9, 179),
        "9 M-13 Y": (9, 155),
        "6-59 M": (6, 59),
        "9 M-<13 Y": (9, 155),
        "9-59 M": (9, 59),
        "9 M-9 Y": (9, 107),
        "6 M-9 Y": (6, 107),
        "6 M-10 Y": (6, 119),
        "9-119 M": (9, 119),
        "9-180 M": (9, 179),
        "6 M-5 Y": (6, 59),
    }

    ages = sia_cal["AGEGROUP"].map(age_range)
    sia_cal[["age_start", "age_end"]] = pd.DataFrame(ages.tolist(), index=sia_cal.index)

    # Stable unique campaign names
    sia_cal["name_base"] = "SIA " + sia_cal["time"].dt.strftime("%b %Y")
    sia_cal["name"] = sia_cal["name_base"] + " #" + (sia_cal.groupby("name_base").cumcount() + 1).astype(str)

    # DOB eligibility window (continuous)
    sia_cal["start_elig"] = sia_cal.apply(lambda r: r["time"] - relativedelta(months=int(r["age_end"]) + 1), axis=1)
    sia_cal["end_elig"]   = sia_cal.apply(lambda r: r["time"] - relativedelta(months=int(r["age_start"])), axis=1)

    # Province targeting
    area_map = {
        "6 districts in Kashmir affected by the earthquake": ["ajk", "gb"],
        "Phase 1: 4 districts: Mirpur, Mardan, Gujrat, Dadu": ["ajk", "kp", "punjab", "sindh"],
        "Phase 2: 6 districts": PROVINCES,
        "Phase 3: 40 districts": PROVINCES,
        "Phase 4: Sindh (48 districts)": ["sindh"],
        "Phase 5": PROVINCES,
        "Flood response - Balochistan (16 Districts out of 30 Districts)": ["balochistan"],
        "8 districts of Sindh and 4 districts of Balochistan": ["sindh", "balochistan"],
        "7 districts of KP": ["kp"],
        "11 districts of Punjab": ["punjab"],
        "18 towns of Karachi, 05 Districts of Sindh": ["sindh"],
        "7 districts of Gilgit Baltistan": ["gb"],
        "10 districts of AJK": ["ajk"],
        "Sindh": ["sindh"],
        "KP": ["kp"],
        "AJK": ["ajk"],
        "Punjab": ["punjab"],
        "CDA - Islamabad": ["ict"],
        "ICT - Islamabad": ["ict"],
        "Balochistan": ["balochistan"],
        "Gilgit Baltistan": ["gb"],
        "FATA": ["kp"],
        "High Risk UC s of 5 districts of Balochistan\n9-59 months in district Duki, Killa Abdullah and Sibi\n60-119 months in district Gwadar and Kalat": ["balochistan"],
        "181 selected UCs in 17 districts of Punjab": ["punjab"],
        "2/3 in 2021": PROVINCES,
    }

    sia_cal["provinces"] = sia_cal["ACTIVITY_AREAS_COMMENT"].map(area_map)
    sia_cal["provinces"] = sia_cal["provinces"].apply(lambda x: x if isinstance(x, list) else PROVINCES)

    return sia_cal


def _campaign_colname(campaign_idx: int, sia_row: pd.Series) -> str:
    """
    Build a deterministic, filesystem/df-friendly column name for this campaign.
    """
    t = pd.to_datetime(sia_row["time"])
    stamp = t.strftime("%Y%m%d")
    safe_name = _safe_col(sia_row.get("name", f"campaign_{campaign_idx}"))
    return f"sia__{stamp}__{int(campaign_idx):03d}__{safe_name}"


def _eligibility_semimonth_bounds(sia_row: pd.Series):
    """
    Convert continuous DOB eligibility bounds into semimonth-stamp bounds (left-open, right-closed).
    Returns (start_sm, end_sm) as Timestamps on the semimonth grid.
    """
    start = _to_semimonth_stamp(pd.to_datetime(sia_row["start_elig"]))
    end   = _to_semimonth_stamp(pd.to_datetime(sia_row["end_elig"]))
    return start, end


def add_sia_eligibility_columns_prov(
    prov_sm: pd.DataFrame,
    sia_cal: pd.DataFrame,
    provinces=PROVINCES,
) -> pd.DataFrame:
    """
    Add 0/1 SIA eligibility columns to semimonthly_prov_dataset:
      1 iff province targeted AND birth semimonth is in (start_elig_sm, end_elig_sm]
    """
    df = prov_sm.copy()
    df["province"] = df["province"].astype(str).str.lower()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()

    for i, row in sia_cal.iterrows():
        col = _campaign_colname(i, row)
        start_sm, end_sm = _eligibility_semimonth_bounds(row)
        targeted = set([p.lower() for p in row["provinces"]])

        df[col] = np.where(
            (df["province"].isin(targeted)) &
            (df["time"] > start_sm) & (df["time"] <= end_sm),
            1,
            0
        ).astype(int)

    return df


def add_sia_eligibility_columns_national(
    nat_sm: pd.DataFrame,
    sia_cal: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add 0/1 SIA eligibility columns to national semimonth dataset:
      1 iff birth semimonth is age-eligible (targeting ignored).
    """
    df = nat_sm.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()

    for i, row in sia_cal.iterrows():
        col = _campaign_colname(i, row)
        start_sm, end_sm = _eligibility_semimonth_bounds(row)
        df[col] = np.where((df["time"] > start_sm) & (df["time"] <= end_sm), 1, 0).astype(int)

    return df

# --------------------------------------------------------------------
# Eligibility counts from semimonth datasets
# --------------------------------------------------------------------

def eligible_from_sia_column_prov(
    prov_sm_with_sia: pd.DataFrame,
    sia_col: str,
    province: str,
    births_col: str = "births",
) -> float:
    """
    Eligibility count for one province, one SIA, using 0/1 column * births.
    """
    df = prov_sm_with_sia
    m = (df["province"] == province) & (df[sia_col].astype(int) == 1)
    return float(pd.to_numeric(df.loc[m, births_col], errors="coerce").fillna(0.0).sum())


def eligible_from_sia_column_national(
    nat_sm_with_sia: pd.DataFrame,
    sia_col: str,
    births_col: str = "births",
) -> float:
    df = nat_sm_with_sia
    m = (df[sia_col].astype(int) == 1)
    return float(pd.to_numeric(df.loc[m, births_col], errors="coerce").fillna(0.0).sum())

# --------------------------------------------------------------------
# Main build
# --------------------------------------------------------------------

if __name__ == "__main__":
    # Load births + convert to semimonth
    monthly_births = pd.read_pickle(IN_MONTHLY_BIRTHS).reset_index(drop=False)
    if "province" not in monthly_births.columns:
        monthly_births = monthly_births.rename(columns={"index": "province"})

    births_sm = monthly_to_semimonthly_births(
        monthly_births,
        time_col="time",
        group_col="province",
        cols=("avg", "var"),
    )
    births_sm["province"] = births_sm["province"].astype(str).str.lower()
    births_sm["time"] = pd.to_datetime(births_sm["time"], errors="coerce")
    births_sm = births_sm.dropna(subset=["time"]).copy()
    births_sm["std"] = np.sqrt(pd.to_numeric(births_sm["var"], errors="coerce").fillna(0.0).clip(lower=0.0))

    # Rename to consistent schema
    births_sm = births_sm.rename(columns={"avg": "births", "var": "births_var", "std": "births_std"})

    # Load MCV monthly + convert to semimonth
    mcv_monthly = pd.read_pickle(IN_MCV_MONTHLY).reset_index(drop=False)
    if "province" not in mcv_monthly.columns:
        mcv_monthly = mcv_monthly.rename(columns={"index": "province"})

    mcv_sm = monthly_rates_to_semimonthly(
        mcv_monthly,
        time_col="time",
        group_col="province",
        rate_cols=("mcv1", "mcv2"),
        var_cols=("mcv1_var", "mcv2_var"),
        make_std=True,
    )
    mcv_sm["province"] = mcv_sm["province"].astype(str).str.lower()

    # Linelist -> semimonth cases
    cases_sm = build_semimonth_cases(IN_LINELIST)
    cases_sm["province"] = cases_sm["province"].astype(str).str.lower()

    # Merge to province semimonth base grid
    base = pd.merge(
        births_sm,
        mcv_sm,
        on=["province", "time"],
        how="outer",
        validate="m:m",
    )
    prov_sm = pd.merge(
        base,
        cases_sm,
        on=["province", "time"],
        how="left",
        validate="m:m",
    )
    prov_sm = prov_sm.sort_values(["province", "time"]).reset_index(drop=True)

    # Build SIA calendar
    sia_cal = _parse_sia_calendar(os.path.join("_data",SIA_CAL_FILE)).copy()

    # Add SIA eligibility columns to prov_sm (0/1 per row)
    prov_sm = add_sia_eligibility_columns_prov(prov_sm, sia_cal, provinces=PROVINCES)
    prov_sm.to_pickle(OUT_PROV_PKL)
    print("Wrote:", OUT_PROV_PKL)

    # Build national semimonth dataset
    numeric_cols = [
        "births", "births_var",
        "mcv1", "mcv1_var", "mcv2", "mcv2_var",
        "confirmed", "discarded", "clinical", "cases",
    ]
    for c in numeric_cols:
        if c in prov_sm.columns:
            prov_sm[c] = pd.to_numeric(prov_sm[c], errors="coerce")

    # Totals (sums across provinces)
    national_totals = (
        prov_sm.groupby("time", as_index=False)
               .agg(
                   births=("births", "sum"),
                   births_var=("births_var", "sum"),
                   confirmed=("confirmed", "sum"),
                   discarded=("discarded", "sum"),
                   clinical=("clinical", "sum"),
                   cases=("cases", "sum"),
               )
    )
    national_totals["births_std"] = np.sqrt(national_totals["births_var"].fillna(0.0).clip(lower=0.0))

    # National MCV from model national monthly estimates
    nat_mcv_monthly = pd.read_pickle(IN_MCV_NAT).copy()
    nat_mcv_monthly["time"] = pd.to_datetime(nat_mcv_monthly["time"], errors="coerce")

    # Convert monthly -> semimonthly via interpolation
    nat_mcv_monthly["province"] = "national"
    nat_mcv_sm = monthly_rates_to_semimonthly(
        nat_mcv_monthly,
        time_col="time",
        group_col="province",
        rate_cols=("mcv1", "mcv2"),
        var_cols=("mcv1_var", "mcv2_var"),
        make_std=True,
    )
    nat_mcv_sm = nat_mcv_sm.drop(columns=["province"])
    national_mcv = nat_mcv_sm

    nat_sm = pd.merge(national_totals, national_mcv, on="time", how="outer").sort_values("time")
    nat_sm.insert(0, "province", "national")

    # Add national 0/1 SIA eligibility columns (age window only, targeting ignored)
    nat_sm = add_sia_eligibility_columns_national(nat_sm, sia_cal)

    nat_sm.to_pickle(OUT_NAT_PKL)
    print("Wrote:", OUT_NAT_PKL)

    # Compute elig_* columns in sia_cal from the semimonthly datasets
    # Province eligibility: sum births where (province, time) is marked eligible.
    # National eligibility: sum births where national time is marked eligible.

    sia_cols = {}
    for i, row in sia_cal.iterrows():
        sia_cols[i] = _campaign_colname(i, row)

    # Province eligibles
    for prov in PROVINCES:
        colname = f"elig_{prov}"
        vals = []
        for i in sia_cal.index:
            vals.append(eligible_from_sia_column_prov(prov_sm, sia_cols[i], prov, births_col="births"))
        sia_cal[colname] = vals

    # National eligible (targeting ignored)
    elig_pak = []
    for i in sia_cal.index:
        elig_pak.append(eligible_from_sia_column_national(nat_sm, sia_cols[i], births_col="births"))
    sia_cal["elig_pakistan"] = elig_pak

    # Targeted-area eligible = sum across targeted provinces
    targeted_elig = []
    for i, row in sia_cal.iterrows():
        provs = [p.lower() for p in row["provinces"]]
        s = 0.0
        for p in provs:
            if p in PROVINCES:
                s += float(row.get(f"elig_{p}", 0.0))
        targeted_elig.append(s)
    sia_cal["total_elig_targeted"] = targeted_elig

    # Coverage = doses / eligibility
    sia_cal["coverage_targeted_area"] = (
        pd.to_numeric(sia_cal["doses"], errors="coerce").fillna(0.0)
        / pd.to_numeric(sia_cal["total_elig_targeted"], errors="coerce").replace(0.0, np.nan)
    ).fillna(0.0).clip(upper=1.0)

    sia_cal["coverage_pakistan"] = (
        pd.to_numeric(sia_cal["doses"], errors="coerce").fillna(0.0)
        / pd.to_numeric(sia_cal["elig_pakistan"], errors="coerce").replace(0.0, np.nan)
    ).fillna(0.0).clip(upper=1.0)

    # Save sia_cal
    sia_cal.to_pickle(OUT_SIA_CAL_PKL)
    print("Wrote:", OUT_SIA_CAL_PKL)

    # Preview
    print("\nPreview sia_cal:")
    show_cols = ["time", "name", "AGEGROUP", "doses", "elig_pakistan", "total_elig_targeted",
                 "coverage_pakistan", "coverage_targeted_area"]
    show_cols = [c for c in show_cols if c in sia_cal.columns]
    print(sia_cal[show_cols].head())

