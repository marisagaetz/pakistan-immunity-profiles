""" survey_io.py

Unified survey-loading utilities for DHS and MICS (Pakistan).

1. Set your survey directory:
   Edit BASE and SURVEYS at the top of this file so that SURVEYS points
   to the folder containing your DHS and MICS survey subfolders
   (e.g. DHS5_2006/, MICS6_Punjab_2017/, etc.).

2. Run this file directly to inspect your data: 
   This prints a debug report 
   showing the unique values in every column for all survey files listed in 
   the path lists below. Use this to spot unexpected or inconsistent values.

3. Update the fix maps as needed:
   If the debug report reveals new inconsistencies (e.g. a province name
   you haven't seen before, a new spelling of "missing"), update the
   relevant map: PROVINCE_MAP, EDU_CATS, MCV_CATS, MCV_YEAR_FIX,
   MONTH_FIX, or DOSES_FIX.

4. Import from other scripts:
   Once the data is clean, use load_survey() in your analysis scripts:
       from survey_io import load_survey
       df = load_survey(path, add_survey=True, columns=["province", "mcv1"])

Note: the loader infers survey type (DHS/MICS), MICS wave (4/5/6),
   and recode type (IR/BR/KR/WM/BH/CH) from the folder and file names.
   Survey folders must contain "dhs" or "mics" in the name (e.g.
   DHS7_2017/, MICS6_Punjab_2017/), and MICS folders must include the
   wave number. Recode files must contain the recode code in the
   filename (e.g. PKIR52FL.DTA, wm.sav, ch.sav).

Supported survey types: DHS, MICS (waves 4–6)
Supported recode types: IR, BR, KR (DHS); WM, BH, CH (MICS)
"""

import os
import re
import pandas as pd
import numpy as np

# -------------------------------------------------------------------
# Load survey data
# -------------------------------------------------------------------

BASE = os.path.dirname(os.path.dirname(__file__))
SURVEYS = os.path.join(BASE, "_data", "_surveys")

dhs_ir_paths = [
    os.path.join(SURVEYS, "DHS5_2006", "PKIR52DT", "pkir52fl.dta"),
    os.path.join(SURVEYS, "DHS6_2012", "PKIR61DT", "PKIR61FL.DTA"),
    os.path.join(SURVEYS, "DHS7_2017", "PKIR71DT", "PKIR71FL.DTA"),
    os.path.join(SURVEYS, "DHS7_2019", "PKIQ7AFL_recode.DTA")
]

dhs_br_paths = [
    os.path.join(SURVEYS, "DHS5_2006", "PKBR52DT", "pkbr52fl.dta"),
    os.path.join(SURVEYS, "DHS6_2012", "PKBR61DT", "PKBR61FL.DTA"),
    os.path.join(SURVEYS, "DHS7_2017", "PKBR71DT", "PKBR71FL.DTA")
]

dhs_kr_paths = [
    os.path.join(SURVEYS, "DHS5_2006", "PKKR52DT", "pkkr52fl.dta"),
    os.path.join(SURVEYS, "DHS6_2012", "PKKR61DT", "PKKR61FL.DTA"),
    os.path.join(SURVEYS, "DHS7_2017", "PKKR71DT", "PKKR71FL.DTA")
]

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
    os.path.join(SURVEYS, "MICS6_Sindh_2018", "wm.sav")
]

mics_bh_paths = [
    os.path.join(SURVEYS, "MICS6_Balochistan_2019", "bh.sav"),
    os.path.join(SURVEYS, "MICS6_KP_2019", "bh.sav"),
    os.path.join(SURVEYS, "MICS6_Punjab_2017", "bh.sav"),
    os.path.join(SURVEYS, "MICS6_Sindh_2018", "bh.sav")
]

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
    os.path.join(SURVEYS, "MICS6_Sindh_2018", "ch.sav")
]

# -------------------------------------------------------------------
# Infer survey type and recode type
# -------------------------------------------------------------------

def infer_survey_type(path):
    lower = path.lower()
    if "dhs" in lower: return "dhs"
    if "mics" in lower: return "mics"
    raise ValueError(f"Cannot detect survey type from path: {path}")

def infer_survey_folder(path):
    """Return the folder name that includes 'dhs' or 'mics'."""
    parts = path.lower().split(os.path.sep)
    for p in parts:
        if "dhs" in p or "mics" in p:
            return p
    return None

def infer_mics_wave(path):
    lower = path.lower()
    if "mics6" in lower: return 6
    if "mics5" in lower: return 5
    if "mics4" in lower: return 4
    raise ValueError(f"Cannot detect MICS wave from path: {path}")


def infer_recode_type(path):
    base = os.path.basename(path).lower()
    if "ir" in base or "iq" in base: return "ir"
    if "br" in base: return "br"
    if "kr" in base: return "kr"
    if base.startswith("wm"): return "wm"
    if base.startswith("bh"): return "bh"
    if base.startswith("ch"): return "ch"
    raise ValueError(f"Cannot infer recode type from filename: {path}")


# -------------------------------------------------------------------
# Schemas (full raw→recoded mapping)
# -------------------------------------------------------------------

# --- DHS schemas ---
# Raw variable codes follow the DHS recode manual naming convention.
# See https://dhsprogram.com/publications/publication-dhsg4-dhs-questionnaires-and-manuals.cfm

DHS_IR_SCHEMA = {
    "caseid": "caseid",        # unique respondent ID
    "awfactt": "awfactt",      # all-women factor (for ever-married surveys)
    "v011": "mom_DoB",         # CMC date of birth of mother
    "v013": "mom_age",         # age in 5-year groups
    "v106": "mom_edu",         # highest education level
    "v224": "num_brs",         # number of entries in birth recode
    "v008": "interview_date",  # CMC date of interview
    "v024": "province",        # region/province
    "v025": "area",            # urban/rural
    "v005": "weight",          # sample weight (÷1e6)
}

DHS_BR_SCHEMA = {
    "caseid": "caseid",        # mother's unique ID
    "bord": "bord",            # birth order number
    "b3": "child_DoB",         # CMC date of birth of child
    "v011": "mom_DoB",         # CMC date of birth of mother
    "v008": "interview_date",  # CMC date of interview
    "v005": "weight",          # sample weight (÷1e6)
    "v024": "province",        # region/province
}

DHS_KR_SCHEMA = {
    "caseid": "caseid",        # mother's unique ID
    "bord": "bord",            # birth order number
    "b3": "child_DoB",         # CMC date of birth of child
    "b5": "live_child",        # is the child alive? (yes/no)
    "v011": "mom_DoB",         # CMC date of birth of mother
    "v008": "interview_date",  # CMC date of interview
    "h9": "mcv1",              # measles/MCV1 vaccination status
    "v005": "weight",          # sample weight (÷1e6)
    "h9d": "mcv1_day",         # day of MCV1 vaccination (from card)
    "h9m": "mcv1_mon",         # month of MCV1 vaccination (from card)
    "h9y": "mcv1_yr",          # year of MCV1 vaccination (from card)
}

# --- MICS schemas ---
# Raw variable codes follow MICS naming conventions, which vary by wave.

MICS4_WM_SCHEMA = MICS5_WM_SCHEMA = {
    "HH1": "cluster",             # cluster number
    "HH2": "hh",                  # household number
    "LN": "line_num",             # woman's line number
    "WM6D": "interview_day",      # interview day
    "WM6M": "interview_mon",      # interview month
    "WM6Y": "interview_year",     # interview year
    "WB1M": "mom_birth_mon",      # mother's birth month
    "WB1Y": "mom_birth_year",     # mother's birth year
    "WB4": "mom_edu",             # highest education level
    "CM10": "num_brs",            # total children ever born
    "HH6": "area",                # urban/rural
    "wmweight": "weight",         # woman's sample weight
}

MICS6_WM_SCHEMA = {
    "HH1": "cluster",
    "HH2": "hh",
    "LN": "line_num",
    "WM6D": "interview_day",
    "WM6M": "interview_mon",
    "WM6Y": "interview_year",
    "WB3M": "mom_birth_mon",      # note: WB3M in MICS6 vs WB1M in MICS4/5
    "WB3Y": "mom_birth_year",     # note: WB3Y in MICS6 vs WB1Y in MICS4/5
    "welevel": "mom_edu",         # note: welevel in MICS6 vs WB4 in MICS4/5
    "CM11": "num_brs",            # note: CM11 in MICS6 vs CM10 in MICS4/5
    "HH6": "area",
    "wmweight": "weight",
}

# MICS4 and MICS5 did not include a birth history (BH) recode file.
MICS4_BH_SCHEMA = MICS5_BH_SCHEMA = {}

MICS6_BH_SCHEMA = {
    "HH1": "cluster",
    "HH2": "hh",
    "LN": "line_num",
    "BHLN": "bord",               # birth order (line number in birth history)
    "BH4M": "child_birth_mon",    # child's birth month
    "BH4Y": "child_birth_year",   # child's birth year
    "wmweight": "weight",
}

MICS4_CH_SCHEMA = {
    "HH1": "cluster",
    "HH2": "hh",
    "UF6": "line_num",            # caregiver's line number
    "AG1D": "child_birth_day",
    "AG1M": "child_birth_mon",
    "AG1Y": "child_birth_year",
    "AG2": "child_age",           # age in completed years
    "IM3MY": "mcv1_year",         # MCV1 year from vaccination card
    "IM16": "mcv1",               # MCV1 ever received (mother's recall)
}

MICS5_CH_SCHEMA = {
    "HH1": "cluster",
    "HH2": "hh",
    "UF6": "line_num",
    "AG1D": "child_birth_day",
    "AG1M": "child_birth_mon",
    "AG1Y": "child_birth_year",
    "AG2": "child_age",
    "IM3M1Y": "mcv1_year",        # MCV1 year from card (MICS5 naming)
    "IM3M2Y": "mcv2_year",        # MCV2 year from card
    "IM16": "mcv1",               # MCV1 ever received (recall)
}

MICS6_CH_SCHEMA = {
    "HH1": "cluster",
    "HH2": "hh",
    "UF4": "line_num",            # note: UF4 in MICS6 vs UF6 in MICS4/5
    "UB1D": "child_birth_day",    # note: UB1D in MICS6 vs AG1D in MICS4/5
    "UB1M": "child_birth_mon",
    "UB1Y": "child_birth_year",
    "UB2": "child_age",           # note: UB2 in MICS6 vs AG2 in MICS4/5
    "IM6M1Y": "mcv1_year",        # MCV1 year from card (MICS6 naming)
    "IM6M2Y": "mcv2_year",        # MCV2 year from card (MICS6 naming)
    "IM26": "mcv1",               # MCV1 ever received (MICS6 naming)
    "IM26A": "num_doses",         # total measles/rubella doses received
}

def get_schema(path):
    survey = infer_survey_type(path)
    recode = infer_recode_type(path)
    survey_name = infer_survey_folder(path)

    dhs_recode_schema = {
        "ir": DHS_IR_SCHEMA,
        "br": DHS_BR_SCHEMA,
        "kr": DHS_KR_SCHEMA
    }

    if survey == "dhs":
        if survey_name == "dhs7_2017":
            schema = dhs_recode_schema[recode].copy()
            schema["sv005"] = "sweight"
            if recode == "kr":
                schema["h9a"] = "mcv2"
            return schema
        else:
            return dhs_recode_schema[recode]

    wave = infer_mics_wave(path)
    if recode == "wm":
        return {4: MICS4_WM_SCHEMA, 5: MICS5_WM_SCHEMA, 6: MICS6_WM_SCHEMA}[wave]
    if recode == "bh":
        return {4: MICS4_BH_SCHEMA, 5: MICS5_BH_SCHEMA, 6: MICS6_BH_SCHEMA}[wave]
    if recode == "ch":
        return {4: MICS4_CH_SCHEMA, 5: MICS5_CH_SCHEMA, 6: MICS6_CH_SCHEMA}[wave]

    raise ValueError(f"No schema for path: {path}")


# -------------------------------------------------------------------
# Cleaning utilities
# -------------------------------------------------------------------

PROVINCE_MAP = {
    "nwfp": "kp",
    "islamabad (ict)": "ict",
    "gilgit baltistan": "gb",
    "khyber pakhtunkhwa": "kp",
    "fata": "kp",
    "kpk": "kp",
}

EDU_CATS = {
    np.nan: "no education",
    "non-formal": "no education",
    "no response": "no education",
    "missing": "no education",
    "missing/dk": "no education",
    "dk/missing": "no education",
    "dk": "no education",
    "none/preschool": "no education",
    "pre-primary or none": "no education",
    "eccde": "no education",
    "vei/iei": "no education",
    "preschool": "primary",
    "madrassa": "primary",
    "middle": "secondary",
    "matric": "secondary",
    "matriculation": "secondary",
    "senior secondary": "secondary",
    "junior secondary": "secondary",
    "secondary technical": "secondary",
    "secondary / secondary-technical": "secondary",
    "higher/tertiary": "higher",
    "above matric": "higher",
    "master's degree or mbbs, phd, mphil, bsc (4 years)": "higher",
    "less than class 1 completed": "no education",
    "": "no education"
}

MCV_CATS = {
    "no": 0,
    "reported by mother": 1,
    "vaccination date on card": 1,
    "vacc. date on card": 1,
    "vaccination marked on card": 1,
    "vacc. marked on card": 1,
    "yes": 1,
    "don't know": np.nan,
    "dk": np.nan,
    "missing": np.nan,
    "9.0": np.nan,
    "inconsistant": np.nan,
    "no response": np.nan,
}

MCV_YEAR_FIX = {
    # Strings that mean "no usable year" → treat as missing
    "missing": np.nan,
    "inconsistant": np.nan,
    "inconsistent": np.nan,
    "not given": np.nan,
    "dk": np.nan,
    "no response": np.nan,
    # Strings that mean "vaccinated but year not recorded" → map to 1
    # so that pd.to_numeric().notnull() returns True, which feeds into 
    # the mcv1_with_card flag downstream in clean_mics().
    "mother reported": 1,
    "marked on card": 1,
}

DOSES_FIX = {
    "dk": 0,
    "no response": 0
}

MONTH_FIX = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "dk": np.nan, "missing": np.nan, "no response": np.nan,
    "inconsistent": np.nan, "inconsistant": np.nan
}

def fix_month_col(x):
    return (
        x.astype(str).str.strip().str.lower()
         .replace(MONTH_FIX).astype(float)
    )

def fix_year_col(x):
    return pd.to_numeric(x, errors="coerce")


def clean_dhs(df):
    if "caseid" in df:
        df["caseid"] = df["caseid"].astype(str).str.strip()
    if "province" in df:
        df["province"] = df["province"].astype(str).str.lower().replace(PROVINCE_MAP)
    if "area" in df:
        df["area"] = df["area"].astype(str).str.lower()
    if "mom_edu" in df:
        df["mom_edu"] = df["mom_edu"].astype(str).str.lower().replace(EDU_CATS)
    if "mcv1" in df:
        df["mcv1"] = df["mcv1"].astype(str).str.lower().replace(MCV_CATS)
    if "mcv1_day" in df:
        df["mcv1_day"] = pd.to_numeric(df["mcv1_day"], errors="coerce")
    if "mcv1_mon" in df:
        df["mcv1_mon"] = pd.to_numeric(df["mcv1_mon"], errors="coerce")
    if "mcv1_yr" in df:
        df["mcv1_yr"] = pd.to_numeric(df["mcv1_yr"], errors="coerce")
    if "mcv2" in df:
        df["mcv2"] = df["mcv2"].astype(str).str.lower().replace(MCV_CATS)
    if "live_child" in df:
        df["live_child"] = df["live_child"].astype(str).str.lower()
    return df


def clean_mics(df):
    if "area" in df:
        df["area"] = df["area"].astype(str).str.lower()

    if "mom_edu" in df:
        df["mom_edu"] = df["mom_edu"].astype(str).str.lower().replace(EDU_CATS)

    for col in ["mom_birth_mon", "child_birth_mon", "interview_mon"]:
        if col in df:
            df[col] = fix_month_col(df[col])

    for col in ["mom_birth_year", "child_birth_year"]:
        if col in df:
            df[col] = fix_year_col(df[col])

    if "mcv1" in df:
        df["mcv1"] = df["mcv1"].astype(str).str.lower().replace(MCV_CATS)
        df["mcv1_year"] = df["mcv1_year"].astype(str).str.lower().replace(MCV_YEAR_FIX)
        df["mcv1_with_card"] = pd.to_numeric(df["mcv1_year"], errors="coerce").notnull()
        df["mcv1"] = (df["mcv1_with_card"] | df["mcv1"]).astype(int)

    if "num_doses" in df:
        df["num_doses"] = df["num_doses"].astype(str).str.lower().replace(DOSES_FIX)
        df["num_doses"] = pd.to_numeric(df["num_doses"], errors="coerce")

    if "mcv2_year" in df:
        df["mcv2_year"] = df["mcv2_year"].astype(str).str.lower().replace(MCV_YEAR_FIX)
        df["mcv2"] = pd.to_numeric(df["mcv2_year"], errors="coerce").notnull().astype(int)
        if "num_doses" in df:
            df["mcv2_dose"] = df["num_doses"] > 1
            df["mcv2"] = (df["mcv2"] | df["mcv2_dose"]).astype(int)

    return df


# -------------------------------------------------------------------
# Fill in missing year values
# -------------------------------------------------------------------

def cms_to_datetime(a, d=15):
    years = 1900 + ((a - 1) // 12)
    months = (a - 1) % 12 + 1
    return pd.to_datetime({"year": years, "month": months, "day": d})

def infer_year_from_path(path):
    for part in path.split(os.path.sep):
        m = re.search(r"(19|20)\d{2}", part)
        if m: return int(m.group())
    return None

def add_year_column(df, path):
    survey = infer_survey_type(path)
    survey_name = infer_survey_folder(path)
    fallback = infer_year_from_path(path)
    single_year_surveys = ["mics4_balochistan_2010", "mics4_punjab_2011",
                           "mics5_sindh_2014", "mics5_punjab_2014",
                           "mics6_kp_2019"]

    if survey == "dhs":
        dt = df["interview_date"]
        if np.issubdtype(dt.dtype, np.number):
            dt = cms_to_datetime(dt)
        df["year"] = dt.dt.year

    if survey == "mics":
        if "interview_year" in df.columns:
            df["year"] = fix_year_col(df["interview_year"])
            if survey_name in single_year_surveys:
                df["year"] = df["year"].fillna(fallback)
        elif survey_name in single_year_surveys:
            df["year"] = fallback
    return df


# -------------------------------------------------------------------
# Fill in missing mom age info
# -------------------------------------------------------------------

def compute_mics_mom_age(df):
    birth_year = fix_year_col(df["mom_birth_year"])
    birth_mon  = fix_month_col(df["mom_birth_mon"]).fillna(7)
    int_mon    = fix_month_col(df["interview_mon"]).fillna(7)
    int_year   = fix_year_col(df["year"])

    df["mom_DoB"] = pd.to_datetime(
        {"year": birth_year, "month": birth_mon, "day": 1},
        errors="coerce"
    )

    df["interview_date"] = pd.to_datetime(
        {"year": int_year, "month": int_mon, "day": 1},
        errors="coerce"
    )

    df["mom_age_cont"] = (df["interview_date"] - df["mom_DoB"]).dt.days / 365.25

    bins = np.arange(15, 55, 5)
    labels = [f"{a}-{a+4}" for a in bins[:-1]]
    df["mom_age"] = pd.cut(df["mom_age_cont"], bins=bins, labels=labels)

    return df


# -------------------------------------------------------------------
# Main loader
# -------------------------------------------------------------------

def load_survey(path, add_survey=False, convert_categoricals=True, columns=None):
    schema = get_schema(path)
    survey = infer_survey_type(path)
    recode = infer_recode_type(path)
    survey_name = infer_survey_folder(path)

    # Always load all raw schema columns
    raw_cols = list(schema.keys())

    if survey == "dhs":
        df = pd.read_stata(path, columns=raw_cols,
                           convert_categoricals=convert_categoricals)
    else:
        df = pd.read_spss(path, usecols=raw_cols,
                          convert_categoricals=convert_categoricals)

    # Apply renaming
    df = df.rename(columns=schema)

    # Clean
    df = clean_dhs(df) if survey == "dhs" else clean_mics(df)

    # Add year
    df = add_year_column(df, path)

    # Compute mom_age (MICS WM only)
    if survey == "mics" and recode == "wm":
        df = compute_mics_mom_age(df)

    # Add survey label
    if add_survey:
        df["survey"] = survey_name

    # Province for MICS (read from folder)
    if survey == "mics" and "province" not in df.columns:
        parts = survey_name.split("_")
        df["province"] = parts[1].lower()

    # Normalize weights
    if "weight" in df:
        df["weight"] *= 1e-6
        df["weight"] *= 1. / (df["weight"].sum())

    # Add the special weights for AJK and GB from the 2017 DHS
    if "sweight" in df:
        df["sweight"] *= 1e-6
        df["sweight"] *= 1. / (df["sweight"].sum())

    # Restrict to requested columns
    if columns is not None:
        keep = [c for c in columns if c in df.columns]
        if add_survey:
            keep.append("survey")
        if survey_name == "dhs7_2017" and "weight" in columns:
            keep.append("sweight")
        if "mcv1" in columns and "mcv2" in df.columns:
            keep.append("mcv2")
        df = df[keep]

    return df


# -------------------------------------------------------------------
# DEBUG: run this file directly to inspect all survey data
# -------------------------------------------------------------------

def debug_print_unique_values(paths, columns=None, max_unique=20, dropna=True):
    """
    Load each survey file via load_survey() and print all unique values
    per column. Use this to verify that province names, education
    categories, area codes, vaccination fields, and year extraction
    are all behaving as expected. Update the fix maps above if you
    spot anything unexpected.
    """
    print("\n==================== SURVEY DEBUG REPORT =====================")

    for path in paths:
        print("\n--------------------------------------------------------------")
        print(f"FILE: {path}")
        print("--------------------------------------------------------------")

        try:
            df = load_survey(path, add_survey=True, convert_categoricals=True, columns=columns)
        except Exception as e:
            print(f"  ERROR loading file: {e}")
            continue

        print(f"Loaded columns: {list(df.columns)}\n")

        for col in df.columns:
            values = df[col].unique()

            if dropna:
                values = [v for v in values if pd.notna(v)]

            try:
                values = sorted(values)
            except Exception:
                pass

            print(f"  {col} ({len(values)} unique):")

            if len(values) > max_unique:
                shown = values[:max_unique]
                print(f"    {shown} ... ({len(values) - max_unique} more)")
            else:
                print(f"    {values}")

        print()


# Optionally uncomment the `columns` filter to narrow the inspection.
if __name__ == "__main__":
    all_paths = (
        dhs_ir_paths +
        dhs_br_paths +
        dhs_kr_paths +
        mics_wm_paths +
        mics_bh_paths +
        mics_ch_paths
    )

    debug_print_unique_values(
        all_paths,
        # columns = ['province', 'mcv1', 'mcv1_day', 'mcv1_mon', 'mcv1_yr'],
        max_unique=10,
    )