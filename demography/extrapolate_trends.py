""" extrapolate_trends.py

Helper functions for loading World Bank and WHO reference series,
extrapolating province-level estimates using reference trends, and
aggregating to national level.

Used by: YearlyBirths.py, MCVCoverageEstimates.py, MonthlyBirths.py

Sections:
  - World Bank CSV loading (national birth rate & population)
  - WHO national MCV1 and MCV2 coverage loading
  - Generic extrapolation with a reference series
  - Rescaling helpers
  - Ratio-based extrapolation variance
  - Population-share (subregion) weighting
  - Census-based additive shares (for AJK & GB)
  - Monthly interpolation
"""

import re
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# World Bank CSV loading
# ---------------------------------------------------------------------

def get_raw_spreadsheet(filepath, min_year=2006, max_year=None, skiprows=4):
    """
    Load WB-style wide CSV.
    """
    # Discover year columns from header
    header = pd.read_csv(filepath, skiprows=skiprows, nrows=0).columns
    available_years = sorted(
        int(s) for s in (str(c).strip() for c in header) if re.fullmatch(r"\d{4}", s)
    )
    if not available_years:
        raise ValueError(f"No year columns detected in {filepath}. Check skiprows={skiprows}.")

    y0 = int(min_year)
    y1 = int(max_year) if max_year is not None else int(max(available_years))

    use_years = [y for y in available_years if y0 <= y <= y1]
    if not use_years:
        raise ValueError(
            f"No available year columns in [{y0}, {y1}] for {filepath}. "
            f"File has years {available_years[0]}..{available_years[-1]}."
        )

    columns = ["Country Name"] + [str(y) for y in use_years]
    dtypes = {"Country Name": str, **{str(y): np.float64 for y in use_years}}

    df = pd.read_csv(
        filepath, skiprows=skiprows, header=0,
        usecols=columns, dtype=dtypes,
    )
    df.columns = [c.lower().replace(" name", "") for c in df.columns]
    df["country"] = df["country"].str.lower()
    return df


def GetWBPopulation(filepath, min_year=2006, max_year=None, countries=None):
    """
    Load World Bank Population CSV and return long-form population time series.
    """
    population = get_raw_spreadsheet(
        filepath, min_year=min_year, max_year=max_year,
    )
    if countries is not None:
        population = population.loc[population["country"].isin(countries)]

    # Wide to long
    population = population.set_index("country").stack(dropna=False).reset_index()
    population.columns = ["country", "year", "population"]
    population["year"] = population["year"].astype(int)
    population["time"] = pd.to_datetime(
        {"year": population["year"], "month": 6, "day": 15}
    )
    return population


def load_wb_cbr(filepath, year_min, extrapolate_to=None, countries=None):
    """
    Load World Bank crude birth rate from a CSV file.
    """
    raw = get_raw_spreadsheet(filepath, min_year=int(year_min))
    if countries is not None:
        raw = raw.loc[raw["country"].isin(countries)]

    # Wide to long
    wb = raw.set_index("country").stack(dropna=False).reset_index()
    wb.columns = ["country", "year", "wb_br"]
    wb["year"] = wb["year"].astype(int)
    wb = wb[["year", "wb_br"]].copy()

    if extrapolate_to is not None:
        wb = extrapolate_linear_tail(wb, "year", "wb_br", end_year=int(extrapolate_to))
    return wb


# ---------------------------------------------------------------------
# WHO national coverage loading
# ---------------------------------------------------------------------

def load_who_national(filepath, year_min, year_max=None, extrapolate_to=None):
    """
    Load WHO/UNICEF national coverage series.
    """
    who = pd.read_excel(filepath)
    who = who.loc[
        who["COVERAGE_CATEGORY_DESCRIPTION"] == "WHO/UNICEF Estimates of National Immunization Coverage"
    ].copy()

    who["YEAR"] = who["YEAR"].astype(int)
    who["COVERAGE"] = who["COVERAGE"] / 100.0

    mask = who["YEAR"] >= int(year_min)
    if year_max is not None:
        mask = mask & (who["YEAR"] <= int(year_max))
    who = who.loc[mask].copy()

    who = who[["YEAR", "COVERAGE"]].rename(columns={"YEAR": "year", "COVERAGE": "who_cov"})
    who = who.sort_values("year").reset_index(drop=True)

    if extrapolate_to is not None:
        who = extrapolate_linear_tail(who, "year", "who_cov", end_year=int(extrapolate_to))

    who["who_cov"] = clip01(who["who_cov"].astype(float))
    return who.set_index("year")["who_cov"]


# ---------------------------------------------------------------------
# Generic extrapolation with a reference series
# ---------------------------------------------------------------------

def clip01(x):
    return np.clip(x, 0.0, 1.0)

def extrapolate_linear_tail(df, year_col, value_col, end_year, k=5):
    """
    Linearly extrapolate a series to end_year using last k observed points.
    Returns a dataframe with observed + future rows, sorted by year.
    """
    out = df[[year_col, value_col]].copy()
    out[year_col] = out[year_col].astype(int)

    obs = out.dropna(subset=[value_col]).sort_values(year_col)
    if obs.empty:
        return out

    y_max = int(obs[year_col].max())
    if end_year <= y_max:
        return obs.reset_index(drop=True)

    tail = obs.tail(k)
    if len(tail) < 2:
        return obs.reset_index(drop=True)

    x = tail[year_col].values.astype(float)
    y = tail[value_col].values.astype(float)
    slope, intercept = np.polyfit(x, y, 1)

    future_years = np.arange(y_max + 1, end_year + 1, dtype=int)
    future_vals = intercept + slope * future_years

    future = pd.DataFrame({year_col: future_years, value_col: future_vals})
    return pd.concat([obs, future], axis=0).sort_values(year_col).reset_index(drop=True)


def compute_window_ratio_to_ref(
    sdf, ref_df, window=5, side="back",
    year_col="year", value_col="value", var_col=None,
    ref_col="ref", eps=1e-12,
):
    """
    Compute geometric-mean ratio of value_col / ref_col over a window
    of observed years from either the beginning ('back') or end ('forward')
    of the observed range. When var_col is provided, uses inverse-variance weighting.
    """
    d = sdf[[year_col, value_col] + ([var_col] if var_col and var_col in sdf.columns else [])].copy()
    d = d.dropna(subset=[year_col, value_col])
    d[year_col] = d[year_col].astype(int)

    r = ref_df[[year_col, ref_col]].copy()
    r[year_col] = r[year_col].astype(int)

    d = d.merge(r, on=year_col, how="left").dropna(subset=[ref_col])
    d = d.loc[(d[value_col] > eps) & (d[ref_col] > eps)].copy()
    if d.empty:
        return np.nan

    d = d.sort_values(year_col)
    n = min(int(window), len(d))
    dw = d.iloc[:n] if side == "back" else d.iloc[-n:]

    log_ratios = np.log(dw[value_col].to_numpy(float) / dw[ref_col].to_numpy(float))

    # Inverse-variance weighting if variance column is available
    if var_col and var_col in dw.columns:
        w = 1.0 / np.clip(dw[var_col].to_numpy(float), 1e-6, 1e6)
        return float(np.exp(np.average(log_ratios, weights=w)))

    return float(np.exp(log_ratios.mean()))


def extend_with_ref_window_ratios(
    df, ref_df, start_year, end_year,
    window=5, group_col=None, year_col="year",
    value_col="value", var_col=None, ref_col="ref",
    clip=None, zero_eps=1e-12,
):
    """
    Extend a series forward/backward using a reference series (e.g. WHO 
    MCV or WB CBR).  Uses a geometric-mean ratio over a window of observed 
    years to scale the reference.
    """
    full_years = np.arange(int(start_year), int(end_year) + 1, dtype=int)
    ref_df = ref_df[[year_col, ref_col]].copy()
    ref_df[year_col] = ref_df[year_col].astype(int)

    def _is_missing(s):
        # NaN or effectively zero — both treated as "no estimate"
        miss = ~np.isfinite(s.astype(float))
        miss = miss | (s.astype(float).abs() <= zero_eps)
        return miss

    def _extend_one(sdf):
        sdf = sdf.copy()
        sdf[year_col] = sdf[year_col].astype(int)
        sdf = sdf.sort_values(year_col)

        obs = ~_is_missing(sdf[value_col])
        if obs.sum() == 0:
            sdf = (sdf.set_index(year_col).reindex(full_years)
                   .reset_index().rename(columns={"index": year_col}))
            return sdf

        y_first = int(sdf.loc[obs, year_col].min())
        y_last = int(sdf.loc[obs, year_col].max())

        ratio_back = compute_window_ratio_to_ref(
            sdf.loc[obs, :], ref_df=ref_df, window=window, side="back",
            year_col=year_col, value_col=value_col, var_col=var_col,
            ref_col=ref_col, eps=zero_eps,
        )
        ratio_fwd = compute_window_ratio_to_ref(
            sdf.loc[obs, :], ref_df=ref_df, window=window, side="forward",
            year_col=year_col, value_col=value_col, var_col=var_col,
            ref_col=ref_col, eps=zero_eps,
        )

        sdf = (sdf.set_index(year_col).reindex(full_years)
               .reset_index().rename(columns={"index": year_col}))
        sdf = sdf.merge(ref_df, on=year_col, how="left")

        miss = _is_missing(sdf[value_col])
        before = sdf[year_col] < y_first
        after = sdf[year_col] > y_last
        ok_ref = np.isfinite(sdf[ref_col]) & (sdf[ref_col].astype(float) > zero_eps)

        fill_back = miss & before & ok_ref & np.isfinite(ratio_back)
        fill_fwd = miss & after & ok_ref & np.isfinite(ratio_fwd)

        sdf.loc[fill_back, value_col] = ratio_back * sdf.loc[fill_back, ref_col].astype(float)
        sdf.loc[fill_fwd, value_col] = ratio_fwd * sdf.loc[fill_fwd, ref_col].astype(float)

        # Scale variance by ratio^2
        if var_col and var_col in sdf.columns:
            miss_var = _is_missing(sdf[var_col]) | sdf[var_col].isna()
            if fill_back.any():
                anchor_var_back = sdf.loc[sdf[year_col] == y_first, var_col]
                if not anchor_var_back.empty and pd.notna(anchor_var_back.iloc[0]):
                    v0 = float(anchor_var_back.iloc[0])
                    ref_at_first = ref_df.loc[ref_df[year_col] == y_first, ref_col]
                    if not ref_at_first.empty and float(ref_at_first.iloc[0]) > zero_eps:
                        r0 = float(ref_at_first.iloc[0])
                        scale_b = (sdf.loc[fill_back & miss_var, ref_col].astype(float) / r0) ** 2
                        sdf.loc[fill_back & miss_var, var_col] = v0 * scale_b.values

            if fill_fwd.any():
                anchor_var_fwd = sdf.loc[sdf[year_col] == y_last, var_col]
                if not anchor_var_fwd.empty and pd.notna(anchor_var_fwd.iloc[0]):
                    v1 = float(anchor_var_fwd.iloc[0])
                    ref_at_last = ref_df.loc[ref_df[year_col] == y_last, ref_col]
                    if not ref_at_last.empty and float(ref_at_last.iloc[0]) > zero_eps:
                        r1 = float(ref_at_last.iloc[0])
                        scale_f = (sdf.loc[fill_fwd & miss_var, ref_col].astype(float) / r1) ** 2
                        sdf.loc[fill_fwd & miss_var, var_col] = v1 * scale_f.values

        if clip is not None:
            lo, hi = clip
            sdf[value_col] = sdf[value_col].astype(float).clip(lo, hi)

        sdf = sdf.drop(columns=[ref_col])
        return sdf

    if group_col is None:
        return _extend_one(df)

    parts = []
    for g, sdf in df.groupby(group_col, sort=False):
        out = _extend_one(sdf)
        out[group_col] = g
        parts.append(out)

    out = pd.concat(parts, ignore_index=True)
    return out.sort_values([group_col, year_col]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Rescaling helpers
# ---------------------------------------------------------------------

def compute_scale_factors(ref, est, eps=1e-12):
    """
    Compute year-level scale factors: ref / est.
    Both should be Series indexed by year.
    """
    sf = (ref / est).replace([np.inf, -np.inf], np.nan)
    sf = sf.where(np.isfinite(sf))
    sf = sf.where(np.abs(est) > eps)
    return sf


def apply_year_scale(
    df, scale, year_col, mean_col, var_col=None,
    out_mean_col=None, out_var_col=None, clip=None,
):
    """
    Multiply mean_col (and optionally var_col by scale^2) using
    per-year scale factors from a Series indexed by year.
    """
    out = df.copy()
    out_mean_col = out_mean_col or mean_col
    out_var_col = out_var_col or var_col

    out["_scale_y"] = out[year_col].astype(int).map(scale).astype(float)
    m = out["_scale_y"].notna()

    out.loc[m, out_mean_col] = out.loc[m, mean_col].astype(float) * out.loc[m, "_scale_y"].astype(float)
    if var_col and var_col in out.columns and out_var_col:
        out.loc[m, out_var_col] = out.loc[m, var_col].astype(float) * (out.loc[m, "_scale_y"].astype(float) ** 2)

    if clip is not None:
        lo, hi = clip
        out[out_mean_col] = out[out_mean_col].astype(float).clip(lo, hi)

    return out.drop(columns=["_scale_y"])


# ---------------------------------------------------------------------
# Ratio-based extrapolation variance
# ---------------------------------------------------------------------

def compute_ratio_extrapolation_variance(
    df, ref_series, obs_by_group_year,
    year_col="year", group_col="province",
    cov_col="mcv_coverage", var_col="var_coverage",
    window=5, min_ratio_var=0.001, growth_rate=0.2,
):
    """
    For extrapolated years, inflate variance so it grows with distance
    from the observed range.

    Two components are combined (taking the max):
      1) Existing variance * (1 + growth_rate * distance)
      2) Ratio-based floor: Var(ratio) * ref(y)^2 * (1 + growth_rate * d)
         using the observed variability of the group/reference ratio.
    """
    df = df.copy()
    ref = ref_series.dropna().astype(float)

    for g, idx in df.groupby(group_col).groups.items():
        sdf = df.loc[idx].copy()
        sdf[year_col] = sdf[year_col].astype(int)

        group_obs = sorted(obs_by_group_year.get(g, []))
        if len(group_obs) == 0:
            continue

        y_first = int(min(group_obs))
        y_last = int(max(group_obs))

        # Compute ratio variability over observed window
        ratios = []
        for y in group_obs:
            if y in ref.index:
                row_mask = sdf[year_col] == y
                if not row_mask.any():
                    continue
                w = float(ref.loc[y])
                c = float(sdf.loc[row_mask, cov_col].iloc[0])
                if w > 0 and np.isfinite(c):
                    ratios.append(c / w)

        if len(ratios) < 2:
            var_ratio_fwd = min_ratio_var
            var_ratio_bwd = min_ratio_var
        else:
            ratios_fwd = ratios[-min(window, len(ratios)):]
            var_ratio_fwd = max(
                float(np.var(ratios_fwd, ddof=1)) if len(ratios_fwd) > 1 else 0.0,
                min_ratio_var,
            )
            ratios_bwd = ratios[:min(window, len(ratios))]
            var_ratio_bwd = max(
                float(np.var(ratios_bwd, ddof=1)) if len(ratios_bwd) > 1 else 0.0,
                min_ratio_var,
            )

        for i in idx:
            y = int(df.loc[i, year_col])

            if y > y_last:
                d = y - y_last
                growth = 1.0 + growth_rate * d

                existing = df.loc[i, var_col]
                grown = float(existing) * growth if (pd.notna(existing) and float(existing) > 0) else 0.0

                ref_y = float(ref.loc[y]) if y in ref.index else 0.0
                ratio_floor = var_ratio_fwd * (ref_y ** 2) * growth if ref_y > 0 else 0.0

                df.loc[i, var_col] = max(grown, ratio_floor)

            elif y < y_first:
                d = y_first - y
                growth = 1.0 + growth_rate * d

                existing = df.loc[i, var_col]
                grown = float(existing) * growth if (pd.notna(existing) and float(existing) > 0) else 0.0

                ref_y = float(ref.loc[y]) if y in ref.index else 0.0
                ratio_floor = var_ratio_bwd * (ref_y ** 2) * growth if ref_y > 0 else 0.0

                df.loc[i, var_col] = max(grown, ratio_floor)

    if "std" in df.columns:
        df["std"] = np.sqrt(df[var_col].clip(lower=0))
    return df


# ---------------------------------------------------------------------
# Population-share (subregion) weighting
# ---------------------------------------------------------------------

def compute_subregion_popshares(dist, group_col="province", year_col="year",
                                weight_col="weight"):
    """
    Compute each subregion's share of total weight per year.
    """
    tmp = dist.copy()
    tmp[year_col] = tmp[year_col].astype(int)
    grp_w = (
        tmp.groupby([group_col, year_col], as_index=False)[weight_col]
        .sum().rename(columns={weight_col: "_grp_weight"})
    )
    grp_w["popshare"] = grp_w.groupby(year_col)["_grp_weight"].transform(
        lambda x: x / x.sum() if x.sum() != 0 else np.nan
    )
    return grp_w[[group_col, year_col, "popshare"]]


def extend_popshares(popshares, year_min, year_max,
                     group_col="province", year_col="year",
                     popshare_col="popshare"):
    """
    Extend popshares to cover [year_min, year_max] by forward-filling
    and back-filling each group's share from the nearest observed year.
    """
    full_years = np.arange(int(year_min), int(year_max) + 1, dtype=int)
    parts = []
    for g, sdf in popshares.groupby(group_col, sort=False):
        sdf = sdf.copy()
        sdf[year_col] = sdf[year_col].astype(int)
        sdf = sdf.set_index(year_col).reindex(full_years)
        sdf[popshare_col] = sdf[popshare_col].ffill().bfill()
        sdf[group_col] = g
        sdf = sdf.reset_index().rename(columns={"index": year_col})
        parts.append(sdf[[group_col, year_col, popshare_col]])
    return pd.concat(parts, ignore_index=True)


def aggregate_to_national(
    df, popshares, group_col="province", year_col="year",
    cov_col="mcv_coverage", var_col="var_coverage",
    popshare_col="popshare",
):
    """
    Aggregate subregion-level coverage to national level, renormalizing
    popshares over only subregions with non-NaN coverage each year.
    """
    merged = df.merge(popshares, on=[group_col, year_col], how="left")
    merged[popshare_col] = merged[popshare_col].astype(float)

    has_cov = merged[cov_col].notna()
    merged["_ps_adj"] = np.where(has_cov, merged[popshare_col], 0.0)
    merged["_ps_adj"] = merged.groupby(year_col)["_ps_adj"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0.0
    )

    merged["_w_cov"] = merged[cov_col].fillna(0) * merged["_ps_adj"]
    merged["_w_var"] = merged[var_col].fillna(0) * (merged["_ps_adj"] ** 2)

    nat = (
        merged.groupby(year_col, as_index=False)
        .agg(nat_est=("_w_cov", "sum"), nat_var=("_w_var", "sum"))
    )
    nat["nat_est"] = nat["nat_est"].astype(float)
    nat["nat_std"] = np.sqrt(nat["nat_var"].clip(lower=0))
    return nat


# ---------------------------------------------------------------------
# Census-based additive territory shares
# ---------------------------------------------------------------------

def compute_additive_territory_popshares(
    touchpoints, year_min, year_max,
):
    """
    Compute population shares for territories NOT covered by the World Bank
    national series (e.g. AJK, GB), expressed as fractions of the WB population.

    Touchpoint values are percentages of the WB population, so we just
    divide by 100 and interpolate between census years.

    These shares are meant to be used alongside DHS-based popshares for
    the main (WB) provinces, which sum to 1.0.  The total across all
    provinces is then 1.0 + sum(additive shares).
    """
    tp = pd.DataFrame(touchpoints).T.sort_index() / 100.0
    years = np.arange(int(year_min), int(year_max) + 1, dtype=int)
    frac = tp.reindex(years).interpolate(method="linear").ffill().bfill()
    frac = frac.fillna(0.0)

    frac.index.name = "year"
    out = frac.reset_index().melt(id_vars="year", var_name="province", value_name="popshare")
    out["year"] = out["year"].astype(int)
    return out.sort_values(["province", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Monthly interpolation
# ---------------------------------------------------------------------

def interpolate_yearly_to_monthly(
    df_yearly, group_col="province", year_col="year", time_col="time",
    cov_cols=("mcv_coverage",), var_cols=("var_coverage",),
    std_suffix="_std", start_year=None, end_year=None, anchor_month=1,
):
    """
    Linearly interpolate yearly series to monthly resolution.
    """
    out_parts = []
    y_min = int(df_yearly[year_col].min()) if start_year is None else int(start_year)
    y_max = int(df_yearly[year_col].max()) if end_year is None else int(end_year)
    monthly_time = pd.date_range(start=f"{y_min}-01-01", end=f"{y_max}-12-01", freq="MS")
    cov_cols = list(cov_cols or [])
    var_cols = list(var_cols or [])

    for g, sdf in df_yearly.groupby(group_col, sort=False):
        sdf = sdf.copy()
        sdf[year_col] = sdf[year_col].astype(int)
        sdf = sdf.sort_values(year_col)
        sdf[time_col] = pd.to_datetime(sdf[year_col].astype(str) + f"-{int(anchor_month):02d}-01")
        sdf = sdf.set_index(time_col).reindex(monthly_time)
        sdf[group_col] = g

        for c in cov_cols:
            if c in sdf.columns:
                sdf[c] = sdf[c].astype(float).interpolate(method="time").clip(0.0, 1.0)
        for v in var_cols:
            if v in sdf.columns:
                sdf[v] = sdf[v].astype(float).interpolate(method="time").clip(lower=0.0)
                sdf[f"{v}{std_suffix}"] = np.sqrt(sdf[v])

        out_parts.append(sdf.reset_index().rename(columns={"index": time_col}))

    out = pd.concat(out_parts, ignore_index=True)
    return out.sort_values([group_col, time_col]).reset_index(drop=True)