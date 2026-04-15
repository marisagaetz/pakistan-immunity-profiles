""" VisualizeInputs.py

Generates two figures per province (+ national):
  Figure 1: Data overview (cases, births, vaccination w/ SIAs, age onset)
  Figure 2: Age-at-infection distribution (overall + infant breakdown)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------

# Shared color palette
colors = ["#FF420E", "#0078ff", "#BF00BA", "xkcd:goldenrod",
          "#00ff07", "k", "#00BF05"]

def axes_setup(axes):
    """Standard formatting: offset left spine, hide top/right, light grid."""
    axes.spines["left"].set_position(("axes", -0.025))
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.grid(color="grey", alpha=0.2)
    return

# ------------------------------------------------------------
# Main script
# ------------------------------------------------------------

if __name__ == "__main__":

    # Loop over provinces + national (province=None)
    provinces = ["punjab", "sindh", "khyber pakhtunkhwa", "balochistan", None]

    # The linelist and SIA calendar use "kp" instead of the full name
    province_map = {"kp": "khyber pakhtunkhwa"}

    # Load datasets
    epi = pd.read_pickle("pickle_jar/semimonthly_prov_dataset.pkl")
    epi = epi.loc[epi["time"].dt.year >= 2009]
    epi["province"] = epi["province"].replace(province_map)

    linelist = pd.read_pickle("pickle_jar/combined_linelist_regressed.pkl")
    linelist["province"] = linelist["province"].str.lower().str.strip()
    linelist = linelist.sort_values("date_onset").reset_index(drop=True)

    sia_cal = pd.read_pickle("pickle_jar/sia_cal.pkl")

    # Per-province (+ national) loop
    for province in provinces:
        # Select the right epi time series
        if province is not None:
            sf = epi.set_index(["province", "time"]).copy()
            sf = sf.loc[province].sort_index()
            ll = linelist.loc[linelist["province"] == province].copy()
        else:
            # National dataset is pre-aggregated
            sf = pd.read_pickle("pickle_jar/national_dataset.pkl")
            sf = sf.loc[sf["time"].dt.year >= 2009]
            sf = sf.set_index("time")
            sf = sf.sort_index()
            ll = linelist.copy()

        # Age distribution from linelist
        # Weight each case by its confirmation probability
        ll = ll.loc[ll["age_months"].notna()].copy()
        ll["weight"] = ll["conf_prob"].fillna(ll["is_case"]).fillna(0.0)
        ll["age_years"] = (ll["age_months"] / 12.0).astype(int)

        # Bin into 0, 1, 2, … AGE_CAP years
        AGE_CAP = 14
        ll["age_bin"] = ll["age_years"].clip(upper=AGE_CAP)
        hist = ll.groupby("age_bin")["weight"].sum()
        hist = hist.reindex(range(AGE_CAP + 1), fill_value=0.0)

        # Normalize to percentage of all cases
        if hist.sum() > 0:
            hist = hist / hist.sum() * 100.0

        region_name = province.title() if province is not None else "Pakistan"
        print(f"{region_name}: age 0-1 = {hist.get(0, 0.0):.1f}%")

        # Infant (<1 yr) breakdown using INFANT_BIN_MONTHS
        infant_bin_edges = list(range(0, 13, 1))
        infant_labels = [
            f"{infant_bin_edges[i]}-{infant_bin_edges[i+1]}m"
            for i in range(len(infant_bin_edges) - 1)
        ]

        ll_infant = ll.loc[ll["age_months"] < infant_bin_edges[-1]].copy()
        ll_infant["infant_bin"] = pd.cut(
            ll_infant["age_months"], bins=infant_bin_edges,
            labels=infant_labels, right=False,
        )
        infant_hist = ll_infant.groupby("infant_bin", observed=False)["weight"].sum()
        infant_hist = infant_hist.reindex(infant_labels, fill_value=0.0)
        if infant_hist.sum() > 0:
            infant_hist = infant_hist / infant_hist.sum() * 100.0

        # SIA doses for this province
        if province is not None:
            # Map full province name → column name (KP special case)
            elig_name = province.replace("khyber pakhtunkhwa", "kp")
            elig_col = f"elig_{elig_name}"
            if elig_col in sia_cal.columns:
                sias = sia_cal[["time", "doses", elig_col, "total_elig_targeted"]].copy()
                # Pro-rate national doses by this province's share of eligible pop
                sias["plot_doses"] = sias["doses"] * (sias[elig_col] / sias["total_elig_targeted"])
            else:
                sias = pd.DataFrame(columns=["time", "plot_doses"])
        else:
            sias = sia_cal[["time", "doses"]].copy()
            sias["plot_doses"] = sias["doses"]

        sias = sias.loc[sias["plot_doses"] > 0].copy()

        # ---------------------------------------------------------------
        # FIGURE 1: Data overview (cases, births, vaccination, age inset)
        # ---------------------------------------------------------------
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 3)
        case_ax = fig.add_subplot(gs[0, :-1])
        demo_ax = fig.add_subplot(gs[1, :])
        vacc_ax = fig.add_subplot(gs[2, :])
        age_ax  = fig.add_subplot(gs[0, -1])
        axes = [case_ax, demo_ax, vacc_ax, age_ax]

        # Panel 0: Reported cases (semimonthly → monthly)
        sf_monthly_cases = sf[["cases"]].resample("MS").sum()

        axes_setup(axes[0])
        axes[0].fill_between(
            sf_monthly_cases.index, 0, sf_monthly_cases["cases"],
            facecolor=colors[0], edgecolor="None", alpha=0.6,
        )
        axes[0].set_ylim((0, None))
        axes[0].set_ylabel("Monthly cases")
        axes[0].legend(frameon=False, fontsize=12, loc="upper right")

        # Panel 1: Annual births (lightly smoothed step function) 
        births = sf[["births", "births_var"]].resample("YS").sum()
        births["err"] = np.sqrt(births["births_var"])

        axes_setup(axes[1])
        axes[1].fill_between(
            births.index,
            (births["births"] - 2.0 * births["err"]).values,
            (births["births"] + 2.0 * births["err"]).values,
            facecolor=colors[1], edgecolor="None", alpha=0.3,
            step="post",
        )
        axes[1].step(births.index, births["births"].values,
                     where="post", lw=3, color=colors[1])
        axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        axes[1].set_ylabel("Annual births")

        # Panel 2: MCV1/MCV2 coverage + SIA markers
        COVERAGE_HALFWIDTH = 0.05  # constant ±5pp uncertainty band

        mcv1 = sf[["mcv1"]].resample("MS").mean()
        mcv2 = sf[["mcv2"]].resample("MS").mean()
        mcv2 = mcv2.loc[mcv2["mcv2"] > 0]
        axes_setup(axes[2])

        # MCV1
        axes[2].fill_between(
            mcv1.index,
            np.clip(mcv1["mcv1"] - COVERAGE_HALFWIDTH, 0, 1).values,
            np.clip(mcv1["mcv1"] + COVERAGE_HALFWIDTH, 0, 1).values,
            facecolor=colors[2], edgecolor="None", alpha=0.2,
        )
        axes[2].plot(mcv1["mcv1"], lw=3, color=colors[2], label="MCV1")

        # MCV2
        MCV2_COLOR = "#2CBAB1"
        axes[2].fill_between(
            mcv2.index,
            (mcv2["mcv2"] - COVERAGE_HALFWIDTH).values,
            (mcv2["mcv2"] + COVERAGE_HALFWIDTH).values,
            facecolor=MCV2_COLOR, edgecolor="None", alpha=0.2,
        )
        axes[2].plot(mcv2["mcv2"], lw=3, color=MCV2_COLOR, label="MCV2")

        # SIA vertical markers on the vaccination panel
        SIA_COLOR = "#D4950D"
        if not sias.empty:
            sias["dose_frac"] = sias["plot_doses"] / sias["plot_doses"].max()
            y0, y1 = 0, 1.1

            # Only label SIAs above a province-specific dose threshold
            SIA_LABEL_MIN = {
                "punjab": 5e6,
                "sindh": 5e6,
                "khyber pakhtunkhwa": 5e6,
                "balochistan": 5e5,
                None: 2e7,
            }
            for _, r in sias.iterrows():
                if r["time"].year < 2009:
                    continue
                height = 0.35 * r["dose_frac"]
                axes[2].axvline(r["time"], ymax=height / y1,
                                color=SIA_COLOR, lw=2)
                # Annotate large SIAs with dose count
                if r["plot_doses"] >= SIA_LABEL_MIN.get(province, 5e5):
                    if r["plot_doses"] >= 1e6:
                        label = f"{r['plot_doses']/1e6:0.1f}M"
                    else:
                        label = f"{r['plot_doses']/1e3:0.0f}k"
                    axes[2].text(
                        r["time"], height, label,
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        color=SIA_COLOR, fontsize=16,
                    )
            # Invisible line just for the legend entry
            axes[2].plot([], color=SIA_COLOR, lw=2, label="SIA doses")

        axes[2].set_ylim((0, 1.1))
        axes[2].set_ylabel("Coverage")

        LEGEND_LOC = {
            "punjab": "center right",
            "sindh": "center right",
            "khyber pakhtunkhwa": "upper right",
            "balochistan": "upper right",
            None: "center right",
        }
        axes[2].legend(frameon=False, fontsize=14,
                       loc=LEGEND_LOC.get(province, "center right"))

        # Panel 3 (inset): Age-at-infection distribution 
        axes_setup(axes[3])
        axes[3].plot(hist.index, hist.values,
                     color=colors[0], lw=4, drawstyle="steps-post")
        axes[3].set_xticks(range(0, AGE_CAP + 1, 5))
        axes[3].set_ylim((0, None))
        axes[3].set_xlabel("Age")
        axes[3].set_ylabel("% of cases")
        axes[3].text(0.99, 0.95, "Infection Age",
                     horizontalalignment="right", verticalalignment="top",
                     fontsize=28, color=colors[0],
                     transform=axes[3].transAxes)

        # Title and panel labels 
        fig.suptitle(f"Data Overview in {region_name}", fontsize=30)
        fig.tight_layout(h_pad=0.0, w_pad=0, rect=[0, 0, 1, 0.95])

        axes[0].text(0, 0.95, "Reported Cases",
                     horizontalalignment="left", verticalalignment="top",
                     fontsize=28, color=colors[0],
                     transform=axes[0].transAxes)
        axes[1].text(0, 0.95, "Births",
                     horizontalalignment="left", verticalalignment="top",
                     fontsize=28, color=colors[1],
                     transform=axes[1].transAxes)
        axes[2].text(0, 0.95, "Vaccination",
                     horizontalalignment="left", verticalalignment="top",
                     fontsize=28, color=colors[2],
                     transform=axes[2].transAxes)

        plt.show()

        # ---------------------------------------------------------------
        # FIGURE 2: Age distribution — overall + infant (<1 yr) breakdown
        # ------------------------------------------------------------
        fig2, (ax_overall, ax_infant) = plt.subplots(
            1, 2, figsize=(10, 4.5),
            gridspec_kw={"width_ratios": [1, 1.2]},
        )

        # Left panel: overall (0–10 yr) age distribution
        OVERALL_AGE_CAP = 10
        hist_cropped = hist.reindex(range(OVERALL_AGE_CAP + 1), fill_value=0.0)

        axes_setup(ax_overall)
        ax_overall.plot(hist_cropped.index, hist_cropped.values,
                        color=colors[0], lw=4, drawstyle="steps-post")
        ax_overall.set_xticks(range(0, OVERALL_AGE_CAP + 1, 1))
        ax_overall.set_xlim((-0.5, OVERALL_AGE_CAP + 0.5))
        ax_overall.set_ylim((0, None))
        ax_overall.set_xlabel("Age (years)", fontsize=20)
        ax_overall.set_ylabel("% of cases <10", fontsize=18)
        ax_overall.text(0.95, 0.95, "Ages 0-10Y",
                        horizontalalignment="right", verticalalignment="top",
                        fontsize=28, color=colors[0],
                        transform=ax_overall.transAxes)

        # Right panel: infant (<1 yr) breakdown
        axes_setup(ax_infant)
        # x = bin edges; y has final value repeated so steps-post draws last bar
        plot_x = np.array(infant_bin_edges)
        plot_y = np.append(infant_hist.values, infant_hist.values[-1])
        ax_infant.plot(plot_x, plot_y,
                       color=colors[0], lw=4, drawstyle="steps-post")
        ax_infant.set_xticks(infant_bin_edges)
        ax_infant.set_xlim((-0.5, 12.5))
        ax_infant.set_ylim((0, None))
        ax_infant.set_xlabel("Age (months)", fontsize=20)
        ax_infant.set_ylabel("% of cases <1", fontsize=18)
        ax_infant.text(0.05, 0.95, "Ages 0-1Y",
                       horizontalalignment="left", verticalalignment="top",
                       fontsize=28, color=colors[0],
                       transform=ax_infant.transAxes)

        ax_overall.tick_params(axis="both", labelsize=18)
        ax_infant.tick_params(axis="both", labelsize=18)

        fig2.suptitle(f"Age at Infection — {region_name}", fontsize=24)
        fig2.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show() 