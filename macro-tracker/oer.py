"""Regional Owners' Equivalent Rent (OER) data and analytics.

fetch_oer_levels()  -> regional OER index levels from FRED (feeds the "CPI Index Level 3" tab)
compute_oer()       -> MoM % and weighted contributions, all derived from those levels
averages_table()    -> trailing 3/6/12-month average MoM and annualized rates
rankings_table()    -> regions ranked by latest MoM and by contribution
"""

import pandas as pd
import pyfredapi as pf
import streamlit as st

START = pd.Timestamp("2013-07-01")

# series: FRED id; cpi_weight: relative importance in headline CPI (% of CPI);
# oer_share: share of national OER (%).
REGIONS = {
    "Northeast": dict(series="CUUR0100SEHC", cpi_weight=5.148, oer_share=19.60),
    "Midwest": dict(series="CUUR0200SEHC", cpi_weight=4.717, oer_share=18.00),
    "South": dict(series="CUUR0300SEHC", cpi_weight=9.247, oer_share=35.30),
    "West": dict(series="CUUR0400SEHC", cpi_weight=7.092, oer_share=27.10),
}

NATIONAL = "National OER"
WINDOWS = (3, 6, 12)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_oer_levels(api_key: str) -> pd.DataFrame:
    """Regional OER index levels, one column per region, indexed by month start.

    Starts one month before START so the first MoM change is available.
    """
    cols = {}
    for region, meta in REGIONS.items():
        raw = pf.get_series(series_id=meta["series"], api_key=api_key)
        s = pd.to_numeric(
            raw.assign(date=pd.to_datetime(raw["date"])).set_index("date")["value"],
            errors="coerce",
        ).dropna()
        s.index = s.index.to_period("M").to_timestamp()
        cols[region] = s
    levels = pd.DataFrame(cols).sort_index()
    levels.index.name = "month"
    return levels.loc[START - pd.DateOffset(months=1):]


def compute_oer(levels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Everything on the OER tab is derived from `levels`, so the two tabs stay linked.

    Returns:
      mom          MoM % per region, plus the share-weighted national OER MoM
      contrib_cpi  contribution to headline CPI MoM, in pp (MoM x CPI weight / 100)
      contrib_oer  contribution to national OER MoM, in pp (MoM x OER share / 100)
    """
    regions = list(REGIONS)
    cpi_w = pd.Series({r: REGIONS[r]["cpi_weight"] / 100 for r in regions})
    oer_s = pd.Series({r: REGIONS[r]["oer_share"] / 100 for r in regions})

    mom = levels[regions].pct_change(fill_method=None) * 100

    contrib_cpi = mom * cpi_w
    contrib_cpi["Total"] = contrib_cpi.sum(axis=1, min_count=len(regions))

    contrib_oer = mom * oer_s
    contrib_oer[NATIONAL] = contrib_oer.sum(axis=1, min_count=len(regions))

    mom[NATIONAL] = contrib_oer[NATIONAL]
    return {
        "mom": mom.iloc[1:],
        "contrib_cpi": contrib_cpi.iloc[1:],
        "contrib_oer": contrib_oer.iloc[1:],
    }


def averages_table(mom: pd.DataFrame) -> pd.DataFrame:
    """Trailing average MoM and annualized (compounded) rate over 3/6/12 months."""
    mom = mom.dropna(how="all")
    out = pd.DataFrame(index=mom.columns)
    for n in WINDOWS:
        tail = mom.tail(n)
        complete = tail.notna().all()
        avg = tail.mean().where(complete)
        ann = (((1 + tail / 100).prod()) ** (12 / n) - 1) * 100
        out[f"{n}M Avg MoM"] = avg
        out[f"{n}M Annualized"] = ann.where(complete)
    out.index.name = "Region"
    return out


def estimate_next(levels: pd.DataFrame, mom: pd.DataFrame, years: int = 5):
    """Statistical estimate of the next monthly OER print for each region.

    Est. MoM = trailing 6-month average MoM (run-rate)
             + seasonal adjustment for the target calendar month.

    The seasonal adjustment is how much that calendar month has typically run above
    or below its own year's average, over the last `years` complete calendar years
    (needs at least 3, otherwise it is 0). The series are not seasonally adjusted,
    so this keeps e.g. a typically soft January from being treated as a trend change.
    """
    regions = list(REGIONS)
    m = mom[regions].dropna()
    if len(m) < 6:
        return None, pd.DataFrame()

    latest = m.index[-1]
    target = latest + pd.DateOffset(months=1)
    run_rate = m.tail(6).mean()

    full_years = m.groupby(m.index.year).filter(lambda g: len(g) == 12)
    dev = full_years - full_years.groupby(full_years.index.year).transform("mean")
    same = dev[dev.index.month == target.month].tail(years)
    seasonal = same.mean() if len(same) >= 3 else pd.Series(0.0, index=regions)

    est = run_rate + seasonal
    cpi_w = pd.Series({r: REGIONS[r]["cpi_weight"] / 100 for r in regions})
    oer_s = pd.Series({r: REGIONS[r]["oer_share"] / 100 for r in regions})

    out = pd.DataFrame(
        {
            "6M Run-rate": run_rate,
            "Seasonal Adj.": seasonal,
            "Est. MoM %": est,
            "Last Level": levels.loc[latest, regions],
            "Est. Level": levels.loc[latest, regions] * (1 + est / 100),
            "Est. Contribution to CPI (pp)": est * cpi_w,
        }
    )
    out.loc[NATIONAL] = [
        (run_rate * oer_s).sum(),
        (seasonal * oer_s).sum(),
        (est * oer_s).sum(),
        float("nan"),
        float("nan"),
        (est * cpi_w).sum(),
    ]
    out.index.name = "Region"
    return target, out


def rankings_table(mom: pd.DataFrame, contrib_cpi: pd.DataFrame, contrib_oer: pd.DataFrame):
    """Regions ranked (1 = highest, ties share the mean rank) for the latest month."""
    regions = list(REGIONS)
    complete = mom[regions].dropna()
    if complete.empty:
        return None, pd.DataFrame()
    latest = complete.index[-1]

    out = pd.DataFrame(
        {
            "MoM %": mom.loc[latest, regions],
            "Contribution to CPI (pp)": contrib_cpi.loc[latest, regions],
            "Contribution to OER (pp)": contrib_oer.loc[latest, regions],
        }
    )
    ranked = pd.DataFrame(index=regions)
    ranked["MoM %"] = out["MoM %"]
    ranked["MoM Rank"] = out["MoM %"].round(9).rank(ascending=False, method="average")
    ranked["Contribution to CPI (pp)"] = out["Contribution to CPI (pp)"]
    ranked["CPI Contribution Rank"] = (
        out["Contribution to CPI (pp)"].round(9).rank(ascending=False, method="average")
    )
    ranked["Contribution to OER (pp)"] = out["Contribution to OER (pp)"]
    ranked["OER Contribution Rank"] = (
        out["Contribution to OER (pp)"].round(9).rank(ascending=False, method="average")
    )
    ranked.index.name = "Region"
    return latest, ranked.sort_values("MoM Rank")
