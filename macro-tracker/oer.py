"""Regional Owners' Equivalent Rent (OER) data and analytics.

fetch_oer_levels()  -> regional OER index levels from FRED (feeds the "CPI Index Level 3" tab)
compute_oer()       -> MoM % and weighted contributions, all derived from those levels
averages_table()    -> trailing 3/6/12-month average MoM and annualized rates
estimate_next()     -> next-print estimate using the six-month survey panel pattern
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


NATIONAL_SERIES = "CUUR0000SEHC"  # published US city average OER, for cross-checking

# BLS publishes OER of residences (SEHC) and OER of primary residence (SEHC01).
TICKER_SETS = {"SEHC": "", "SEHC01": "01"}


def series_id(region: str, suffix: str = "") -> str:
    return REGIONS[region]["series"] + suffix


def _month_series(series_id_: str, api_key: str) -> pd.Series:
    raw = pf.get_series(series_id=series_id_, api_key=api_key)
    s = pd.to_numeric(
        raw.assign(date=pd.to_datetime(raw["date"])).set_index("date")["value"],
        errors="coerce",
    ).dropna()
    s.index = s.index.to_period("M").to_timestamp()
    return s


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_oer_levels(api_key: str, suffix: str = "") -> pd.DataFrame:
    """Regional OER index levels, one column per region, indexed by month start.

    `suffix` selects the ticker family ("" = ...SEHC, "01" = ...SEHC01).
    Starts one month before START so the first MoM change is available.
    """
    cols = {region: _month_series(series_id(region, suffix), api_key) for region in REGIONS}
    levels = pd.DataFrame(cols).sort_index()
    levels.index.name = "month"
    return levels.loc[START - pd.DateOffset(months=1):]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_national_level(api_key: str, suffix: str = "") -> pd.Series:
    """Published national OER index, to cross-check any regional blend against."""
    return _month_series(NATIONAL_SERIES + suffix, api_key)


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


def _fit_beta(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Through-the-origin OLS slope of y on x, with its t-statistic."""
    sxx = float((x * x).sum())
    if len(x) < 24 or sxx == 0:
        return 0.0, float("nan")
    beta = float((x * y).sum() / sxx)
    resid = y - beta * x
    se = (float((resid**2).sum()) / (len(x) - 1) / sxx) ** 0.5
    return beta, (beta / se if se else float("nan"))


def estimate_next(levels: pd.DataFrame, mom: pd.DataFrame, backtest_months: int = 60):
    """Estimate the next monthly OER print for each region.

    BLS prices each rental unit once every six months, and rents mostly reset once a
    year, so a month that runs hot tends to be followed six months later by a
    cooler one. The model captures that directly:

        Est. MoM = 12M trend  +  beta x (surprise six months earlier)

    * 12M trend  = average MoM of the last 12 months (a full year, so no seasonal
      bias; deliberately not the 6-month average).
    * surprise   = MoM of the month six back minus the 12-month trend that stood
      before it.
    * beta       = pooled slope of surprise on the surprise six months earlier,
      fit across all four regions. It is estimated, not assumed: a negative beta
      means "hot then cold".

    Returns (target month, table, info) where `info` carries beta, its t-stat,
    the source month and a backtest of the model against trend alone.
    """
    regions = list(REGIONS)
    m = mom[regions].dropna()
    if len(m) < 12 + 6 + 12:
        return None, pd.DataFrame(), {}

    latest = m.index[-1]
    target = latest + pd.DateOffset(months=1)
    source = target - pd.DateOffset(months=6)

    trend_hist = m.rolling(12).mean().shift(1)  # what was known before each month
    dev = m - trend_hist
    pairs = pd.concat(
        [pd.DataFrame({"x": dev[r].shift(6), "y": dev[r]}).dropna() for r in regions]
    )
    beta, tstat = _fit_beta(pairs["x"], pairs["y"])

    # Backtest: refit beta using only data before each month, then score the call.
    model_err, trend_err = [], []
    tested = sorted(pairs.index.unique())[-backtest_months:]
    for t in tested:
        train, test = pairs[pairs.index < t], pairs[pairs.index == t]
        b, _ = _fit_beta(train["x"], train["y"])
        model_err += list((test["y"] - b * test["x"]).abs())
        trend_err += list(test["y"].abs())
    info = {
        "beta": beta,
        "tstat": tstat,
        "source": source,
        "backtest_months": len(tested),
        "mae_model": float(pd.Series(model_err).mean()) if model_err else float("nan"),
        "mae_trend": float(pd.Series(trend_err).mean()) if trend_err else float("nan"),
    }

    trend = m.tail(12).mean()
    surprise = dev.loc[source] if source in dev.index else pd.Series(float("nan"), index=regions)
    surprise = surprise.fillna(0.0)
    adj = beta * surprise
    est = trend + adj

    cpi_w = pd.Series({r: REGIONS[r]["cpi_weight"] / 100 for r in regions})
    oer_s = pd.Series({r: REGIONS[r]["oer_share"] / 100 for r in regions})
    last_level = levels.loc[latest, regions]
    est_level = last_level * (1 + est / 100)

    out = pd.DataFrame(
        {
            "12M Trend": trend,
            "Surprise 6M Ago": surprise,
            "Adj. (β × surprise)": adj,
            "Est. MoM %": est,
            "Last Level": last_level,
            "Est. Level": est_level,
            "Est. Contribution to CPI (pp)": est * cpi_w,
        }
    )
    # National = share-weighted blend of the regions, levels included.
    out.loc[NATIONAL] = [
        (trend * oer_s).sum(),
        (surprise * oer_s).sum(),
        (adj * oer_s).sum(),
        (est * oer_s).sum(),
        (last_level * oer_s).sum(),
        (est_level * oer_s).sum(),
        (est * cpi_w).sum(),
    ]
    out.index.name = "Region"
    return target, out, info


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
