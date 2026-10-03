"""Data layer for the CPI nowcast beat/miss tracker.

fetch_fred_data()     -> actual month-over-month CPI change from FRED
fetch_nowcast_data()  -> Cleveland Fed nowcast (month-over-month % change)
process_beat_miss()   -> merge the two and classify BEAT / MISS / IN-LINE
"""

import pandas as pd
import pyfredapi as pf
import requests
import streamlit as st

NOWCAST_URL = (
    "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json"
)
START = pd.Timestamp("2013-07-01")
IN_LINE_THRESHOLD = 0.01  # percentage points

# metric label -> (FRED series id, series name inside the Cleveland Fed JSON)
METRICS = {
    "Headline CPI": ("CPIAUCSL", "CPI Inflation"),
    "Core CPI": ("CPILFESL", "Core CPI Inflation"),
}

LOOKBACK_YEARS = {"1Y": 1, "2Y": 2, "3Y": 3, "5Y": 5, "10Y": 10, "MAX": None}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fred_data(series_id: str, api_key: str) -> pd.DataFrame:
    """Actual month-over-month % change, indexed by month start, from July 2013.

    The full history is pulled so the first month (July 2013) has a prior
    month to compare against.
    """
    raw = pf.get_series(series_id=series_id, api_key=api_key)
    level = pd.to_numeric(
        raw.assign(date=pd.to_datetime(raw["date"])).set_index("date")["value"],
        errors="coerce",
    ).dropna()
    mom = (level.pct_change() * 100).loc[START:]
    out = mom.rename("actual").to_frame()
    out.index = out.index.to_period("M").to_timestamp()
    out.index.name = "month"
    return out


def _last_value(series: dict) -> float | None:
    """Last non-empty point of a FusionCharts series (the final pre-release nowcast)."""
    vals = [d["value"] for d in series["data"] if d.get("value") not in ("", None)]
    return float(vals[-1]) if vals else None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nowcast_data() -> pd.DataFrame:
    """Latest Cleveland Fed nowcast per target month, one column per metric.

    The nowcast page draws its charts from a JSON file; reading that file is
    more stable than parsing the rendered HTML table. Each chart object is one
    target month holding the daily nowcast path, so the last value is the
    most recent print for that month.
    """
    resp = requests.get(NOWCAST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()

    rows = []
    for chart in resp.json():
        by_name = {s["seriesname"]: s for s in chart["dataset"]}
        row = {"month": pd.Period(chart["chart"]["subcaption"], freq="M").to_timestamp()}
        for label, (_, json_name) in METRICS.items():
            row[label] = _last_value(by_name[json_name]) if json_name in by_name else None
        rows.append(row)

    return pd.DataFrame(rows).set_index("month").sort_index().loc[START:]


def _classify(spread: float) -> str:
    if pd.isna(spread):
        return "PENDING"  # nowcast exists, actual not yet released
    if abs(round(spread, 6)) <= IN_LINE_THRESHOLD:
        return "IN-LINE"
    return "BEAT" if spread > 0 else "MISS"


def process_beat_miss(actual: pd.DataFrame, nowcast: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Join actuals to nowcasts for one metric and compute spread and outcome."""
    df = nowcast[[metric]].rename(columns={metric: "nowcast"}).dropna()
    df = df.join(actual, how="left")  # left join keeps upcoming months with no actual yet
    df["spread"] = df["actual"] - df["nowcast"]
    df["beat_miss"] = df["spread"].map(_classify)
    return df


def filter_lookback(df: pd.DataFrame, lookback: str) -> pd.DataFrame:
    """Keep the last N full calendar years plus the current (partial) year."""
    years = LOOKBACK_YEARS[lookback]
    if years is None or df.empty:
        return df
    # Whole calendar years: 1Y = last full year plus the current year to date.
    return df[df.index.year >= df.index.max().year - years]
