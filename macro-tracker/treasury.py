"""US Treasury daily par yield curve: loading and calculations (no plotting).

Source of truth is Treasury.gov's own Daily Treasury Par Yield Curve Rates feed
(published same day, unlike FRED's DGS series which lag). FRED is only a fallback.

Loading:  fetch_treasury_history() / load_yield_curve()
Maths:    changes(), curve_on(), compute_spreads(), target_date(), asof_value()
All yields are in percent; all changes and spreads are in basis points.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import date

import pandas as pd
import pyfredapi as pf
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TREASURY_CSV_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
)
USER_AGENT = "macro-tracker/1.0 (Streamlit dashboard; Treasury par yield curve loader)"
TIMEOUT = (5, 25)  # (connect, read) seconds
CACHE_TTL = 1800  # 30 minutes
EARLIEST_YEAR = 1990  # first year with the full 1M-30Y curve is well after this; NaN fills gaps

SOURCE_TREASURY = "US Treasury (par yield curve)"
SOURCE_FRED = "FRED (DGS series) - fallback"

# FRED fallback: series id -> normalized tenor label
FRED_SERIES = {
    "DGS1MO": "1M",
    "DGS3MO": "3M",
    "DGS6MO": "6M",
    "DGS1": "1Y",
    "DGS2": "2Y",
    "DGS3": "3Y",
    "DGS5": "5Y",
    "DGS7": "7Y",
    "DGS10": "10Y",
    "DGS20": "20Y",
    "DGS30": "30Y",
}

# name -> (short leg, long leg); spread = long - short, in bp
SPREADS = {  # display order
    "3M10Y": ("3M", "10Y"),
    "2s5s": ("2Y", "5Y"),
    "2s10s": ("2Y", "10Y"),
    "5s30s": ("5Y", "30Y"),
    "10s30s": ("10Y", "30Y"),
}

HORIZONS = ("1D", "1W", "1M", "3M", "YTD")


class TreasuryError(RuntimeError):
    """Treasury.gov could not be loaded or returned an unexpected format."""


@dataclass(frozen=True)
class CurveData:
    df: pd.DataFrame  # date index, columns in maturity order; NaN where no data
    source: str
    fallback: bool
    error: str | None  # why Treasury.gov was not used, if it wasn't

    @property
    def as_of(self) -> pd.Timestamp | None:
        return None if self.df.empty else self.df.index[-1]


# ------------------------------------------------------------------ tenor labels

_TENOR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(months?|mo|m|years?|yr|y)\s*$", re.IGNORECASE)


# Tenors the feed publishes but this dashboard deliberately does not show.
EXCLUDED_TENORS = frozenset({"1.5M"})


def drop_excluded(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c in EXCLUDED_TENORS])


def tenor_years(label: str) -> float | None:
    """Maturity in years from any label the feed uses ('1 Mo', '1.5 Month', '10 Yr', '3M', '2Y')."""
    match = _TENOR_RE.match(str(label))
    if not match:
        return None
    n, unit = float(match.group(1)), match.group(2).lower()
    return n / 12 if unit.startswith("m") else n


def tenor_label(label: str) -> str | None:
    """Normalized label: '1 Mo' -> '1M', '1.5 Month' -> '1.5M', '10 Yr' -> '10Y'."""
    years = tenor_years(label)
    if years is None:
        return None
    return f"{round(years * 12, 6):g}M" if years < 1 else f"{years:g}Y"


# ----------------------------------------------------------------------- parsing


def parse_par_yield_csv(text: str) -> pd.DataFrame:
    """Parse one Treasury CSV response into a date-indexed frame in maturity order.

    Tenors are discovered from the header (the feed has added 2M, 4M and 1.5M over
    the years), blanks stay NaN, and nothing is filled or interpolated.
    """
    if not text.strip():
        return pd.DataFrame()
    raw = pd.read_csv(io.StringIO(text))
    if "Date" not in raw.columns:
        raise TreasuryError(f"unexpected Treasury CSV format; columns: {list(raw.columns)[:6]}")

    out = pd.DataFrame({"date": pd.to_datetime(raw["Date"], format="%m/%d/%Y")})
    tenors = {}
    for col in raw.columns:
        years, label = tenor_years(col), tenor_label(col)
        if years is not None:
            tenors[label] = (years, pd.to_numeric(raw[col], errors="coerce"))
    if not tenors:
        raise TreasuryError("no tenor columns found in Treasury CSV")

    for label, (_, values) in sorted(tenors.items(), key=lambda kv: kv[1][0]):
        out[label] = values
    out = out.set_index("date").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.dropna(how="all")


def maturity_order(columns) -> list[str]:
    return sorted(columns, key=lambda c: tenor_years(c) if tenor_years(c) is not None else 1e9)


# ----------------------------------------------------------------------- loading


def build_session() -> requests.Session:
    """Session with a proper User-Agent and retry/backoff on transient failures."""
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch_treasury_year(year: int, session: requests.Session | None = None) -> pd.DataFrame:
    session = session or build_session()
    params = {
        "type": "daily_treasury_yield_curve",
        "field_tdr_date_value": year,
        "page": "",
        "_format": "csv",
    }
    try:
        resp = session.get(TREASURY_CSV_URL.format(year=year), params=params, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise TreasuryError(f"{year}: {type(exc).__name__}") from exc
    return parse_par_yield_csv(resp.text)


def fetch_treasury_history(start_year: int, end_year: int | None = None) -> pd.DataFrame:
    """Concatenate year-by-year pulls. Any failed year raises, so a gap is never silent."""
    end_year = end_year or date.today().year
    session = build_session()
    frames = [fetch_treasury_year(y, session) for y in range(start_year, end_year + 1)]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise TreasuryError("Treasury returned no observations")
    df = pd.concat(frames).sort_index()
    df = drop_excluded(df[~df.index.duplicated(keep="last")])
    return df[maturity_order(df.columns)]  # missing tenors in early years stay NaN


def fetch_fred_curve(api_key: str, start_year: int) -> pd.DataFrame:
    """FRED fallback: the DGS series, one column per tenor, NaN on holidays."""
    cols = {}
    for series_id, label in FRED_SERIES.items():
        raw = pf.get_series(series_id=series_id, api_key=api_key)
        s = pd.to_numeric(
            raw.assign(date=pd.to_datetime(raw["date"])).set_index("date")["value"],
            errors="coerce",
        )
        cols[label] = s
    df = pd.DataFrame(cols).sort_index()
    df = df.loc[f"{start_year}-01-01":].dropna(how="all")
    if df.empty:
        raise TreasuryError("FRED returned no observations")
    return df[maturity_order(df.columns)]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _cached_treasury(start_year: int) -> pd.DataFrame:
    return fetch_treasury_history(start_year)  # raises on failure, so failures are not cached


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _cached_fred(api_key: str, start_year: int) -> pd.DataFrame:
    return fetch_fred_curve(api_key, start_year)


def clear_curve_cache() -> None:
    """Manual refresh: drops only this module's caches, not the rest of the app's."""
    _cached_treasury.clear()
    _cached_fred.clear()


def load_yield_curve(years: int = 5, api_key: str | None = None) -> CurveData:
    """Load `years` of history (whole calendar years, so at least that much).

    Treasury.gov first; on any failure, FRED if a key is available. Never raises:
    if nothing works the returned frame is empty and `error` says why.
    """
    start_year = date.today().year - years
    try:
        # drop_excluded again here so a stale cached frame can never show an excluded tenor
        return CurveData(drop_excluded(_cached_treasury(start_year)), SOURCE_TREASURY, False, None)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"

    if api_key:
        try:
            return CurveData(drop_excluded(_cached_fred(api_key, start_year)), SOURCE_FRED, True, reason)
        except Exception as exc:  # message omitted: FRED errors can echo the request URL
            reason += f"; FRED fallback also failed ({type(exc).__name__})"
    return CurveData(pd.DataFrame(), "unavailable", True, reason)


# ------------------------------------------------------------------ calculations


def asof_value(series: pd.Series, target: pd.Timestamp | None) -> float:
    """Last observation at or before `target` (trading-day aware). NaN if none exists.

    NaNs in the series are skipped, never filled: a tenor that did not exist yet
    returns NaN, and a missing day falls back to the previous real observation.
    """
    if target is None:
        return float("nan")
    s = series.dropna().loc[:target]
    return float(s.iloc[-1]) if len(s) else float("nan")


_OFFSET_RE = re.compile(r"^(\d+)([DWMY])$")
_UNIT_NAMES = {"D": "day", "W": "week", "M": "month", "Y": "year"}
_UNIT_OFFSETS = {"D": "days", "W": "weeks", "M": "months", "Y": "years"}


def target_date(index: pd.DatetimeIndex, latest: pd.Timestamp, horizon: str) -> pd.Timestamp | None:
    """Reference date for a horizon. The caller takes the last observation at/before it.

    Horizons: '1D' (previous trading day), 'YTD' (end of the prior year), or any
    '<n><D|W|M|Y>' calendar offset such as '2W', '2M', '6M', '2Y'.
    """
    if horizon == "1D":  # previous trading day = previous date in the data
        earlier = index[index < latest]
        return earlier[-1] if len(earlier) else None
    if horizon == "YTD":  # last observation of the prior year
        return pd.Timestamp(latest.year - 1, 12, 31)
    match = _OFFSET_RE.match(horizon)
    if not match or int(match.group(1)) < 1:
        raise ValueError(f"unknown horizon {horizon!r}")
    n, unit = int(match.group(1)), match.group(2)
    return latest - pd.DateOffset(**{_UNIT_OFFSETS[unit]: n})


def horizon_label(horizon: str) -> str:
    """'2M' -> '2 months ago', '1D' -> '1 day ago', 'YTD' -> 'Prior year-end'."""
    if horizon == "YTD":
        return "Prior year-end"
    match = _OFFSET_RE.match(horizon)
    if not match:
        raise ValueError(f"unknown horizon {horizon!r}")
    n, unit = int(match.group(1)), match.group(2)
    return f"{n} {_UNIT_NAMES[unit]}{'s' if n != 1 else ''} ago"


def changes(df: pd.DataFrame, horizons=HORIZONS, scale: float = 100.0) -> pd.DataFrame:
    """Latest level plus change vs each horizon, one row per column of `df`.

    scale=100 turns a percent-yield difference into bp; use scale=1 for series that
    are already in bp (spreads). No lookahead: reference observations are always on or
    before the reference date and strictly before the latest date.
    """
    if df.empty:
        return pd.DataFrame()
    latest_date = df.index[-1]
    latest = df.iloc[-1]
    out = pd.DataFrame({"Latest": latest})
    for h in horizons:
        target = target_date(df.index, latest_date, h)
        prior = pd.Series({c: asof_value(df[c], target) for c in df.columns})
        out[h] = (latest - prior) * scale
    out.index.name = "Tenor"
    return out


def curve_on(df: pd.DataFrame, target: pd.Timestamp | None) -> tuple[pd.Timestamp, pd.Series] | None:
    """The whole curve as it stood on the last trading day at or before `target`."""
    if target is None:
        return None
    rows = df.loc[:target]
    return (rows.index[-1], rows.iloc[-1]) if len(rows) else None


def compute_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Curve spreads in bp (long - short). NaN wherever either leg is missing."""
    out = pd.DataFrame(index=df.index)
    for name, (short, long_) in SPREADS.items():
        if short in df.columns and long_ in df.columns:
            out[name] = (df[long_] - df[short]) * 100
        else:
            out[name] = float("nan")
    return out


def slice_lookback(data: pd.DataFrame | pd.Series, lookback: str):
    """1Y / 3Y / 5Y / Max, measured back from the latest observation."""
    if lookback == "Max" or len(data) == 0:
        return data
    years = int(lookback.rstrip("Y"))
    return data.loc[data.index[-1] - pd.DateOffset(years=years):]
