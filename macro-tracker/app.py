"""Macroeconomic data tracker built on Streamlit and the FRED API."""

import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyfredapi as pf
from plotly.subplots import make_subplots
import requests
import streamlit as st

from nowcast import (
    LOOKBACK_YEARS,
    METRICS,
    fetch_fred_data,
    fetch_nowcast_data,
    filter_lookback,
    process_beat_miss,
)

FRED_API_ROOT = "https://api.stlouisfed.org/fred"

PRESETS = {
    "Real GDP": "GDPC1",
    "CPI (All Urban Consumers)": "CPIAUCSL",
    "Unemployment Rate": "UNRATE",
    "Federal Funds Rate": "FEDFUNDS",
    "10-Year Treasury Yield": "DGS10",
    "Industrial Production": "INDPRO",
    "Retail Sales": "RSAFS",
}

MONTHS = list(calendar.month_abbr)[1:]

# Heatmap rows, top to bottom. Each row gets its own color scale in every year.
HEATMAP_ROWS = [
    ("nowcast", "Nowcast"),
    ("actual", "Actual"),
    ("spread", "Spread (Actual − Nowcast)"),
]

st.set_page_config(page_title="Macro Tracker", page_icon="📈", layout="wide")


def get_api_key() -> str | None:
    """Read the FRED key from st.secrets without ever echoing it."""
    try:
        return st.secrets["fred"]["api_key"]
    except (KeyError, FileNotFoundError):
        return None


# ---------------------------------------------------------------- nowcast tab


def _color_axis(values: np.ndarray, is_spread: bool) -> dict:
    """Color scale fitted to one row of one year, so no two heatmaps share a scale."""
    finite = values[np.isfinite(values)]
    if is_spread:
        # Diverging, centred on zero: green = BEAT (actual hotter), red = MISS.
        limit = max(0.02, float(np.abs(finite).max())) if finite.size else 0.02
        return dict(colorscale="RdYlGn", cmin=-limit, cmax=limit, cmid=0, showscale=False)
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    if hi - lo < 0.02:
        hi = lo + 0.02
    return dict(colorscale="YlOrRd", cmin=lo, cmax=hi, showscale=False)


def build_year_heatmap(df: pd.DataFrame, year: int) -> go.Figure:
    """Jan–Dec heatmap for one calendar year: Nowcast, Actual and Spread rows,
    each with an independent color scale."""
    months = df[df.index.year == year].reindex(
        pd.date_range(f"{year}-01-01", periods=12, freq="MS")
    )

    fig = make_subplots(
        rows=len(HEATMAP_ROWS), cols=1, shared_xaxes=True, vertical_spacing=0.06
    )
    layout = {}
    for i, (col, label) in enumerate(HEATMAP_ROWS, start=1):
        values = months[col].to_numpy(dtype=float)
        axis = "coloraxis" if i == 1 else f"coloraxis{i}"
        fig.add_trace(
            go.Heatmap(
                z=[values],
                x=MONTHS,
                y=[label],
                coloraxis=axis,
                texttemplate="%{z:.2f}",
                hoverongaps=False,
            ),
            row=i,
            col=1,
        )
        layout[axis] = _color_axis(values, is_spread=(col == "spread"))

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        **layout,
    )
    fig.update_xaxes(showticklabels=False, type="category")
    fig.update_xaxes(showticklabels=True, side="top", row=1, col=1)
    return fig


def render_nowcast(api_key: str) -> None:
    c1, c2 = st.columns([1, 3])
    metric = c1.selectbox("Metric", list(METRICS))
    lookback = c2.radio("Lookback", list(LOOKBACK_YEARS), index=4, horizontal=True)

    try:
        with st.spinner("Loading nowcasts and CPI actuals…"):
            nowcast = fetch_nowcast_data()
            actual = fetch_fred_data(METRICS[metric][0], api_key)
    except requests.RequestException as exc:
        st.error(f"Data request failed ({type(exc).__name__}). Try again shortly.")
        return
    except Exception as exc:
        st.error(f"Could not load data: {type(exc).__name__}")
        return

    df = filter_lookback(process_beat_miss(actual, nowcast, metric), lookback)
    released = df[df["beat_miss"] != "PENDING"]
    pending = len(df) - len(released)

    if released.empty:
        st.warning("No released months in this window yet.")
        return

    counts = released["beat_miss"].value_counts()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("BEAT", int(counts.get("BEAT", 0)))
    k2.metric("MISS", int(counts.get("MISS", 0)))
    k3.metric("IN-LINE", int(counts.get("IN-LINE", 0)))
    k4.metric("Avg spread (pp)", f"{released['spread'].mean():+.3f}")
    k5.metric("Months", f"{len(released)}" + (f" (+{pending} pending)" if pending else ""))

    st.caption(
        f"Window: {df.index.min():%b %Y} – {df.index.max():%b %Y} "
        f"({lookback} = last {LOOKBACK_YEARS[lookback]} full year(s) + current year)"
        if LOOKBACK_YEARS[lookback]
        else f"Window: {df.index.min():%b %Y} – {df.index.max():%b %Y} (all data)"
    )
    for year in sorted(df.index.year.unique(), reverse=True):
        st.subheader(f"{year}")
        st.plotly_chart(
            build_year_heatmap(df, year), width="stretch", key=f"hm_{metric}_{year}"
        )
    st.caption(
        "Values are month-over-month % change. Each year and each row (Nowcast, Actual, "
        "Spread) has its own color scale. Spread: green = BEAT, red = MISS. "
        "IN-LINE = within ±0.01 pp. Blank cells have no nowcast or no CPI release yet."
    )

    with st.expander("Raw data"):
        table = df.sort_index(ascending=False).reset_index()
        table["month"] = table["month"].dt.strftime("%Y-%m")
        st.dataframe(table, width="stretch", hide_index=True)
        st.download_button(
            "Download CSV",
            table.to_csv(index=False).encode("utf-8"),
            file_name=f"{metric.lower().replace(' ', '_')}_nowcast_vs_actual.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------ series explorer


@st.cache_data(ttl=3600, show_spinner=False)
def load_series(series_id: str, api_key: str) -> pd.DataFrame:
    df = pf.get_series(series_id=series_id, api_key=api_key)
    df = df[["date", "value"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().set_index("date")


@st.cache_data(ttl=86400, show_spinner=False)
def load_metadata(series_id: str, api_key: str) -> dict:
    resp = requests.get(
        f"{FRED_API_ROOT}/series",
        params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["seriess"][0]


def render_explorer(api_key: str) -> None:
    c1, c2, c3 = st.columns(3)
    choice = c1.selectbox("Indicator", [*PRESETS, "Custom…"])
    if choice == "Custom…":
        series_id = c2.text_input("FRED series ID", value="GDP").strip().upper()
    else:
        series_id = PRESETS[choice]
        c2.caption(f"Series ID: `{series_id}`")
    transform = c3.radio("Transform", ["Level", "% change (YoY)", "% change (period)"])

    if not series_id:
        st.info("Enter a FRED series ID to begin.")
        return

    try:
        with st.spinner("Fetching data from FRED…"):
            data = load_series(series_id, api_key)
            meta = load_metadata(series_id, api_key)
    except requests.HTTPError as exc:
        st.error(f"FRED request failed: HTTP {exc.response.status_code}. Check the series ID and key.")
        return
    except Exception as exc:  # pyfredapi raises its own error types
        st.error(f"Could not load `{series_id}`: {type(exc).__name__}")
        return

    if data.empty:
        st.warning("This series returned no observations.")
        return

    min_date, max_date = data.index.min().date(), data.index.max().date()
    start, end = st.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=(max(min_date, (pd.Timestamp(max_date) - pd.DateOffset(years=20)).date()), max_date),
    )

    view = data.loc[str(start):str(end)].copy()
    if transform == "% change (period)":
        view["value"] = view["value"].pct_change() * 100
    elif transform == "% change (YoY)":
        periods_per_year = {"D": 252, "W": 52, "M": 12, "Q": 4, "SA": 2, "A": 1}
        periods = periods_per_year.get(meta.get("frequency_short"), 12)
        view["value"] = view["value"].pct_change(periods) * 100
    view = view.dropna()

    st.subheader(meta["title"])
    st.caption(
        f"{meta['frequency']} · {meta['units']} · {meta['seasonal_adjustment']} · "
        f"Last updated {meta['last_updated'][:10]}"
    )

    latest = data["value"].iloc[-1]
    prev = data["value"].iloc[-2] if len(data) > 1 else None
    m1, m2, m3 = st.columns(3)
    m1.metric("Latest", f"{latest:,.2f}", None if prev is None else f"{latest - prev:,.2f}")
    m2.metric("Latest date", str(max_date))
    m3.metric("Observations", f"{len(data):,}")

    tab_chart, tab_table = st.tabs(["Chart", "Data"])
    with tab_chart:
        st.line_chart(view["value"])
    with tab_table:
        st.dataframe(view.sort_index(ascending=False), width="stretch")
        st.download_button(
            "Download CSV",
            view.to_csv().encode("utf-8"),
            file_name=f"{series_id}.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------------------ main


def main() -> None:
    st.title("📈 Macroeconomic Data Tracker")

    api_key = get_api_key()
    if not api_key:
        st.error("No FRED API key found.")
        st.markdown(
            "Add it to `.streamlit/secrets.toml` locally, or to **App settings → "
            "Secrets** on Streamlit Community Cloud:\n\n"
            '```toml\n[fred]\napi_key = "your-key-here"\n```'
        )
        st.stop()

    tab_nowcast, tab_explorer = st.tabs(["CPI Nowcast Beat/Miss", "Series Explorer"])
    with tab_nowcast:
        render_nowcast(api_key)
    with tab_explorer:
        render_explorer(api_key)


main()
