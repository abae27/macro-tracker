"""Macroeconomic data tracker built on Streamlit and the FRED API."""

import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyfredapi as pf
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

HEATMAP_VIEWS = {
    "Spread (Actual − Nowcast)": "spread",
    "Actual": "actual",
    "Nowcast": "nowcast",
}

st.set_page_config(page_title="Macro Tracker", page_icon="📈", layout="wide")


def get_api_key() -> str | None:
    """Read the FRED key from st.secrets without ever echoing it."""
    try:
        return st.secrets["fred"]["api_key"]
    except (KeyError, FileNotFoundError):
        return None


# ---------------------------------------------------------------- nowcast tab


def build_heatmap(df: pd.DataFrame, value_col: str, metric: str) -> go.Figure:
    """Month (Jan–Dec) by year grid of the chosen value."""
    d = df.assign(year=df.index.year, mon=df.index.month)
    pivot = d.pivot_table(index="mon", columns="year", values=value_col, aggfunc="first")
    pivot = pivot.reindex(range(1, 13))

    is_spread = value_col == "spread"
    if is_spread:
        # Diverging scale centred on zero: green = BEAT (actual hotter), red = MISS.
        limit = max(0.05, float(np.nanpercentile(np.abs(pivot.values), 95)))
        scale = dict(colorscale="RdYlGn", zmin=-limit, zmax=limit, zmid=0)
        label = "pp"
    else:
        scale = dict(colorscale="YlOrRd")
        label = "% m/m"

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[str(y) for y in pivot.columns],
            y=MONTHS,
            texttemplate="%{z:.2f}",
            hoverongaps=False,
            colorbar=dict(title=label),
            **scale,
        )
    )
    fig.update_yaxes(autorange="reversed")  # January at the top
    fig.update_xaxes(type="category", side="top")
    fig.update_layout(
        title=f"{metric}: {value_col}",
        height=520,
        margin=dict(l=10, r=10, t=80, b=10),
    )
    return fig


def render_nowcast(api_key: str) -> None:
    c1, c2, c3 = st.columns([1, 2, 2])
    metric = c1.selectbox("Metric", list(METRICS))
    lookback = c2.radio("Lookback", list(LOOKBACK_YEARS), index=4, horizontal=True)
    view = c3.radio("Heatmap shows", list(HEATMAP_VIEWS), horizontal=True)

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

    st.plotly_chart(build_heatmap(df, HEATMAP_VIEWS[view], metric), width="stretch")
    st.caption(
        "Values are month-over-month % change. BEAT = actual above nowcast; "
        "IN-LINE = within ±0.01 pp. Pending months have a nowcast but no CPI release yet."
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
