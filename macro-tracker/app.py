"""Macroeconomic data tracker built on Streamlit and the FRED API."""

import pandas as pd
import pyfredapi as pf
import requests
import streamlit as st

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

st.set_page_config(page_title="Macro Tracker", page_icon="📈", layout="wide")


def get_api_key() -> str | None:
    """Read the FRED key from st.secrets without ever echoing it."""
    try:
        return st.secrets["fred"]["api_key"]
    except (KeyError, FileNotFoundError):
        return None


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

    with st.sidebar:
        st.header("Series")
        choice = st.selectbox("Indicator", [*PRESETS, "Custom…"])
        if choice == "Custom…":
            series_id = st.text_input("FRED series ID", value="GDP").strip().upper()
        else:
            series_id = PRESETS[choice]
            st.caption(f"Series ID: `{series_id}`")

        transform = st.radio(
            "Transform",
            ["Level", "% change (YoY)", "% change (period)"],
        )

    if not series_id:
        st.info("Enter a FRED series ID to begin.")
        st.stop()

    try:
        with st.spinner("Fetching data from FRED…"):
            data = load_series(series_id, api_key)
            meta = load_metadata(series_id, api_key)
    except requests.HTTPError as exc:
        st.error(f"FRED request failed: HTTP {exc.response.status_code}. Check the series ID and key.")
        st.stop()
    except Exception as exc:  # pyfredapi raises its own error types
        st.error(f"Could not load `{series_id}`: {type(exc).__name__}")
        st.stop()

    if data.empty:
        st.warning("This series returned no observations.")
        st.stop()

    min_date, max_date = data.index.min().date(), data.index.max().date()
    with st.sidebar:
        start, end = st.slider(
            "Date range",
            min_value=min_date,
            max_value=max_date,
            value=(max(min_date, max_date - pd.DateOffset(years=20)), max_date),
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

    latest, prev = data["value"].iloc[-1], data["value"].iloc[-2] if len(data) > 1 else None
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest", f"{latest:,.2f}", None if prev is None else f"{latest - prev:,.2f}")
    c2.metric("Latest date", str(max_date))
    c3.metric("Observations", f"{len(data):,}")

    tab_chart, tab_table = st.tabs(["Chart", "Data"])
    with tab_chart:
        st.line_chart(view["value"])
    with tab_table:
        st.dataframe(view.sort_index(ascending=False), use_container_width=True)
        st.download_button(
            "Download CSV",
            view.to_csv().encode("utf-8"),
            file_name=f"{series_id}.csv",
            mime="text/csv",
        )


main()
