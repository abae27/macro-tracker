"""Macroeconomic data tracker built on Streamlit and the FRED API."""

import calendar

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyfredapi as pf
from plotly.subplots import make_subplots
import requests
import streamlit as st

from oer import (
    NATIONAL,
    REGIONS,
    averages_table,
    compute_oer,
    estimate_next,
    fetch_oer_levels,
    rankings_table,
)
from nowcast import (
    LOOKBACK_YEARS,
    METRICS,
    fetch_fred_data,
    fetch_nowcast_data,
    filter_lookback,
    monthly_summary,
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

# Nowcast / Actual: lowest = light blue, middle = yellow, highest = red.
LEVEL_COLORSCALE = [
    [0.00, "#cfe8f7"],
    [0.25, "#f3f0c0"],
    [0.50, "#ffe45e"],
    [0.75, "#fb9a4b"],
    [1.00, "#d73027"],
]

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


def _color_axis(values: np.ndarray) -> dict:
    """Blue-yellow-red scale fitted to one row of one year, so no two heatmaps share a scale."""
    finite = values[np.isfinite(values)]
    lo = float(finite.min()) if finite.size else 0.0
    hi = float(finite.max()) if finite.size else 1.0
    if hi - lo < 0.02:
        hi = lo + 0.02
    return dict(colorscale=LEVEL_COLORSCALE, cmin=lo, cmax=hi, showscale=False)


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
        layout[axis] = _color_axis(values)

    fig.update_layout(
        template="plotly_white",
        font_color="#0a1f3d",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
    st.subheader(f"Monthly averages & rankings ({lookback})")
    st.dataframe(
        monthly_summary(df),
        width="stretch",
        column_config={
            "Avg Nowcast": st.column_config.NumberColumn(format="%.3f"),
            "Avg Actual": st.column_config.NumberColumn(format="%.3f"),
            "Nowcast Rank": st.column_config.NumberColumn(format="%.1f"),
            "Actual Rank": st.column_config.NumberColumn(format="%.1f"),
            "Hit Ratio": st.column_config.NumberColumn(format="%.2f"),
            "Hit Rank": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption(
        "Averages are the mean m/m print for each calendar month across the window. "
        "Rank 1 = highest (ties share the mean rank, e.g. 9.5). "
        "Hit Ratio = BEAT / (BEAT + MISS); IN-LINE and pending months are excluded."
    )

    for year in sorted(df.index.year.unique(), reverse=True):
        st.subheader(f"{year}")
        st.plotly_chart(
            build_year_heatmap(df, year), width="stretch", key=f"hm_{metric}_{year}"
        )
    st.caption(
        "Values are month-over-month % change. Each year and each row (Nowcast, Actual, "
        "Spread) has its own color scale: light blue = lowest, yellow = middle, red = highest. "
        "Spread above zero is a BEAT, below zero a MISS. "
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


# ------------------------------------------------------------------ OER tabs


def build_region_heatmap(series: pd.Series) -> go.Figure:
    """One region's MoM prints: months down the left, one column per calendar year
    (each year colored on its own scale), then Avg and Rank columns on the right.

    Rank 1 = highest monthly average (ties share the mean rank, e.g. 9.5).
    """
    d = pd.DataFrame({"v": series, "year": series.index.year, "mon": series.index.month})
    pivot = d.pivot_table(index="mon", columns="year", values="v", aggfunc="first")
    pivot = pivot.reindex(range(1, 13))

    avg = pivot.mean(axis=1)  # NaN-skipping mean across the years shown
    rank = avg.round(9).rank(ascending=False, method="average")

    # (header, values, text format, color axis)
    columns = [
        (str(y), pivot[y].to_numpy(dtype=float), "%{z:.2f}", _color_axis(pivot[y].to_numpy(dtype=float)))
        for y in pivot.columns
    ]
    columns.append(("Avg", avg.to_numpy(dtype=float), "%{z:.3f}", _color_axis(avg.to_numpy(dtype=float))))
    # Reversed scale so rank 1 (highest) is red, matching "highest = red".
    rank_axis = dict(
        colorscale=[[1 - pos, color] for pos, color in LEVEL_COLORSCALE][::-1],
        cmin=1,
        cmax=12,
        showscale=False,
    )
    columns.append(("Rank", rank.to_numpy(dtype=float), "%{z:.1f}", rank_axis))

    fig = make_subplots(
        rows=1,
        cols=len(columns),
        shared_yaxes=True,
        horizontal_spacing=0.006,
    )
    layout = {}
    for i, (header, values, fmt, axis_cfg) in enumerate(columns, start=1):
        axis = "coloraxis" if i == 1 else f"coloraxis{i}"
        fig.add_trace(
            go.Heatmap(
                z=values.reshape(-1, 1),
                x=[header],
                y=MONTHS,
                coloraxis=axis,
                texttemplate=fmt,
                hoverongaps=False,
            ),
            row=1,
            col=i,
        )
        layout[axis] = axis_cfg

    fig.update_layout(
        template="plotly_white",
        font_color="#0a1f3d",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=440,
        margin=dict(l=10, r=10, t=40, b=10),
        **layout,
    )
    fig.update_yaxes(autorange="reversed")  # January at the top
    fig.update_xaxes(type="category", side="top")
    return fig


def _load_oer(api_key: str):
    try:
        with st.spinner("Loading regional OER from FRED…"):
            return fetch_oer_levels(api_key)
    except Exception as exc:  # pyfredapi raises its own error types
        st.error(f"Could not load regional OER data: {type(exc).__name__}")
        return None


def _month_table(df: pd.DataFrame, lookback: str) -> tuple[pd.DataFrame, dict]:
    """Newest-first table with YYYY-MM dates, same as the nowcast raw-data table."""
    view = filter_lookback(df, lookback).sort_index(ascending=False)
    view = view.rename_axis("month").reset_index()
    view["month"] = view["month"].dt.strftime("%Y-%m")
    config = {c: st.column_config.NumberColumn(format="%.3f") for c in df.columns}
    return view, config


def render_oer_levels(levels: pd.DataFrame, lookback: str) -> None:
    if levels.dropna(how="all").empty:
        st.warning("FRED returned no observations for the regional OER series.")
        return

    latest = levels.dropna(how="all").iloc[-1]
    prev = levels.dropna(how="all").iloc[-2] if len(levels.dropna(how="all")) > 1 else None
    st.caption(f"Latest observation: {levels.dropna(how='all').index[-1]:%b %Y}")
    cols = st.columns(len(REGIONS))
    for col, region in zip(cols, REGIONS):
        delta = None if prev is None else f"{latest[region] - prev[region]:+.3f}"
        col.metric(region, f"{latest[region]:,.3f}", delta)

    view = filter_lookback(levels, lookback)
    st.line_chart(view)

    table, config = _month_table(levels, lookback)
    st.dataframe(table, width="stretch", hide_index=True, column_config=config)
    st.download_button(
        "Download CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="oer_index_levels.csv",
        mime="text/csv",
        key="dl_levels",
    )

    with st.expander("Series & weights"):
        st.dataframe(
            pd.DataFrame(
                {
                    "FRED series": [m["series"] for m in REGIONS.values()],
                    "Weight in headline CPI": [m["cpi_weight"] for m in REGIONS.values()],
                    "Share of national OER (%)": [m["oer_share"] for m in REGIONS.values()],
                },
                index=pd.Index(list(REGIONS), name="Region"),
            ),
            width="stretch",
            column_config={
                "Weight in headline CPI": st.column_config.NumberColumn(format="%.3f"),
                "Share of national OER (%)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    st.caption(
        "Index levels are not seasonally adjusted (CUUR series). Everything below is "
        "computed from this table."
    )


def render_oer_analysis(levels: pd.DataFrame, lookback: str) -> None:
    calc = compute_oer(levels)
    mom, contrib_cpi, contrib_oer = calc["mom"], calc["contrib_cpi"], calc["contrib_oer"]
    complete = mom.dropna()
    if complete.empty:
        st.warning("Not enough data to compute month-over-month changes.")
        return
    latest = complete.index[-1]

    k1, k2, k3 = st.columns(3)
    k1.metric(f"National OER MoM ({latest:%b %Y})", f"{mom.loc[latest, NATIONAL]:.3f}%")
    k2.metric("Contribution to headline CPI", f"{contrib_cpi.loc[latest, 'Total']:.3f} pp")
    top = mom.loc[latest, list(REGIONS)].idxmax()
    k3.metric("Hottest region", top, f"{mom.loc[latest, top]:.3f}%")

    target, est, info = estimate_next(levels, mom)
    if target is not None:
        st.subheader(f"Estimated next print ({target:%b %Y})")
        st.dataframe(
            est,
            width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.3f") for c in est.columns},
        )
        st.caption(
            f"Est. MoM = 12-month trend + β × the {info['source']:%b %Y} surprise (that month's "
            f"MoM vs. the trend before it). OER is priced in six-month panels, so a hot month "
            f"tends to be followed by a cooler one six months later. "
            f"β = {info['beta']:+.2f} (t = {info['tstat']:.1f}), fit across all four regions. "
            f"Backtest, last {info['backtest_months']} months with β refit each month: average "
            f"error {info['mae_model']:.3f} pp vs {info['mae_trend']:.3f} pp for the 12-month "
            f"trend alone. National Last/Est. Level is the OER-share-weighted blend of the "
            f"regional levels. A statistical estimate, not the BLS release."
        )

    st.line_chart(filter_lookback(mom, lookback))

    t_mom, t_cpi, t_oer = st.tabs(
        ["MoM %", "Contribution to headline CPI (pp)", "Contribution to national OER (pp)"]
    )
    for tab, frame, name in (
        (t_mom, mom, "mom"),
        (t_cpi, contrib_cpi, "contribution_to_cpi"),
        (t_oer, contrib_oer, "contribution_to_oer"),
    ):
        with tab:
            table, config = _month_table(frame, lookback)
            st.dataframe(table, width="stretch", hide_index=True, column_config=config)
            st.download_button(
                "Download CSV",
                table.to_csv(index=False).encode("utf-8"),
                file_name=f"oer_{name}.csv",
                mime="text/csv",
                key=f"dl_{name}",
            )
    st.caption(
        "MoM = % change in the index level. Contribution to headline CPI = MoM × region's "
        "CPI weight ÷ 100. Contribution to national OER = MoM × region's share of OER ÷ 100, "
        "and the shares sum to 100%, so those contributions add up to the National OER MoM."
    )

    st.subheader("Monthly prints by region (MoM %)")
    heat = filter_lookback(mom, lookback)
    for region in REGIONS:
        st.markdown(f"**{region}**")
        st.plotly_chart(
            build_region_heatmap(heat[region]), width="stretch", key=f"oer_hm_{region}"
        )
    st.caption(
        "Month-over-month % change in each region's OER index. Every year column, the Avg "
        "column and the Rank column has its own color scale: light blue = lowest, yellow = "
        "middle, red = highest. Avg is the mean of that month across the years shown; "
        "Rank 1 = highest average (ties share the mean rank, e.g. 9.5)."
    )

    st.subheader("Trailing averages")
    avg = averages_table(mom)
    st.dataframe(
        avg,
        width="stretch",
        column_config={c: st.column_config.NumberColumn(format="%.3f") for c in avg.columns},
    )
    st.caption(
        "Average MoM is the simple mean of the last N monthly changes; Annualized compounds "
        "them to a yearly rate. Based on the latest data, regardless of the lookback above."
    )

    st.subheader(f"Regional rankings ({latest:%b %Y})")
    _, ranked = rankings_table(mom, contrib_cpi, contrib_oer)
    st.dataframe(
        ranked,
        width="stretch",
        column_config={
            "MoM %": st.column_config.NumberColumn(format="%.3f"),
            "Contribution to CPI (pp)": st.column_config.NumberColumn(format="%.3f"),
            "Contribution to OER (pp)": st.column_config.NumberColumn(format="%.3f"),
            "MoM Rank": st.column_config.NumberColumn(format="%.1f"),
            "CPI Contribution Rank": st.column_config.NumberColumn(format="%.1f"),
            "OER Contribution Rank": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    st.caption("Rank 1 = highest. Ties share the mean rank (e.g. 2.5).")


def render_oer(api_key: str) -> None:
    """Combined OER tab: index levels first, then the analysis computed from them."""
    lookback = st.radio(
        "Lookback", list(LOOKBACK_YEARS), index=4, horizontal=True, key="oer_lookback"
    )
    levels = _load_oer(api_key)
    if levels is None:
        return

    st.header("Month-over-month, contributions & rankings")
    render_oer_analysis(levels, lookback)
    st.divider()
    st.header("Index levels")
    render_oer_levels(levels, lookback)


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

    tab_cpi, tab_explorer = st.tabs(["US CPI", "Series Explorer"])
    with tab_cpi:
        tab_nowcast, tab_levels = st.tabs(["CPI Nowcast Beat/Miss", "CPI Index Level 3"])
        with tab_nowcast:
            render_nowcast(api_key)
        with tab_levels:
            (tab_oer,) = st.tabs(["OER"])
            with tab_oer:
                render_oer(api_key)
    with tab_explorer:
        render_explorer(api_key)


main()
