"""Yield curve page: plotting functions plus render_yield_curve().

Data loading and maths live in treasury.py so other pages can reuse them.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from treasury import (
    HORIZONS,
    SPREADS,
    changes,
    clear_curve_cache,
    compute_spreads,
    curve_on,
    load_yield_curve,
    slice_lookback,
    target_date,
    tenor_years,
)

INK = "#0a1f3d"
# latest curve in navy, then progressively lighter / warmer comparison curves
CURVE_STYLES = [
    ("Latest", INK, "solid", 3.5),
    ("1 day ago", "#1e6fd9", "solid", 2),
    ("1 week ago", "#4da3ff", "dash", 2),
    ("1 month ago", "#fb9a4b", "dash", 2),
    ("1 year ago", "#d73027", "dot", 2),
]
CURVE_OFFSETS = {"1 day ago": "1D", "1 week ago": "1W", "1 month ago": "1M", "1 year ago": "1Y"}
LOOKBACKS = ["1Y", "3Y", "5Y", "Max"]
FULL_HISTORY_YEARS = date.today().year - 1990


def _base_layout(fig: go.Figure, height: int, **kwargs) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font_color=INK,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
        **kwargs,
    )
    return fig


# ----------------------------------------------------------------------- plotting


def curve_figure(df: pd.DataFrame, log_x: bool = False) -> go.Figure:
    """Yield vs maturity (numeric years) for the latest date and four lookbacks."""
    latest_date = df.index[-1]
    fig = go.Figure()
    for name, color, dash, width in CURVE_STYLES:
        if name == "Latest":
            found = (latest_date, df.iloc[-1])
        else:
            found = curve_on(df, target_date(df.index, latest_date, CURVE_OFFSETS[name]))
        if found is None:
            continue
        when, row = found
        row = row.dropna()  # a gap is a gap: tenors with no print are not drawn or bridged
        fig.add_trace(
            go.Scatter(
                x=[tenor_years(c) for c in row.index],
                y=row.values,
                mode="lines+markers",
                name=f"{name} ({when:%Y-%m-%d})",
                line=dict(color=color, dash=dash, width=width),
                marker=dict(size=6),
                customdata=list(row.index),
                hovertemplate="%{customdata}: %{y:.2f}%<extra>" + name + "</extra>",
            )
        )
    fig.update_xaxes(
        title="Maturity (years)",
        type="log" if log_x else "linear",
        tickvals=[tenor_years(c) for c in df.columns],
        ticktext=list(df.columns),
        tickangle=-45,
    )
    fig.update_yaxes(title="Yield (%)", ticksuffix="%")
    return _base_layout(fig, 480).update_layout(hovermode="closest")


def series_figure(series: pd.Series, name: str, unit: str) -> go.Figure:
    """Time series of one tenor (unit '%') or spread (unit 'bp'); gaps stay gaps."""
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=name,
            line=dict(color="#1e6fd9", width=2),
            connectgaps=False,
            hovertemplate="%{y:.2f}" + (" bp" if unit == "bp" else "%") + "<extra></extra>",
        )
    )
    if unit == "bp":
        fig.add_hline(y=0, line=dict(color=INK, width=1, dash="dot"))
    fig.update_yaxes(title="Spread (bp)" if unit == "bp" else "Yield (%)")
    return _base_layout(fig, 380, showlegend=False)


# ------------------------------------------------------------------------ the page


def _change_config(first_col: str) -> dict:
    cfg = {c: st.column_config.NumberColumn(f"{c} (bp)", format="%.1f") for c in HORIZONS}
    cfg["Latest"] = st.column_config.NumberColumn(first_col, format="%.2f")
    return cfg


def render_yield_curve(api_key: str | None = None) -> None:
    c1, c2 = st.columns([3, 1])
    full = c1.checkbox(
        "Load full history (since 1990)",
        key="yc_full",
        help="Default loads the last 5 years. Takes a few extra seconds the first time.",
    )
    if c2.button("Refresh data", key="yc_refresh"):
        clear_curve_cache()

    with st.spinner("Loading Treasury yield curve…"):
        data = load_yield_curve(FULL_HISTORY_YEARS if full else 5, api_key)

    if data.df.empty:
        st.error(f"Yield curve data is unavailable right now. {data.error}")
        return

    df = data.df
    st.markdown(f"**Source: {data.source} | Data through {data.as_of:%Y-%m-%d}**")
    if data.fallback:
        st.warning(
            f"Treasury.gov could not be loaded ({data.error}). Showing the FRED daily series "
            "instead; FRED lags Treasury.gov by about a day and has fewer tenors."
        )

    # 1. latest curve vs history
    st.subheader("Yield curve")
    log_x = st.checkbox("Log maturity axis (spreads out the short end)", key="yc_log")
    st.plotly_chart(curve_figure(df, log_x), width="stretch", key="yc_curve")

    # 2. all tenors
    st.subheader("All tenors")
    table = changes(df)
    st.dataframe(
        table, width="stretch", column_config=_change_config("Latest (%)"),
    )
    st.caption(
        "Levels in percent, changes in basis points. Each change compares with the last "
        "observation on or before the reference date (1D = previous trading day, YTD = last "
        "print of the prior year). Blank = that tenor had no print then."
    )

    # 3. spreads
    st.subheader("Spreads")
    spreads = compute_spreads(df)
    st.dataframe(
        changes(spreads, scale=1.0),
        width="stretch",
        column_config=_change_config("Latest (bp)"),
    )
    s1, s2 = st.columns([1, 3])
    spread_name = s1.selectbox("Spread", list(SPREADS), key="yc_spread")
    spread_lb = s2.radio("Lookback", LOOKBACKS, index=2, horizontal=True, key="yc_spread_lb")
    series = slice_lookback(spreads[spread_name], spread_lb)
    short, long_ = SPREADS[spread_name]
    st.plotly_chart(
        series_figure(series, spread_name, "bp"), width="stretch", key="yc_spread_chart"
    )
    st.caption(f"{spread_name} = {long_} yield minus {short} yield, in basis points.")

    # 4. any tenor
    st.subheader("Tenor history")
    t1, t2 = st.columns([1, 3])
    tenor = t1.selectbox("Tenor", list(df.columns), index=list(df.columns).index("10Y")
                         if "10Y" in df.columns else 0, key="yc_tenor")
    tenor_lb = t2.radio("Lookback", LOOKBACKS, index=2, horizontal=True, key="yc_tenor_lb")
    tseries = slice_lookback(df[tenor], tenor_lb)
    st.plotly_chart(series_figure(tseries, tenor, "%"), width="stretch", key="yc_tenor_chart")
    if tseries.isna().any():
        first = df[tenor].first_valid_index()
        st.caption(f"{tenor} has no data before {first:%Y-%m-%d} (the tenor did not exist yet).")
