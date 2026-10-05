"""Yield curve page: plotting functions plus render_yield_curve().

Data loading and maths live in treasury.py so other pages can reuse them.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from treasury import (
    MACD_FAST,
    PIVOT_WINDOW,
    rsi_divergences,
    MACD_SIGNAL,
    MACD_SLOW,
    RSI_AVG_WINDOW,
    RSI_PERIOD,
    Z_WINDOWS,
    macd,
    rsi,
    rsi_average,
    zscore,
    EMA_WINDOWS,
    HORIZONS,
    ema,
    SPREADS,
    changes,
    clear_curve_cache,
    compute_spreads,
    curve_on,
    horizon_label,
    load_yield_curve,
    slice_lookback,
    target_date,
    tenor_years,
)

INK = "#0a1f3d"
# comparison curves cycle through these (the latest curve is always navy and thickest)
COMPARE_COLORS = ["#1e6fd9", "#d73027", "#fb9a4b", "#2a9d8f", "#7b5ea7", "#4da3ff", "#8d99ae", "#e9c46a"]
COMPARE_DASHES = ["solid", "dash", "dot", "dashdot"]

# Kept in the data and tables, but not drawn on the curve chart.
CHART_HIDDEN_TENORS = ["1M"]
Z_COLORS = {63: "#1e6fd9", 126: "#7b5ea7", 252: "#0a1f3d"}
Z_OPTIONS = [f"{n}D" for n in Z_WINDOWS]
EMA_COLORS = {20: "#8e44ad", 50: "#fb8c1a", 100: "#2e9e4f", 200: "#d73027"}  # purple/orange/green/red
EMA_OPTIONS = [f"{n}D" for n in EMA_WINDOWS]
AXIS_TENORS =["3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
COMPARE_PRESETS =["1D", "1W", "2W", "1M", "2M", "3M", "6M", "YTD", "1Y", "2Y", "5Y"]
DEFAULT_COMPARE = ["1D", "1W", "1M", "1Y"]
CUSTOM_UNITS = {"Days": "D", "Weeks": "W", "Months": "M", "Years": "Y"}
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


def comparison_curves(df: pd.DataFrame, horizons: list[str]):
    """(horizon, date, curve) for each requested horizon, nearest date first.

    Horizons that fall before the start of the data are returned separately so the
    page can say so instead of silently dropping them.
    """
    latest_date = df.index[-1]
    found, missing = [], []
    for h in dict.fromkeys(horizons):  # de-duplicate, keep order
        hit = curve_on(df, target_date(df.index, latest_date, h))
        (found if hit else missing).append((h, *hit) if hit else h)
    found.sort(key=lambda item: item[1], reverse=True)
    return found, missing


def curve_figure(df: pd.DataFrame, log_x: bool = False, horizons: list[str] | None = None) -> go.Figure:
    """Yield vs maturity (numeric years): the latest curve plus any chosen comparison dates."""
    latest_date = df.index[-1]
    curves = [("Latest", latest_date, df.iloc[-1], INK, "solid", 3.5)]
    found, _ = comparison_curves(df, DEFAULT_COMPARE if horizons is None else horizons)
    for i, (h, when, row) in enumerate(found):
        curves.append(
            (horizon_label(h), when, row, COMPARE_COLORS[i % len(COMPARE_COLORS)],
             COMPARE_DASHES[i % len(COMPARE_DASHES)], 2)
        )

    fig = go.Figure()
    for name, when, row, color, dash, width in curves:
        # a gap is a gap: tenors with no print are not drawn or bridged
        row = row.drop(CHART_HIDDEN_TENORS, errors="ignore").dropna()
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
    # Label only these on the axis (under 1Y just 3M / 6M); every tenor is still plotted
    # and shows its label on hover.
    labelled = [c for c in AXIS_TENORS if c in df.columns]
    fig.update_xaxes(
        title="Maturity (years)",
        type="log" if log_x else "linear",
        tickvals=[tenor_years(c) for c in labelled],
        ticktext=labelled,
    )
    fig.update_yaxes(title="Yield (%)", ticksuffix="%")
    return _base_layout(fig, 480).update_layout(hovermode="closest")


DIVERGENCE_STYLE = {
    "bullish": dict(symbol="triangle-up", color="#2e9e4f", label="Bullish divergence"),
    "bearish": dict(symbol="triangle-down", color="#d73027", label="Bearish divergence"),
}


def divergence_traces(divs: pd.DataFrame, column: str, unit: str) -> list[go.Scatter]:
    """Triangle markers (green = bullish, red = bearish) at each divergence's second swing.

    `column` is 'price' for the main chart or 'rsi' for the RSI panel.
    """
    traces = []
    for kind, style in DIVERGENCE_STYLE.items():
        sub = divs[divs["kind"] == kind]
        if sub.empty:
            continue
        suffix = " bp" if unit == "bp" else "%"
        details = [
            f"{row.price:.2f}{suffix} vs {row.prev_price:.2f}{suffix} on {row.prev_date:%Y-%m-%d}"
            f"<br>RSI {row.rsi:.1f} vs {row.prev_rsi:.1f}"
            for row in sub.itertuples()
        ]
        traces.append(
            go.Scatter(
                x=sub["date"],
                y=sub[column],
                mode="markers",
                name=style["label"],
                marker=dict(
                    symbol=style["symbol"], size=12, color=style["color"],
                    line=dict(color="white", width=1),
                ),
                text=details,
                hovertemplate="%{text}<extra>" + style["label"] + "</extra>",
            )
        )
    return traces


def series_figure(
    series: pd.Series,
    name: str,
    unit: str,
    emas: dict[int, pd.Series] | None = None,
    divergences: pd.DataFrame | None = None,
) -> go.Figure:
    """Time series of one tenor (unit '%') or spread (unit 'bp'); gaps stay gaps.

    `emas` maps window -> already-computed EMA series, drawn over the main line.
    """
    suffix = " bp" if unit == "bp" else "%"
    fig = go.Figure(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=name,
            line=dict(color="#1e6fd9", width=2),
            connectgaps=False,
            hovertemplate="%{y:.2f}" + suffix + "<extra>" + name + "</extra>",
        )
    )
    for window, line in (emas or {}).items():
        fig.add_trace(
            go.Scatter(
                x=line.index,
                y=line.values,
                mode="lines",
                name=f"{window}D EMA",
                line=dict(color=EMA_COLORS[window], width=1.5),
                connectgaps=False,
                hovertemplate="%{y:.2f}" + suffix + f"<extra>{window}D EMA</extra>",
            )
        )
    shown = divergences is not None and not divergences.empty
    if shown:
        for trace in divergence_traces(divergences, "price", unit):
            fig.add_trace(trace)
    if unit == "bp":
        fig.add_hline(y=0, line=dict(color=INK, width=1, dash="dot"))
    fig.update_yaxes(title="Spread (bp)" if unit == "bp" else "Yield (%)")
    return _align(_base_layout(fig, 380, showlegend=bool(emas) or shown))


def selected_emas(full: pd.Series, windows: list[str], lookback: str) -> dict[int, pd.Series]:
    """EMAs computed on the full history, then cropped to the chart's lookback."""
    out = {}
    for label in windows:
        n = int(label.rstrip("D"))
        out[n] = slice_lookback(ema(full, n), lookback)
    return out


def _align(fig: go.Figure) -> go.Figure:
    """Same fixed left/right margins on every series panel, so their x-axes line up."""
    fig.update_layout(margin=dict(l=70, r=20, t=30, b=10))
    fig.update_yaxes(automargin=False)
    return fig


def _hline(fig: go.Figure, y: float, dash: str = "dash", color: str = "#8d99ae") -> None:
    fig.add_hline(y=y, line=dict(color=color, width=1, dash=dash))


def rsi_figure(
    rsi_line: pd.Series,
    rsi_avg: pd.Series,
    divergences: pd.DataFrame | None = None,
    unit: str = "%",
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rsi_line.index, y=rsi_line.values, mode="lines", name=f"RSI ({RSI_PERIOD})",
        line=dict(color="#1e6fd9", width=1.8), connectgaps=False,
        hovertemplate="%{y:.1f}<extra>RSI</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rsi_avg.index, y=rsi_avg.values, mode="lines", name=f"{RSI_AVG_WINDOW}D average",
        line=dict(color="#fb8c1a", width=1.8), connectgaps=False,
        hovertemplate="%{y:.1f}<extra>" + f"{RSI_AVG_WINDOW}D avg</extra>",
    ))
    if divergences is not None and not divergences.empty:
        for trace in divergence_traces(divergences, "rsi", unit):
            fig.add_trace(trace)
    _hline(fig, 70)
    _hline(fig, 30)
    fig.update_yaxes(title="RSI", range=[0, 100], tickvals=[0, 30, 50, 70, 100])
    return _align(_base_layout(fig, 260, showlegend=True))


def macd_figure(m: pd.DataFrame, unit: str) -> go.Figure:
    hist = m["Histogram"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hist.index, y=hist.values, name="Histogram",
        marker_color=["#2e9e4f" if v >= 0 else "#d73027" for v in hist.fillna(0)],
        hovertemplate="%{y:.2f}<extra>Histogram</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=m.index, y=m["MACD"].values, mode="lines", name="MACD",
        line=dict(color="#1e6fd9", width=1.8), connectgaps=False,
        hovertemplate="%{y:.2f}<extra>MACD</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=m.index, y=m["Signal"].values, mode="lines", name="Signal",
        line=dict(color="#fb8c1a", width=1.8), connectgaps=False,
        hovertemplate="%{y:.2f}<extra>Signal</extra>",
    ))
    _hline(fig, 0, "dot", INK)
    fig.update_yaxes(title=f"MACD ({unit})")
    return _align(_base_layout(fig, 260, showlegend=True))


def zscore_figure(zs: dict[int, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for window, z in zs.items():
        fig.add_trace(go.Scatter(
            x=z.index, y=z.values, mode="lines", name=f"{window}D",
            line=dict(color=Z_COLORS[window], width=1.8), connectgaps=False,
            hovertemplate="%{y:.2f}<extra>" + f"{window}D z</extra>",
        ))
    _hline(fig, 0, "dot", INK)
    _hline(fig, 2)
    _hline(fig, -2)
    fig.update_yaxes(title="Z-score")
    return _align(_base_layout(fig, 260, showlegend=bool(zs)))


def window_divergences(full: pd.Series, lookback: str) -> pd.DataFrame:
    """Divergences found on the full history, kept only if they fall inside `lookback`."""
    divs = rsi_divergences(full, rsi(full))
    start = slice_lookback(full, lookback).index[0] if len(full) else None
    return divs if divs.empty or start is None else divs[divs["date"] >= start]


def render_indicator_panels(
    full: pd.Series, lookback: str, scale: float, unit: str, key: str,
    divergences: pd.DataFrame | None = None, price_unit: str = "%",
) -> None:
    """RSI, MACD and Z-score panels, in that order, under a chart.

    Everything is computed on the full loaded history and cropped to `lookback`
    afterwards, so no line is distorted by a warm-up that starts inside the window.
    """
    r = rsi(full)
    st.markdown(f"**RSI ({RSI_PERIOD}) with {RSI_AVG_WINDOW}-day average**")
    st.plotly_chart(
        rsi_figure(
            slice_lookback(r, lookback), slice_lookback(rsi_average(r), lookback),
            divergences, price_unit,
        ),
        width="stretch", key=f"{key}_rsi",
    )

    st.markdown(f"**MACD ({MACD_FAST}, {MACD_SLOW}, {MACD_SIGNAL})**")
    st.plotly_chart(
        macd_figure(slice_lookback(macd(full, scale=scale), lookback), unit),
        width="stretch", key=f"{key}_macd",
    )

    st.markdown("**Z-score**")
    picked = st.multiselect(
        "Z-score windows",
        Z_OPTIONS,
        default=Z_OPTIONS,
        key=f"{key}_z",
        help="Rolling z-score over trading days. Deselect any to hide it.",
    )
    zs = {int(w.rstrip("D")): slice_lookback(zscore(full, int(w.rstrip("D"))), lookback) for w in picked}
    st.plotly_chart(zscore_figure(zs), width="stretch", key=f"{key}_zchart")
    if not zs:
        st.caption("Select one or more z-score windows above.")


def divergence_picker(container, key: str) -> bool:
    """Checkbox for the RSI divergence triangles on the chart and the RSI panel."""
    return container.checkbox(
        "Show RSI divergences  (green ▲ bullish, red ▼ bearish)",
        value=True,
        key=key,
        help=(
            "Bearish: the series makes a higher swing high while RSI makes a lower high. "
            "Bullish: a lower swing low while RSI makes a higher low. Swings are the extreme of "
            f"{PIVOT_WINDOW} trading days either side, so the newest {PIVOT_WINDOW} days cannot "
            "show a signal yet. Applied to the yield/spread itself, not to bond prices."
        ),
    )


def ema_picker(container, key: str) -> list[str]:
    """Multiselect of the moving averages to overlay; deselect any to hide it."""
    return container.multiselect(
        "Moving averages (EMA)",
        EMA_OPTIONS,
        default=EMA_OPTIONS,
        key=key,
        help="Exponential moving averages over trading days. Deselect any to hide it.",
    )


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

    # 1. any tenor
    st.subheader("Tenor history")
    t1, t2 = st.columns([1, 3])
    tenor = t1.selectbox("Tenor", list(df.columns), index=list(df.columns).index("10Y")
                         if "10Y" in df.columns else 0, key="yc_tenor")
    tenor_lb = t2.radio("Lookback", LOOKBACKS, index=2, horizontal=True, key="yc_tenor_lb")
    tenor_emas = ema_picker(st, "yc_tenor_ema")
    tenor_divs = divergence_picker(st, "yc_tenor_div")
    tenor_div_rows = window_divergences(df[tenor], tenor_lb) if tenor_divs else None
    tseries = slice_lookback(df[tenor], tenor_lb)
    st.plotly_chart(
        series_figure(
            tseries, tenor, "%", selected_emas(df[tenor], tenor_emas, tenor_lb), tenor_div_rows
        ),
        width="stretch",
        key="yc_tenor_chart",
    )
    if tseries.isna().any():
        first = df[tenor].first_valid_index()
        st.caption(f"{tenor} has no data before {first:%Y-%m-%d} (the tenor did not exist yet).")
    render_indicator_panels(
        df[tenor], tenor_lb, scale=100.0, unit="bp", key="yc_tenor_ind",
        divergences=tenor_div_rows, price_unit="%",
    )

    # 2. spreads: chart first, then the table
    st.subheader("Spreads")
    spreads = compute_spreads(df)
    s1, s2 = st.columns([1, 3])
    spread_name = s1.selectbox("Spread", list(SPREADS), key="yc_spread")
    spread_lb = s2.radio("Lookback", LOOKBACKS, index=2, horizontal=True, key="yc_spread_lb")
    spread_emas = ema_picker(st, "yc_spread_ema")
    spread_divs = divergence_picker(st, "yc_spread_div")
    spread_div_rows = window_divergences(spreads[spread_name], spread_lb) if spread_divs else None
    series = slice_lookback(spreads[spread_name], spread_lb)
    short, long_ = SPREADS[spread_name]
    st.plotly_chart(
        series_figure(
            series, spread_name, "bp",
            selected_emas(spreads[spread_name], spread_emas, spread_lb), spread_div_rows,
        ),
        width="stretch",
        key="yc_spread_chart",
    )
    st.caption(f"{spread_name} = {long_} yield minus {short} yield, in basis points.")
    render_indicator_panels(
        spreads[spread_name], spread_lb, scale=1.0, unit="bp", key="yc_spread_ind",
        divergences=spread_div_rows, price_unit="bp",
    )
    st.dataframe(
        changes(spreads, scale=1.0),
        width="stretch",
        column_config=_change_config("Latest (bp)"),
    )

    # 3. latest curve vs history, then the table of all tenors
    st.subheader("Yield curve")
    p1, p2, p3 = st.columns([4, 1, 1])
    picked = p1.multiselect(
        "Compare the latest curve with",
        COMPARE_PRESETS,
        default=DEFAULT_COMPARE,
        key="yc_compare",
        help="Pick any combination. YTD = last print of the prior year; 1D = previous trading day.",
    )
    custom_n = p2.number_input(
        "Custom offset", min_value=0, max_value=60, value=0, step=1, key="yc_custom_n",
        help="Add one more comparison date. 0 = off.",
    )
    custom_unit = p3.selectbox("Unit", list(CUSTOM_UNITS), index=2, key="yc_custom_unit")
    horizons = list(picked)
    if custom_n:
        horizons.append(f"{int(custom_n)}{CUSTOM_UNITS[custom_unit]}")
    log_x = st.checkbox(
        "Log maturity axis (spreads out the short end; untick for true-to-scale spacing)",
        value=True,
        key="yc_log",
    )

    _, missing = comparison_curves(df, horizons)
    st.plotly_chart(curve_figure(df, log_x, horizons), width="stretch", key="yc_curve")
    if missing:
        st.caption(
            "No data on or before these dates, so they are not drawn: "
            + ", ".join(horizon_label(h) for h in missing)
            + "."
        )
    if not horizons:
        st.caption("Select one or more comparison dates above to overlay earlier curves.")

    st.subheader("All tenors")
    st.dataframe(
        changes(df), width="stretch", column_config=_change_config("Latest (%)"),
    )
    st.caption(
        "Levels in percent, changes in basis points. Each change compares with the last "
        "observation on or before the reference date (1D = previous trading day, YTD = last "
        "print of the prior year). Blank = that tenor had no print then."
    )
