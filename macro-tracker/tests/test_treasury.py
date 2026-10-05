"""Tests for the Treasury par yield curve parser, bp-change logic and spreads."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import treasury as t

SAMPLE = Path(__file__).parent / "data" / "treasury_2025_sample.csv"  # real Treasury.gov response excerpt


def ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# ------------------------------------------------------------------------ parser


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return t.parse_par_yield_csv(SAMPLE.read_text())


def test_parser_discovers_tenors_in_maturity_order(sample):
    assert list(sample.columns) == [
        "1M", "1.5M", "2M", "3M", "4M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y",
    ]


def test_parser_sorts_ascending_and_indexes_by_date(sample):
    assert isinstance(sample.index, pd.DatetimeIndex)
    assert sample.index.is_monotonic_increasing
    assert sample.index[0] == ts("2025-01-02") and sample.index[-1] == ts("2025-12-31")
    assert len(sample) == 8


def test_parser_values_match_the_source_rows(sample):
    assert sample.loc["2025-12-31", "10Y"] == 4.18
    assert sample.loc["2025-12-31", "1.5M"] == 3.75
    assert sample.loc["2025-01-02", "30Y"] == 4.79


def test_tenor_that_did_not_exist_is_nan_not_filled(sample):
    # 1.5M first printed on 2025-02-18; earlier rows are blank in the feed
    assert sample.loc["2025-02-18", "1.5M"] == 4.41
    assert sample.loc["2025-02-19", "1.5M"] == 4.42
    for d in ("2025-01-02", "2025-01-03", "2025-02-13", "2025-02-14"):
        assert np.isnan(sample.loc[d, "1.5M"])
    # and nothing was forward/back filled into the older rows
    assert sample["1.5M"].first_valid_index() == ts("2025-02-18")


def test_parser_handles_empty_and_bad_input():
    assert t.parse_par_yield_csv("").empty
    with pytest.raises(t.TreasuryError):
        t.parse_par_yield_csv("foo,bar\n1,2\n")


def test_parser_ignores_non_tenor_columns_and_dedupes():
    text = 'Date,"1 Mo","10 Yr",Note\n01/03/2025,4.0,4.5,x\n01/02/2025,3.9,4.4,y\n01/03/2025,4.1,4.6,z\n'
    df = t.parse_par_yield_csv(text)
    assert list(df.columns) == ["1M", "10Y"]
    assert len(df) == 2 and df.loc["2025-01-03", "10Y"] == 4.6  # last duplicate wins


@pytest.mark.parametrize(
    "label, years, norm",
    [
        ("1 Mo", 1 / 12, "1M"),
        ("1.5 Month", 0.125, "1.5M"),
        ("4 Mo", 4 / 12, "4M"),
        ("1 Yr", 1.0, "1Y"),
        ("30 Yr", 30.0, "30Y"),
        ("10Y", 10.0, "10Y"),
        ("3M", 0.25, "3M"),
    ],
)
def test_tenor_labels(label, years, norm):
    assert t.tenor_years(label) == pytest.approx(years)
    assert t.tenor_label(label) == norm


def test_tenor_label_rejects_non_tenors():
    assert t.tenor_years("Date") is None and t.tenor_label("Note") is None


# ----------------------------------------------------------------- bp-change logic


@pytest.fixture
def ten_year() -> pd.DataFrame:
    idx = pd.to_datetime(
        ["2025-12-30", "2025-12-31", "2026-01-02", "2026-01-05", "2026-01-06",
         "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"]
    )
    return pd.DataFrame({"10Y": [4.00, 4.10, 4.20, 4.25, 4.30, 4.35, 4.40, 4.45, 4.60]}, index=idx)


def test_one_day_is_the_previous_trading_day_not_calendar_day(ten_year):
    # latest Monday 01-12; previous observation is Friday 01-09 (4.45)
    assert t.changes(ten_year).loc["10Y", "1D"] == pytest.approx(15.0)


def test_one_week_uses_observation_on_the_reference_date(ten_year):
    # 01-12 minus 7 days = Monday 01-05 (4.25)
    assert t.changes(ten_year).loc["10Y", "1W"] == pytest.approx(35.0)


def test_ytd_is_measured_from_the_last_print_of_the_prior_year(ten_year):
    assert t.changes(ten_year).loc["10Y", "YTD"] == pytest.approx(50.0)  # vs 12-31 (4.10)


def test_horizon_before_the_start_of_data_is_nan_not_extrapolated(ten_year):
    out = t.changes(ten_year)
    assert np.isnan(out.loc["10Y", "1M"]) and np.isnan(out.loc["10Y", "3M"])


def test_reference_on_a_non_trading_day_uses_previous_observation_not_next():
    idx = pd.to_datetime(["2025-12-11", "2025-12-12", "2025-12-15", "2026-01-14"])
    df = pd.DataFrame({"10Y": [3.80, 3.90, 3.95, 4.40]}, index=idx)
    # 01-14 minus 1 month = Sunday 12-14 -> Friday 12-12 (3.90), not Monday 12-15 (3.95)
    assert t.changes(df).loc["10Y", "1M"] == pytest.approx(50.0)


def test_no_lookahead_future_values_do_not_change_past_references():
    idx = pd.to_datetime(["2025-12-12", "2025-12-15", "2026-01-14"])
    df = pd.DataFrame({"10Y": [3.90, 3.95, 4.40]}, index=idx)
    assert t.asof_value(df["10Y"], ts("2025-12-13")) == 3.90
    assert np.isnan(t.asof_value(df["10Y"], ts("2025-12-11")))
    assert np.isnan(t.asof_value(df["10Y"], None))


def test_tenor_with_no_history_gives_nan_change():
    idx = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    df = pd.DataFrame({"2Y": [3.5, 3.6, 3.7], "1.5M": [np.nan, np.nan, 3.9]}, index=idx)
    out = t.changes(df)
    assert out.loc["2Y", "1D"] == pytest.approx(10.0)
    assert np.isnan(out.loc["1.5M", "1D"])  # nothing to compare with: no interpolation


def test_nan_on_latest_date_gives_nan_change_not_stale_value():
    idx = pd.to_datetime(["2026-01-05", "2026-01-06"])
    df = pd.DataFrame({"10Y": [4.0, np.nan], "2Y": [3.5, 3.6]}, index=idx)
    out = t.changes(df)
    assert np.isnan(out.loc["10Y", "Latest"]) and np.isnan(out.loc["10Y", "1D"])


def test_changes_in_bp_and_scale_one_for_spreads(ten_year):
    assert t.changes(ten_year).loc["10Y", "Latest"] == 4.60  # level stays in percent
    assert t.changes(ten_year, scale=1.0).loc["10Y", "1D"] == pytest.approx(0.15)


def test_curve_on_returns_whole_row_at_or_before_target(ten_year):
    when, row = t.curve_on(ten_year, ts("2026-01-10"))  # Saturday
    assert when == ts("2026-01-09") and row["10Y"] == 4.45
    assert t.curve_on(ten_year, ts("2025-01-01")) is None


@pytest.mark.parametrize(
    "horizon, expected",
    [
        ("2W", "2025-12-29"),
        ("2M", "2025-11-12"),
        ("6M", "2025-07-12"),
        ("2Y", "2024-01-12"),
        ("10D", "2026-01-02"),
    ],
)
def test_generic_horizons_are_calendar_offsets_from_the_latest_date(ten_year, horizon, expected):
    assert t.target_date(ten_year.index, ts("2026-01-12"), horizon) == ts(expected)


def test_custom_horizon_uses_last_observation_at_or_before_target_never_after(ten_year):
    out = t.changes(ten_year, horizons=("2W", "10D"))
    # 2W -> 2025-12-29: the data starts 12-30, so there is nothing at or before it (NaN,
    # not the next available day)
    assert np.isnan(out.loc["10Y", "2W"])
    # 10D -> 2026-01-02: exact observation (4.20) -> 4.60 is +40 bp
    assert out.loc["10Y", "10D"] == pytest.approx(40.0)


def test_bad_horizon_is_rejected(ten_year):
    for bad in ("0D", "week", "1Q", ""):
        with pytest.raises(ValueError):
            t.target_date(ten_year.index, ts("2026-01-12"), bad)


def test_horizon_labels():
    assert t.horizon_label("1D") == "1 day ago"
    assert t.horizon_label("2M") == "2 months ago"
    assert t.horizon_label("1Y") == "1 year ago"
    assert t.horizon_label("YTD") == "Prior year-end"


def test_excluded_tenor_is_dropped_but_other_tenors_are_kept(sample):
    assert "1.5M" in sample.columns  # the parser stays faithful to the feed
    trimmed = t.drop_excluded(sample)
    assert "1.5M" not in trimmed.columns
    assert "1M" in trimmed.columns  # 1M stays in the data (it is only hidden on the chart)
    assert list(trimmed.columns) == [c for c in sample.columns if c != "1.5M"]
    assert trimmed.loc["2025-12-31", "2M"] == 3.67


# ------------------------------------------------------------------------ spreads


@pytest.fixture
def curve() -> pd.DataFrame:
    idx = pd.to_datetime(["2026-01-05", "2026-01-06"])
    return pd.DataFrame(
        {"3M": [4.00, 4.10], "2Y": [3.60, 3.70], "5Y": [3.80, 3.85], "10Y": [4.20, 4.30], "30Y": [4.70, 4.60]},
        index=idx,
    )


def test_spreads_are_long_minus_short_in_bp(curve):
    s = t.compute_spreads(curve)
    last = s.iloc[-1]
    assert last["2s10s"] == pytest.approx(60.0)    # 4.30 - 3.70
    assert last["5s30s"] == pytest.approx(75.0)    # 4.60 - 3.85
    assert last["3M10Y"] == pytest.approx(20.0)    # 4.30 - 4.10
    assert last["2s5s"] == pytest.approx(15.0)     # 3.85 - 3.70
    assert last["10s30s"] == pytest.approx(30.0)   # 4.60 - 4.30
    assert list(s.columns) == ["3M10Y", "2s5s", "2s10s", "5s30s", "10s30s"]


def test_spread_is_nan_when_a_leg_is_missing(curve):
    curve.loc["2026-01-06", "2Y"] = np.nan
    s = t.compute_spreads(curve)
    assert np.isnan(s.loc["2026-01-06", "2s10s"]) and np.isnan(s.loc["2026-01-06", "2s5s"])
    assert s.loc["2026-01-06", "5s30s"] == pytest.approx(75.0)


def test_spread_with_absent_tenor_column_is_all_nan(curve):
    s = t.compute_spreads(curve.drop(columns=["3M"]))
    assert s["3M10Y"].isna().all() and s["2s10s"].notna().all()


def test_spread_changes_in_bp(curve):
    out = t.changes(t.compute_spreads(curve), scale=1.0)
    # 2s10s: 60 now vs (4.20 - 3.60)*100 = 60 yesterday
    assert out.loc["2s10s", "Latest"] == pytest.approx(60.0)
    assert out.loc["2s10s", "1D"] == pytest.approx(0.0)
    assert out.loc["5s30s", "1D"] == pytest.approx(75.0 - 90.0)


# ------------------------------------------------------------------------- EMA


def test_ema_matches_hand_calculation_and_waits_for_a_full_window():
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2026-01-05", periods=4, freq="B"))
    out = t.ema(s, 3)  # alpha = 2 / (3 + 1) = 0.5: 1.0, 1.5, 2.25, 3.125
    assert out.iloc[:2].isna().all()  # fewer than 3 observations: no line yet
    assert out.iloc[2] == pytest.approx(2.25) and out.iloc[3] == pytest.approx(3.125)


def test_ema_of_a_constant_is_that_constant():
    s = pd.Series(4.0, index=pd.date_range("2026-01-05", periods=50, freq="B"))
    assert t.ema(s, 20).dropna().eq(4.0).all()


def test_ema_does_not_fill_a_tenor_that_did_not_exist_yet():
    idx = pd.date_range("2026-01-05", periods=8, freq="B")
    s = pd.Series([np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
    out = t.ema(s, 3)
    assert out.iloc[:2].isna().all()           # before the tenor existed
    assert np.isnan(out.iloc[2]) and np.isnan(out.iloc[3])  # window starts at first real print
    assert out.iloc[4] == pytest.approx(2.25)  # 1.0, 1.5, 2.25 from the 3rd real print on
    assert out.index.equals(s.index)


def test_ema_does_not_interpolate_across_a_missing_day():
    idx = pd.date_range("2026-01-05", periods=5, freq="B")
    s = pd.Series([1.0, 2.0, np.nan, 3.0, 4.0], index=idx)
    out = t.ema(s, 2)  # alpha = 2/3 over the 4 real prints only
    assert np.isnan(out.iloc[2])
    assert out.iloc[3] == pytest.approx(((1 * (1 / 3) + 2 * (2 / 3)) * (1 / 3)) + 3 * (2 / 3))


# ------------------------------------------------------------- RSI / MACD / z-score


def bdays(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-05", periods=n, freq="B")


def test_rsi_matches_hand_calculation_with_wilder_smoothing():
    # period 2, alpha 0.5: avg gain 0.5 -> 0.75, avg loss 0.5 -> 0.25, so RSI = 50 then 75
    out = t.rsi(pd.Series([1.0, 2.0, 1.0, 2.0], index=bdays(4)), 2)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2] == pytest.approx(50.0) and out.iloc[3] == pytest.approx(75.0)


def test_rsi_extremes_and_bounds():
    rising = pd.Series(np.arange(40.0), index=bdays(40))
    assert t.rsi(rising, 14).dropna().eq(100.0).all()
    assert t.rsi(-rising, 14).dropna().eq(0.0).all()
    noisy = pd.Series(np.random.default_rng(0).normal(size=300).cumsum(), index=bdays(300))
    r = t.rsi(noisy).dropna()
    assert r.between(0, 100).all()


def test_rsi_average_waits_for_a_full_window():
    r = pd.Series(np.linspace(40, 60, 30), index=bdays(30))
    avg = t.rsi_average(r, 20)
    assert avg.iloc[:19].isna().all() and avg.iloc[19] == pytest.approx(r.iloc[:20].mean())


def test_macd_matches_hand_calculation():
    out = t.macd(pd.Series([1.0, 2.0, 3.0, 4.0], index=bdays(4)), fast=2, slow=3, signal=2)
    assert out["MACD"].iloc[:2].isna().all()
    assert out["MACD"].iloc[2] == pytest.approx(0.305556, abs=1e-5)
    assert out["MACD"].iloc[3] == pytest.approx(0.393519, abs=1e-5)
    assert np.isnan(out["Signal"].iloc[2]) and out["Signal"].iloc[3] == pytest.approx(0.364198, abs=1e-5)
    assert out["Histogram"].iloc[3] == pytest.approx(0.393519 - 0.364198, abs=1e-5)


def test_macd_scale_converts_percent_to_bp_and_flat_series_is_zero():
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=bdays(4))
    base = t.macd(s, 2, 3, 2)["MACD"]
    assert t.macd(s, 2, 3, 2, scale=100.0)["MACD"].iloc[3] == pytest.approx(base.iloc[3] * 100)
    flat = t.macd(pd.Series(4.0, index=bdays(60)))
    assert flat["MACD"].dropna().abs().max() == pytest.approx(0.0)


def test_zscore_matches_hand_calculation_and_waits_for_a_full_window():
    out = t.zscore(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=bdays(5)), 3)
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == pytest.approx([1.0, 1.0, 1.0])  # mean 2, sample std 1, etc.


def test_zscore_is_nan_when_there_is_no_variation_not_inf():
    out = t.zscore(pd.Series(4.0, index=bdays(10)), 5)
    assert out.isna().all()


def test_indicators_do_not_fill_missing_tenor_history_or_gaps():
    idx = bdays(40)
    s = pd.Series(np.r_[[np.nan] * 10, np.linspace(3, 4, 30)], index=idx)
    for out in (t.rsi(s, 5), t.macd(s, 3, 5, 3)["MACD"], t.zscore(s, 5)):
        assert out.index.equals(idx)
        assert out.iloc[:10].isna().all()  # tenor did not exist yet: stays NaN
    gap = pd.Series([1.0, 2.0, np.nan, 3.0, 4.0, 5.0], index=bdays(6))
    assert np.isnan(t.zscore(gap, 3).iloc[2])  # the missing day itself is not invented


# ------------------------------------------------------------------ divergences

DIV_IDX = pd.bdate_range("2026-01-05", periods=16)
DIV_PRICE = pd.Series([2, 3, 6, 3, 2, 1, 2, 3, 7, 3, 2, 0.5, 2, 3, 4, 5.0], index=DIV_IDX)
DIV_RSI = pd.Series([50, 55, 80, 60, 40, 30, 45, 55, 70, 55, 40, 35, 45, 55, 60, 65.0], index=DIV_IDX)


def positions(idx):
    return [DIV_IDX.get_loc(d) for d in idx]


def test_pivots_are_swing_extremes_and_never_the_last_k_bars():
    highs, lows = t.find_pivots(DIV_PRICE, 2)
    assert positions(highs) == [2, 8] and positions(lows) == [5, 11]
    # the rise at the end (positions 12-15) is not a pivot: it cannot be confirmed yet
    assert all(p < len(DIV_PRICE) - 2 for p in positions(highs) + positions(lows))


def test_flat_top_keeps_only_the_first_bar():
    s = pd.Series([1, 2, 5, 5, 5, 2, 1, 2, 3, 2, 1.0], index=pd.bdate_range("2026-01-05", periods=11))
    highs, _ = t.find_pivots(s, 2)
    # the 5-5-5 plateau (positions 2-4) yields one pivot, at its first bar; position 8 is a
    # genuine separate swing high
    assert [s.index.get_loc(d) for d in highs] == [2, 8]


def test_bearish_and_bullish_divergences_are_found_and_dated_at_the_second_swing():
    d = t.rsi_divergences(DIV_PRICE, DIV_RSI, k=2, min_gap=3, max_gap=20)
    assert list(d["kind"]) == ["bearish", "bullish"]
    bear, bull = d.iloc[0], d.iloc[1]
    assert bear["date"] == DIV_IDX[8] and bear["prev_date"] == DIV_IDX[2]
    assert (bear["price"], bear["prev_price"], bear["rsi"], bear["prev_rsi"]) == (7.0, 6.0, 70.0, 80.0)
    assert bull["date"] == DIV_IDX[11] and (bull["price"], bull["prev_price"]) == (0.5, 1.0)
    assert (bull["rsi"], bull["prev_rsi"]) == (35.0, 30.0)


def test_no_divergence_when_rsi_confirms_the_move():
    confirming = DIV_RSI.copy()
    confirming.iloc[8] = 90.0   # higher high AND higher RSI high: confirmation, not divergence
    confirming.iloc[11] = 20.0  # lower low AND lower RSI low
    assert t.rsi_divergences(DIV_PRICE, confirming, k=2, min_gap=3, max_gap=20).empty


def test_swings_too_close_or_too_far_apart_are_not_compared():
    assert t.rsi_divergences(DIV_PRICE, DIV_RSI, k=2, min_gap=7, max_gap=20).empty  # 6 bars apart
    assert t.rsi_divergences(DIV_PRICE, DIV_RSI, k=2, min_gap=3, max_gap=5).empty


def test_divergence_skips_swings_where_rsi_is_not_yet_available():
    early = DIV_RSI.copy()
    early.iloc[:6] = np.nan  # RSI warm-up covers the first swings
    d = t.rsi_divergences(DIV_PRICE, early, k=2, min_gap=3, max_gap=20)
    assert d.empty or "bearish" not in set(d["kind"])


# ------------------------------------------------------------ loading and fallback


def test_slice_lookback_measures_back_from_latest():
    idx = pd.date_range("2020-01-01", "2026-01-05", freq="B")
    df = pd.DataFrame({"10Y": 1.0}, index=idx)
    assert t.slice_lookback(df, "1Y").index[0] >= ts("2025-01-05")
    assert len(t.slice_lookback(df, "Max")) == len(df)


def test_fred_fallback_columns_and_holiday_gaps(monkeypatch):
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])

    def fake_get_series(series_id, api_key=None, **kw):
        vals = [np.nan, 3.0, 3.1]  # FRED reports holidays as missing
        return pd.DataFrame({"date": dates, "value": vals})

    monkeypatch.setattr(t.pf, "get_series", fake_get_series)
    df = t.fetch_fred_curve("dummy", 2026)
    assert list(df.columns) == ["1M", "3M", "6M", "1Y", "2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
    assert df.index[0] == ts("2026-01-02")  # all-NaN holiday row dropped, nothing filled


def test_load_falls_back_to_fred_and_reports_why(monkeypatch):
    fred = pd.DataFrame({"10Y": [4.2]}, index=pd.to_datetime(["2026-01-05"]))

    def boom(*a, **k):
        raise t.TreasuryError("2026: ConnectTimeout")

    monkeypatch.setattr(t, "_cached_treasury", boom)
    monkeypatch.setattr(t, "_cached_fred", lambda key, year: fred)
    data = t.load_yield_curve(5, api_key="dummy")
    assert data.fallback and data.source == t.SOURCE_FRED
    assert "ConnectTimeout" in data.error and data.as_of == ts("2026-01-05")


def test_load_strips_excluded_tenor_even_from_a_stale_cached_frame(monkeypatch):
    stale = pd.DataFrame(
        {"1M": [4.0], "1.5M": [4.1], "2M": [4.2], "3M": [4.3]}, index=pd.to_datetime(["2026-01-05"])
    )
    monkeypatch.setattr(t, "_cached_treasury", lambda year: stale)
    data = t.load_yield_curve(5)
    assert list(data.df.columns) == ["1M", "2M", "3M"] and not data.fallback


def test_load_never_raises_when_everything_fails(monkeypatch):
    def boom(*a, **k):
        raise t.TreasuryError("down")

    monkeypatch.setattr(t, "_cached_treasury", boom)
    monkeypatch.setattr(t, "_cached_fred", boom)
    data = t.load_yield_curve(5, api_key="dummy")
    assert data.df.empty and data.as_of is None and "down" in data.error
    no_key = t.load_yield_curve(5, api_key=None)  # no FRED key: still a clean result
    assert no_key.df.empty
