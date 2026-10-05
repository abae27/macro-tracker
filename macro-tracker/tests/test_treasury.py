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
    stale = pd.DataFrame({"1M": [4.0], "1.5M": [4.1], "2M": [4.2]}, index=pd.to_datetime(["2026-01-05"]))
    monkeypatch.setattr(t, "_cached_treasury", lambda year: stale)
    data = t.load_yield_curve(5)
    assert list(data.df.columns) == ["1M", "2M"] and not data.fallback


def test_load_never_raises_when_everything_fails(monkeypatch):
    def boom(*a, **k):
        raise t.TreasuryError("down")

    monkeypatch.setattr(t, "_cached_treasury", boom)
    monkeypatch.setattr(t, "_cached_fred", boom)
    data = t.load_yield_curve(5, api_key="dummy")
    assert data.df.empty and data.as_of is None and "down" in data.error
    no_key = t.load_yield_curve(5, api_key=None)  # no FRED key: still a clean result
    assert no_key.df.empty
