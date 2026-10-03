"""Download Cleveland Fed monthly CPI / Core CPI nowcasts (m/m % change) to CSV."""

import pandas as pd
import requests

URL = "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json"


def last_value(series: dict) -> float | None:
    """Last non-empty value in a FusionCharts series."""
    vals = [d["value"] for d in series["data"] if d.get("value") not in ("", None)]
    return float(vals[-1]) if vals else None


def fetch(years: int = 10) -> pd.DataFrame:
    charts = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).json()
    rows = []
    for chart in charts:
        by_name = {s["seriesname"]: s for s in chart["dataset"]}
        rows.append(
            {
                "month": pd.Period(chart["chart"]["subcaption"], freq="M"),
                "cpi_nowcast": last_value(by_name["CPI Inflation"]),
                "core_cpi_nowcast": last_value(by_name["Core CPI Inflation"]),
                "cpi_actual": last_value(by_name["Actual CPI Inflation"]),
                "core_cpi_actual": last_value(by_name["Actual Core CPI Inflation"]),
            }
        )
    df = pd.DataFrame(rows).sort_values("month")
    cutoff = pd.Period(pd.Timestamp.today(), freq="M") - years * 12
    return df[df["month"] > cutoff].reset_index(drop=True)


if __name__ == "__main__":
    df = fetch()
    df.to_csv("nowcast_monthly.csv", index=False)
    print(df.to_string())
