from pathlib import Path
import pandas as pd

from .data_cleaning import prepare_core_frame
from .forecasting import prepare_country_series, arima_forecast
from .risk_index import build_country_risk
from .utils import print_section
from .config import PROJECT_ROOT

# Directory for saving reports
REPORTS_DIR = PROJECT_ROOT / "reports" / "country_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_country_report(country: str, horizon: int = 10):
    """
    Generate a Markdown report for a given country.

    Includes:
    - Basic emission overview (with units)
    - Climate risk index (0–100, unitless)
    - ARIMA forecast table with emissions in tons
    """
    print_section(f"Generating CO₂ report for {country}")

    # Load cleaned data
    df = prepare_core_frame()

    # Build risk index table
    risk_df = build_country_risk(df)

    # Get risk row for this country
    risk_row = risk_df[risk_df["country"] == country]
    if risk_row.empty:
        raise ValueError(f"Country '{country}' not found in risk index table.")
    risk_row = risk_row.iloc[0]

    # Prepare time series for ARIMA
    series = prepare_country_series(df, country)
    if series.empty:
        raise ValueError(f"No emission time series found for '{country}'.")

    # ARIMA forecast
    mean, lower, upper = arima_forecast(series, steps=horizon)

    last_year = int(series.index.max())
    start_year = last_year + 1
    forecast_years = list(range(start_year, start_year + horizon))

    forecast_df = pd.DataFrame(
        {
            "year": forecast_years,
            "forecast": mean.values,
            "lower": lower.values,
            "upper": upper.values,
        }
    )

    # Add units to forecast columns
    forecast_display = forecast_df.rename(
        columns={
            "forecast": "forecast (tons)",
            "lower": "lower (tons)",
            "upper": "upper (tons)",
        }
    )

    # Build report file path
    report_path = REPORTS_DIR / f"{country.replace(' ', '_')}_CO2_report.md"

    # Write Markdown report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# {country} – CO₂ Emission & Climate Risk Report\n\n")

        f.write("## 1. Overview\n\n")
        f.write(f"* **Years covered:** {int(series.index.min())} – {int(series.index.max())}\n")
        f.write(f"* **Total CO₂ emissions over period:** {series.sum():,.2f} tons\n")
        f.write(f"* **Last recorded year ({last_year}) emissions:** {series.iloc[-1]:,.2f} tons\n\n")

        f.write("## 2. Climate Risk Index\n\n")
        f.write(
            "The climate risk index is a unitless composite score in the range "
            "0–100, combining emission level, per-capita emissions, and simple "
            "growth behaviour.\n\n"
        )
        f.write(f"* **Climate risk index:** {risk_row['climate_risk_index']:.2f} / 100\n\n")

        f.write("**Risk components (normalized to [0, 1]):**\n\n")
        f.write(f"* Emission level score (relative): {risk_row['score_level']:.3f}\n")
        f.write(f"* Per-capita emission score (relative): {risk_row['score_per_capita']:.3f}\n")
        f.write(f"* Growth score (relative): {risk_row['score_growth']:.3f}\n\n")

        f.write("## 3. ARIMA Forecast of CO₂ Emissions\n\n")
        f.write(
            f"The table below shows the ARIMA-based forecast for the next "
            f"{horizon} years. All emissions are in **tons of CO₂**.\n\n"
        )

        # Forecast table as Markdown
        f.write(forecast_display.to_markdown(index=False))
        f.write("\n")

    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    # Example usage: generate a report for Germany
    generate_country_report("Bangladesh", horizon=10)
