import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Use a non-interactive backend for Streamlit
plt.switch_backend("Agg")

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[2]  # project root: .../co2_emission
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Project imports
from src.data_cleaning import prepare_core_frame
from src.clustering import run_clustering
from src.risk_index import build_country_risk
from src.forecasting import prepare_country_series, arima_forecast
from src.config import ARIMA_MODELS_DIR
from src.model_comparison import evaluate_models_for_country

from models.arima.arima_model import ARIMACountryModel


# Helper functions
def safe_country_filename(country: str) -> str:
    return country.replace(" ", "_").replace("/", "_")


def arima_forecast_dashboard(series: pd.Series, country: str, horizon: int) -> pd.DataFrame:
    """
    Use saved ARIMA model if available, otherwise fit and save it.
    Returns a forecast dataframe with columns: year, forecast, lower, upper.
    """
    safe_name = safe_country_filename(country)
    model_path = ARIMA_MODELS_DIR / f"{safe_name}_arima.pkl"

    if model_path.exists():
        model = ARIMACountryModel.load(model_path)
    else:
        model = ARIMACountryModel(country=country, order=(1, 1, 1))
        model.fit(series)
        model.save(model_path)

    mean, lower, upper = model.forecast(steps=horizon)
    start_year = int(series.index.max()) + 1
    years = list(range(start_year, start_year + horizon))

    forecast_df = pd.DataFrame(
        {
            "year": years,
            "forecast": mean.values,
            "lower": lower.values,
            "upper": upper.values,
        }
    )
    return forecast_df


def forecast_router(series: pd.Series, country: str, horizon: int) -> pd.DataFrame:
    """
    Router kept for extensibility, but currently always uses ARIMA.
    """
    return arima_forecast_dashboard(series, country, horizon)

# Streamlit app
def main():
    st.title("Global CO₂ Emission Intelligence Dashboard (ARIMA-only)")

    # Load cleaned data
    df = prepare_core_frame()
    countries = sorted(df["country"].unique())

    # Sidebar controls
    st.sidebar.header("Controls")

    country = st.sidebar.selectbox(
        "Select a country",
        countries,
        index=countries.index("Germany") if "Germany" in countries else 0,
    )

    horizon = st.sidebar.slider(
        "Forecast horizon (years)",
        min_value=5,
        max_value=30,
        value=10,
    )

    test_years = st.sidebar.slider(
        "Test years for ARIMA evaluation",
        min_value=3,
        max_value=15,
        value=8,
        help="Number of last years held out for ARIMA model evaluation.",
    )

    # Prepare common slices
    sub = df[df["country"] == country].copy()
    yearly_country = (
        sub.groupby("year")["emission"]
        .sum()
        .reset_index()
        .sort_values("year")
    )

    # Tabs
    tab_eda, tab_forecast, tab_clusters, tab_compare = st.tabs(
        ["EDA", "Forecast (ARIMA)", "Clusters & Risk", "Model Comparison (ARIMA)"]
    )

    # EDA TAB
    with tab_eda:
        st.header("Exploratory Data Analysis")

        st.subheader("Global overview")
        global_yearly = (
            df.groupby("year")["emission"]
            .sum()
            .reset_index()
            .sort_values("year")
        )
        st.write("Global CO₂ emissions over time (tons):")
        st.line_chart(global_yearly.set_index("year")["emission"])

        latest_year = int(df["year"].max())
        df_latest = df[df["year"] == latest_year]
        top_emitters = (
            df_latest.groupby("country")["emission"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )

        st.subheader(f"Top 10 emitters in {latest_year} (CO₂ emissions in tons)")
        st.bar_chart(top_emitters.set_index("country")["emission"])

        st.subheader(f"Per-country summary for {country}")
        display_sub = sub.rename(
            columns={
                "emission": "emission (tons)",
                "population2022": "population (people)",
                "area": "area (km²)",
                "density_km2": "density (people/km²)",
                "share_of_world": "share_of_world (fraction of world)",
            }
        )
        st.write(display_sub.describe(include="all"))

        st.subheader(f"{country} CO₂ emission history (tons)")
        st.line_chart(yearly_country.set_index("year")["emission"])

    # FORECAST TAB (ARIMA)
    with tab_forecast:
        st.header("Forecast (ARIMA)")

        st.subheader("Emission history (CO₂ emissions in tons)")
        st.line_chart(yearly_country.set_index("year")["emission"])

        st.subheader(f"ARIMA forecast for {country} (CO₂ emissions in tons)")
        series = prepare_country_series(df, country)

        forecast_df = forecast_router(series, country, horizon)
        if forecast_df is not None:
            history_df = yearly_country.copy()
            history_df["type"] = "history"
            fc = forecast_df.copy()
            fc["type"] = "forecast"

            combined = pd.concat(
                [
                    history_df.rename(columns={"emission": "value"})[
                        ["year", "value", "type"]
                    ],
                    fc.rename(columns={"forecast": "value"})[
                        ["year", "value", "type"]
                    ],
                ],
                ignore_index=True,
            )

            st.line_chart(
                combined.pivot(index="year", columns="type", values="value")
            )

            st.write("Forecast table (CO₂ emissions in tons):")
            display_fc = forecast_df.rename(
                columns={
                    "forecast": "forecast (tons)",
                    "lower": "lower (tons)",
                    "upper": "upper (tons)",
                }
            )
            st.dataframe(display_fc)

    # CLUSTERS & RISK TAB
    with tab_clusters:
        st.header("Clusters & Risk")

        st.subheader("Country clusters (based on CO₂ emission statistics)")
        df_clustered = run_clustering(n_clusters=4)
        st.dataframe(df_clustered.head(20))

        st.subheader("Climate risk index (0–100, unitless composite)")
        risk_df = build_country_risk(df)
        st.dataframe(risk_df.head(20))

        st.write(f"Selected country risk details ({country}):")
        st.dataframe(risk_df[risk_df["country"] == country])

    # MODEL COMPARISON TAB (ARIMA ONLY)
    with tab_compare:
        st.header(f"ARIMA Model Evaluation for {country}")

        series = prepare_country_series(df, country)

        # Simple guard: need enough data for val + test
        if len(series) <= test_years + 5 + 2:
            st.warning(
                f"Time series for {country} is too short for "
                f"val+test split with test_years={test_years}. "
                "Try reducing the test years or select another country."
            )
        else:
            try:
                results = evaluate_models_for_country(
                    series, country, test_years=test_years, val_years=5
                )
            except Exception as e:
                st.error(f"Error during ARIMA evaluation: {e}")
                results = None

            if results and "ARIMA" in results and "MAE" in results["ARIMA"]:
                arima_res = results["ARIMA"]

                # Metrics table
                st.subheader("Error metrics (ARIMA)")
                metrics_df = pd.DataFrame(
                    {
                        "MAE (tons)": [arima_res["MAE"]],
                        "RMSE (tons)": [arima_res["RMSE"]],
                        "MAPE (%)": [arima_res["MAPE"]],
                        "SMAPE (%)": [arima_res["SMAPE"]],
                        "order (p,d,q)": [str(arima_res.get("order", ""))],
                    },
                    index=["ARIMA"],
                )
                st.dataframe(metrics_df)

                # Test segment for plotting
                test = series.iloc[-test_years:]
                years_test = test.index.values
                preds = arima_res["pred"]

                st.subheader("Actual vs predicted (CO₂ emissions in tons)")
                fig1, ax1 = plt.subplots(figsize=(10, 4))
                ax1.plot(test.index, test.values, marker="o", label="Actual")
                ax1.plot(years_test, preds, marker="o", label="ARIMA")

                ax1.set_xlabel("Year")
                ax1.set_ylabel("CO₂ emissions (tons)")
                ax1.grid(True)
                ax1.legend()
                fig1.tight_layout()
                st.pyplot(fig1)

                st.subheader("Absolute error per year (tons)")
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                errors = np.abs(test.values - preds)
                ax2.plot(years_test, errors, marker="o", label="ARIMA error")

                ax2.set_xlabel("Year")
                ax2.set_ylabel("Absolute error (tons)")
                ax2.grid(True)
                ax2.legend()
                fig2.tight_layout()
                st.pyplot(fig2)
            else:
                st.info("No valid ARIMA evaluation results available yet.")


if __name__ == "__main__":
    main()
