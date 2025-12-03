import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure matplotlib works fine in Streamlit
plt.switch_backend("Agg")

# Make sure Python can see the project root (where src/ and models/ live)
ROOT = Path(__file__).resolve().parents[2]  # project root: .../co2_emission
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_cleaning import prepare_core_frame
from src.clustering import run_clustering
from src.risk_index import build_country_risk
from src.forecasting import prepare_country_series, arima_forecast
from src.config import ARIMA_MODELS_DIR, PROPHET_MODELS_DIR, LSTM_MODELS_DIR
from src.model_comparison import evaluate_models_for_country

from models.arima.arima_model import ARIMACountryModel

# Prophet wrapper for forecast tab
try:
    from models.prophet.prophet_model import ProphetCountryModel, PROPHET_AVAILABLE
except Exception:
    PROPHET_AVAILABLE = False
    ProphetCountryModel = None  # type: ignore

# Global LSTM wrapper for forecast and comparison
try:
    from models.lstm.lstm_model import load_global_lstm, global_lstm_forecast_country
    GLOBAL_LSTM_PATH = LSTM_MODELS_DIR / "global_lstm.pt"
    LSTM_AVAILABLE = GLOBAL_LSTM_PATH.exists()
except Exception:
    GLOBAL_LSTM_PATH = None
    LSTM_AVAILABLE = False

GLOBAL_LSTM_CACHE = None  # lazy-loaded cache: (model, mean, std, country_to_idx)


def safe_country_filename(country: str) -> str:
    return country.replace(" ", "_").replace("/", "_")


def arima_forecast_dashboard(series: pd.Series, country: str, horizon: int) -> pd.DataFrame:
    """
    Use saved ARIMA model if available, otherwise fit and save.
    Returns forecast dataframe with columns: year, forecast, lower, upper.
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


def prophet_forecast_dashboard(series: pd.Series, country: str, horizon: int):
    """
    Use saved Prophet model if available, otherwise fit and save.
    Returns forecast dataframe with columns: year, forecast, lower, upper.
    """
    if not PROPHET_AVAILABLE or ProphetCountryModel is None:
        st.warning("Prophet is not available. Install it or check your environment.")
        return None

    safe_name = safe_country_filename(country)
    model_path = PROPHET_MODELS_DIR / f"{safe_name}_prophet.pkl"

    if model_path.exists():
        model = ProphetCountryModel.load(model_path)
    else:
        model = ProphetCountryModel(country=country)
        model.fit(series)
        model.save(model_path)

    forecast = model.forecast(periods=horizon)
    forecast_tail = forecast.tail(horizon).copy()
    forecast_tail["year"] = forecast_tail["ds"].dt.year

    forecast_df = forecast_tail[["year", "yhat", "yhat_lower", "yhat_upper"]].rename(
        columns={
            "yhat": "forecast",
            "yhat_lower": "lower",
            "yhat_upper": "upper",
        }
    )
    return forecast_df


def lstm_forecast_dashboard(series: pd.Series, country: str, horizon: int, seq_len: int = 5):
    """
    Use pre-trained global LSTM model (one model for all countries) to forecast.
    """
    global GLOBAL_LSTM_CACHE

    if not LSTM_AVAILABLE or GLOBAL_LSTM_PATH is None or not GLOBAL_LSTM_PATH.exists():
        st.warning(
            "Global LSTM model not available. "
            "Run `python -m src.train_lstm_model` first to train it."
        )
        return None

    # Lazy-load global LSTM
    if GLOBAL_LSTM_CACHE is None:
        model, mean, std, country_to_idx = load_global_lstm(GLOBAL_LSTM_PATH)
        GLOBAL_LSTM_CACHE = (model, mean, std, country_to_idx)
    else:
        model, mean, std, country_to_idx = GLOBAL_LSTM_CACHE

    values = series.values.astype(float)

    try:
        preds = global_lstm_forecast_country(
            series=values,
            country=country,
            model=model,
            mean=mean,
            std=std,
            country_to_idx=country_to_idx,
            horizon=horizon,
            seq_len=seq_len,
        )
    except ValueError as e:
        st.warning(str(e))
        return None

    last_year = int(series.index.max())
    years = list(range(last_year + 1, last_year + 1 + horizon))

    forecast_df = pd.DataFrame(
        {
            "year": years,
            "forecast": preds,
            "lower": preds,  # no CI for LSTM
            "upper": preds,
        }
    )
    return forecast_df


def forecast_router(series: pd.Series, country: str, horizon: int, model_choice: str):
    """
    Dispatch to ARIMA, Prophet or LSTM forecast functions based on choice.
    """
    if model_choice == "ARIMA":
        return arima_forecast_dashboard(series, country, horizon)
    elif model_choice == "Prophet":
        return prophet_forecast_dashboard(series, country, horizon)
    elif model_choice == "LSTM":
        return lstm_forecast_dashboard(series, country, horizon)
    else:
        # fallback: simple ARIMA from src.forecasting
        mean, lower, upper = arima_forecast(series, steps=horizon)
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


def main():
    st.title("Global CO₂ Emission Intelligence Dashboard")

    df = prepare_core_frame()
    countries = sorted(df["country"].unique())

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
        "Test years for model comparison",
        min_value=3,
        max_value=15,
        value=8,
        help="How many last years to hold out as test for ARIMA vs Prophet vs LSTM comparison.",
    )

    model_options = ["ARIMA"]
    if PROPHET_AVAILABLE and ProphetCountryModel is not None:
        model_options.append("Prophet")
    if LSTM_AVAILABLE and GLOBAL_LSTM_PATH is not None and GLOBAL_LSTM_PATH.exists():
        model_options.append("LSTM")

    model_choice = st.sidebar.radio(
        "Forecast model (Forecast tab)",
        model_options,
        index=0,
        help="Model used in the Forecast tab for forward prediction.",
    )

    sub = df[df["country"] == country].copy()
    yearly_country = (
        sub.groupby("year")["emission"]
        .sum()
        .reset_index()
        .sort_values("year")
    )

    tab_eda, tab_forecast, tab_clusters, tab_compare = st.tabs(
        ["EDA", "Forecast", "Clusters & Risk", "Model Comparison"]
    )

    # EDA tab
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
        st.bar_chart(
            data=top_emitters.set_index("country")["emission"]
        )

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

    # Forecast tab
    with tab_forecast:
        st.header("Forecast")

        st.subheader("Emission history (CO₂ emissions in tons)")
        st.line_chart(yearly_country.set_index("year")["emission"])

        st.subheader(f"{model_choice} forecast for {country} (CO₂ emissions in tons)")
        series = prepare_country_series(df, country)

        forecast_df = forecast_router(series, country, horizon, model_choice)
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

    # Clusters & Risk tab
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

    # Model Comparison tab
    with tab_compare:
        st.header(f"Model Comparison for {country}")

        series = prepare_country_series(df, country)

        if len(series) <= test_years + 5 + 2:
            st.warning(
                f"Time series for {country} is too short for "
                f"val+test split with test_years={test_years}. "
                "Try reducing the test years or pick another country."
            )
        else:
            try:
                results = evaluate_models_for_country(
                    series, country, test_years=test_years, val_years=5
                )
            except Exception as e:
                st.error(f"Error during model comparison: {e}")
                results = None

            if results:
                metrics_table = {}
                for model_name, res in results.items():
                    if "MAE" in res:
                        metrics_table[model_name] = {
                            "MAE (tons)": res["MAE"],
                            "RMSE (tons)": res["RMSE"],
                            "MAPE (%)": res["MAPE"],
                            "SMAPE (%)": res["SMAPE"],
                        }

                if metrics_table:
                    st.subheader("Error metrics (lower is better)")
                    st.dataframe(
                        pd.DataFrame(metrics_table)
                        .T.style.format("{:.3f}")
                    )
                else:
                    st.info("No valid metrics to display yet.")

                # Test segment and plot configs
                test = series.iloc[-test_years:]
                years_test = test.index.values

                if any("pred" in res for res in results.values()):
                    st.subheader("Actual vs predicted (CO₂ emissions in tons)")
                    fig1, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.plot(
                        test.index,
                        test.values,
                        marker="o",
                        label="Actual",
                    )

                    for model_name, res in results.items():
                        if "pred" in res:
                            ax1.plot(
                                years_test,
                                res["pred"],
                                marker="o",
                                label=model_name,
                            )

                    ax1.set_xlabel("Year")
                    ax1.set_ylabel("CO₂ emissions (tons)")
                    ax1.grid(True)
                    ax1.legend()
                    fig1.tight_layout()
                    st.pyplot(fig1)

                    st.subheader("Absolute error per year (tons)")
                    fig2, ax2 = plt.subplots(figsize=(10, 4))
                    for model_name, res in results.items():
                        if "pred" in res:
                            errors = np.abs(test.values - res["pred"])
                            ax2.plot(
                                years_test,
                                errors,
                                marker="o",
                                label=f"{model_name} error",
                            )

                    ax2.set_xlabel("Year")
                    ax2.set_ylabel("Absolute error (tons)")
                    ax2.grid(True)
                    ax2.legend()
                    fig2.tight_layout()
                    st.pyplot(fig2)


if __name__ == "__main__":
    main()
