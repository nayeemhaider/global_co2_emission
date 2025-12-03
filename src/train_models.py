import warnings
import joblib
from typing import Dict

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .data_cleaning import prepare_core_frame
from .config import ARIMA_MODELS_DIR
from .utils import print_section, wrap_print


def prepare_country_series(df: pd.DataFrame, country: str) -> pd.Series:
    sub = df[df["country"] == country].copy()
    sub = (
        sub.groupby("year")["emission"]
        .sum()
        .sort_values()
    )
    return sub


def fit_arima_for_country(series: pd.Series, order=(1, 1, 1)):
    model = ARIMA(series, order=order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit()
    return fitted


def train_arima_all_countries(order=(1, 1, 1)) -> Dict[str, str]:
    print_section("Training ARIMA models for all countries")
    df = prepare_core_frame()

    countries = sorted(df["country"].unique())
    saved_paths = {}

    for country in countries:
        series = prepare_country_series(df, country)
        if len(series) < 5:
            # too short, skip
            continue

        wrap_print(f"Training ARIMA{order} for {country} with {len(series)} years of data.")
        try:
            model = fit_arima_for_country(series, order=order)
        except Exception as e:
            print(f"Failed to fit model for {country}: {e}")
            continue

        country_safe = country.replace(" ", "_").replace("/", "_")
        model_path = ARIMA_MODELS_DIR / f"{country_safe}_arima.pkl"
        joblib.dump(model, model_path)
        saved_paths[country] = str(model_path)

    print_section("ARIMA training completed")
    print(f"Models saved for {len(saved_paths)} countries in: {ARIMA_MODELS_DIR}")
    return saved_paths


if __name__ == "__main__":
    train_arima_all_countries(order=(1, 1, 1))
