from typing import Dict

import pandas as pd

from .data_cleaning import prepare_core_frame
from .config import PROPHET_MODELS_DIR
from .utils import print_section, wrap_print
from models.prophet.prophet_model import ProphetCountryModel, PROPHET_AVAILABLE


def prepare_country_series(df: pd.DataFrame, country: str) -> pd.Series:
    sub = df[df["country"] == country].copy()
    sub = (
        sub.groupby("year")["emission"]
        .sum()
        .sort_values()
    )
    sub.index.name = "year"
    sub.name = "emission"
    return sub


def train_prophet_all_countries() -> Dict[str, str]:
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet not installed. Run `pip install prophet` first.")

    print_section("Training Prophet models for all countries")
    df = prepare_core_frame()
    countries = sorted(df["country"].unique())
    saved_paths = {}

    for country in countries:
        series = prepare_country_series(df, country)
        if len(series) < 5:
            continue

        wrap_print(f"Training Prophet for {country} with {len(series)} years of data.")
        try:
            model = ProphetCountryModel(country=country)
            model.fit(series)
        except Exception as e:
            print(f"Failed Prophet model for {country}: {e}")
            continue

        safe_name = country.replace(" ", "_").replace("/", "_")
        model_path = PROPHET_MODELS_DIR / f"{safe_name}_prophet.pkl"
        model.save(model_path)
        saved_paths[country] = str(model_path)

    print_section("Prophet training completed")
    print(f"Models saved for {len(saved_paths)} countries in: {PROPHET_MODELS_DIR}")
    return saved_paths


if __name__ == "__main__":
    train_prophet_all_countries()
