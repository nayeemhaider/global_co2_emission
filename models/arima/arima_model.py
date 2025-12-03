# models/arima/arima_model.py

import warnings
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper


class ARIMACountryModel:
    """
    Wrapper for ARIMA model per country.
    Handles fit, forecast, save, and load.
    """

    def __init__(self, country: str, order=(1, 1, 1)):
        self.country = country
        self.order = order
        self.model = None
        self.fitted: ARIMAResultsWrapper | None = None

    def fit(self, series: pd.Series):
        """
        series: indexed by year, values = emissions
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = ARIMA(series, order=self.order)
            self.fitted = self.model.fit()
        return self

    def forecast(self, steps: int = 10) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Returns (mean, lower, upper)
        """
        if self.fitted is None:
            raise RuntimeError("Model not fitted yet.")
        forecast_res = self.fitted.get_forecast(steps=steps)
        mean = forecast_res.predicted_mean
        ci = forecast_res.conf_int()
        lower = ci.iloc[:, 0]
        upper = ci.iloc[:, 1]
        return mean, lower, upper

    def save(self, path: Path):
        """
        Save in a dict format so we can reload with metadata.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "country": self.country,
                "order": self.order,
                "fitted": self.fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "ARIMACountryModel":
        obj = joblib.load(path)

        # Legacy case: file contains just an ARIMAResultsWrapper
        if isinstance(obj, ARIMAResultsWrapper):
            # Try to recover order from the underlying model
            try:
                order = obj.model.order
            except Exception:
                order = (1, 1, 1)

            # Infer country from filename (everything before "_arima.pkl")
            stem = path.stem  # e.g. "Germany_arima"
            country = stem.replace("_arima", "")

            inst = cls(country=country, order=order)
            inst.fitted = obj
            return inst

        # New format: dict
        if isinstance(obj, dict):
            country = obj.get("country", "Unknown")
            order = obj.get("order", (1, 1, 1))
            fitted = obj.get("fitted", None)

            inst = cls(country=country, order=order)
            inst.fitted = fitted
            return inst

        # Fallback: unexpected format
        raise TypeError(
            f"Unsupported ARIMA model file format at {path}. "
            f"Got object of type: {type(obj)}"
        )
