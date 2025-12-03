from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetCountryModel:
    """
    Wrapper around Prophet for a single country.
    """

    def __init__(self, country: str):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Run `pip install prophet`.")
        self.country = country
        self.model: Optional[Prophet] = None

    def fit(self, series: pd.Series):
        """
        series: index = year (int), values = emissions
        Converted to Prophet dataframe with ds/y.
        """
        df_prophet = (
            series.reset_index()
            .rename(columns={"year": "ds", "emission": "y"})
        )
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")
        self.model = Prophet()
        self.model.fit(df_prophet)
        return self

    def forecast(self, periods: int = 10) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        future = self.model.make_future_dataframe(periods=periods, freq="Y")
        forecast = self.model.predict(future)
        return forecast

    def save(self, path: Path):
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "country": self.country,
                "model": self.model,
            },
            path
        )

    @classmethod
    def load(cls, path: Path) -> "ProphetCountryModel":
        obj = joblib.load(path)
        inst = cls(country=obj["country"])
        inst.model = obj["model"]
        return inst
