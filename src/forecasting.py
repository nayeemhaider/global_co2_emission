import warnings
from typing import Tuple

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .data_cleaning import prepare_core_frame
from .utils import print_section

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

def prepare_country_series(df: pd.DataFrame, country: str) -> pd.Series:
    sub = df[df["country"] == country].copy()
    sub = (
        sub.groupby("year")["emission"]
        .sum()
        .sort_values()
    )
    return sub

def arima_forecast(series: pd.Series, steps: int = 10) -> Tuple[pd.Series, pd.Series, pd.Series]:
    model = ARIMA(series, order=(1, 1, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = model.fit()
    forecast_res = fitted.get_forecast(steps=steps)
    mean = forecast_res.predicted_mean
    ci = forecast_res.conf_int()
    lower = ci.iloc[:, 0]
    upper = ci.iloc[:, 1]
    return mean, lower, upper

def prophet_forecast(series: pd.Series, periods: int = 10) -> pd.DataFrame:
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet not installed. Install with `pip install prophet`.")
    df_prophet = (
        series.reset_index()
        .rename(columns={"year": "ds", "emission": "y"})
    )
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods, freq="Y")
    forecast = model.predict(future)
    return forecast

def demo_forecast(country: str = "Germany"):
    df = prepare_core_frame()
    print_section(f"Forecast demo for {country}")
    series = prepare_country_series(df, country)

    print("ARIMA forecast:")
    mean, lower, upper = arima_forecast(series, steps=10)
    print(mean)

    if PROPHET_AVAILABLE:
        print("\nProphet forecast:")
        forecast = prophet_forecast(series, periods=10)
        print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))

if __name__ == "__main__":
    demo_forecast("Germany")
