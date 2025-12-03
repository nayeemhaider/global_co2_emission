import numpy as np
import pandas as pd
from pathlib import Path

from statsmodels.tsa.arima.model import ARIMA

# Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

from src.config import LSTM_MODELS_DIR

# Global LSTM (optional)
try:
    from models.lstm.lstm_model import load_global_lstm, global_lstm_forecast_country
    GLOBAL_LSTM_PATH = LSTM_MODELS_DIR / "global_lstm.pt"
    LSTM_AVAILABLE = GLOBAL_LSTM_PATH.exists()
except Exception:
    LSTM_AVAILABLE = False
    GLOBAL_LSTM_PATH = None


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = y_true != 0
    if nonzero.sum() == 0:
        return np.nan
    return float(
        np.mean(
            np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])
        )
        * 100
    )


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    return float(
        np.mean(
            2 * np.abs(y_pred - y_true) / denom
        )
        * 100
    )


def train_val_test_split_series(series: pd.Series, val_years: int, test_years: int):
    """
    Split time series into train, validation and test.
    Validation and test are taken from the end of the series.
    """
    n = len(series)
    if n <= val_years + test_years + 2:
        raise ValueError(
            f"Series too short for requested val={val_years}, "
            f"test={test_years} (len={n})."
        )

    train = series.iloc[: -(val_years + test_years)]
    val = series.iloc[-(val_years + test_years): -test_years]
    test = series.iloc[-test_years:]
    return train, val, test


def tune_arima_order(
    train: pd.Series,
    val: pd.Series,
    p_values=(0, 1, 2, 3),
    d_values=(0, 1, 2),
    q_values=(0, 1, 2, 3),
) -> tuple:
    """
    Small grid search over ARIMA(p,d,q) using validation RMSE.
    Keeps search space compact to be safe for yearly data.
    """
    best_order = (1, 1, 1)
    best_rmse = np.inf
    y_val = val.values.astype(float)

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    model = ARIMA(train, order=order)
                    fitted = model.fit()
                    fc = fitted.forecast(steps=len(val))
                    y_pred = fc.values.astype(float)
                    err = rmse(y_val, y_pred)
                except Exception:
                    continue

                if err < best_rmse:
                    best_rmse = err
                    best_order = order

    return best_order


def arima_forecast_with_tuning(series: pd.Series, val_years: int, test_years: int):
    """
    Tune ARIMA order on validation, then refit on train+val and forecast test.
    Returns predicted values for test and best (p,d,q).
    """
    train, val, test = train_val_test_split_series(series, val_years, test_years)
    best_order = tune_arima_order(train, val)
    series_train_val = pd.concat([train, val])

    model = ARIMA(series_train_val, order=best_order)
    fitted = model.fit()
    fc = fitted.forecast(steps=len(test))
    pred_test = fc.values.astype(float)

    return pred_test, best_order


def prophet_forecast_with_tuning(series: pd.Series, val_years: int, test_years: int):
    """
    Tune Prophet changepoint_prior_scale on validation RMSE,
    then refit on train+val and forecast test.
    Returns test predictions and best CPS value.
    """
    if not PROPHET_AVAILABLE:
        return None, None

    train, val, test = train_val_test_split_series(series, val_years, test_years)

    def to_prophet_df(s: pd.Series) -> pd.DataFrame:
        df = s.reset_index()
        df = df.rename(columns={"year": "ds", s.name: "y"})
        df["ds"] = pd.to_datetime(df["ds"], format="%Y")
        return df

    series = series.copy()
    series.name = "emission"

    train_df = to_prophet_df(train)
    val_df = to_prophet_df(val)

    cps_grid = [0.01, 0.1, 0.3]
    best_cps = 0.1
    best_rmse = np.inf

    for cps in cps_grid:
        try:
            m = Prophet(changepoint_prior_scale=cps)
            m.fit(train_df)

            horizon = len(val)
            future = m.make_future_dataframe(periods=horizon, freq="Y")
            fc = m.predict(future)
            fc_val = fc.tail(horizon)

            y_pred = fc_val["yhat"].values.astype(float)
            y_true = val.values.astype(float)
            err = rmse(y_true, y_pred)
        except Exception:
            continue

        if err < best_rmse:
            best_rmse = err
            best_cps = cps

    train_val = pd.concat([train, val])
    train_val_df = to_prophet_df(train_val)

    m_best = Prophet(changepoint_prior_scale=best_cps)
    m_best.fit(train_val_df)

    horizon_test = len(test)
    future = m_best.make_future_dataframe(periods=horizon_test, freq="Y")
    fc_full = m_best.predict(future)
    fc_test = fc_full.tail(horizon_test)

    pred_test = fc_test["yhat"].values.astype(float)
    return pred_test, best_cps


def evaluate_models_for_country(
    series: pd.Series,
    country: str,
    test_years: int = 5,
    val_years: int = 5,
):
    """
    Compare ARIMA (tuned), Prophet (tuned), and global LSTM (if available)
    for a single country's emission time series.

    Returns a dict of
      {
        "ARIMA": {...},
        "Prophet": {...},
        "LSTM_global": {...}
      }
    """
    results = {}

    series = series.copy()
    series.name = "emission"

    train, val, test = train_val_test_split_series(series, val_years, test_years)
    y_test = test.values.astype(float)

    # ARIMA
    try:
        pred_arima, best_order = arima_forecast_with_tuning(series, val_years, test_years)
        results["ARIMA"] = {
            "order": best_order,
            "MAE": mae(y_test, pred_arima),
            "RMSE": rmse(y_test, pred_arima),
            "MAPE": mape(y_test, pred_arima),
            "SMAPE": smape(y_test, pred_arima),
            "pred": pred_arima,
        }
    except Exception as e:
        results["ARIMA"] = {"error": str(e)}

    # Prophet
    if PROPHET_AVAILABLE:
        try:
            pred_prophet, best_cps = prophet_forecast_with_tuning(series, val_years, test_years)
            if pred_prophet is not None:
                results["Prophet"] = {
                    "changepoint_prior_scale": best_cps,
                    "MAE": mae(y_test, pred_prophet),
                    "RMSE": rmse(y_test, pred_prophet),
                    "MAPE": mape(y_test, pred_prophet),
                    "SMAPE": smape(y_test, pred_prophet),
                    "pred": pred_prophet,
                }
        except Exception as e:
            results["Prophet"] = {"error": str(e)}

    # Global LSTM
    if LSTM_AVAILABLE and GLOBAL_LSTM_PATH is not None and GLOBAL_LSTM_PATH.exists():
        try:
            model, mean, std, country_to_idx = load_global_lstm(GLOBAL_LSTM_PATH)
            train_val = pd.concat([train, val])

            preds_lstm = global_lstm_forecast_country(
                series=train_val.values.astype(float),
                country=country,
                model=model,
                mean=mean,
                std=std,
                country_to_idx=country_to_idx,
                horizon=len(test),
                seq_len=5,
            )

            results["LSTM_global"] = {
                "MAE": mae(y_test, preds_lstm),
                "RMSE": rmse(y_test, preds_lstm),
                "MAPE": mape(y_test, preds_lstm),
                "SMAPE": smape(y_test, preds_lstm),
                "pred": preds_lstm,
            }
        except Exception as e:
            results["LSTM_global"] = {"error": str(e)}

    return results
