# src/model_comparison.py

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# ---- Metrics ----
def mae(y, y2): return float(np.mean(np.abs(y - y2)))
def rmse(y, y2): return float(np.sqrt(np.mean((y - y2) ** 2)))
def mape(y, y2):
    mask = y != 0
    if not mask.any(): return np.nan
    return float(np.mean(np.abs((y[mask] - y2[mask]) / y[mask])) * 100)
def smape(y, y2):
    denom = (np.abs(y) + np.abs(y2) + 1e-9)
    return float(np.mean(2 * np.abs(y - y2) / denom) * 100)


# ---- Split ----
def split_series(series, test_years=5, val_years=5):
    n = len(series)
    if n <= val_years + test_years + 5:
        raise ValueError("Series too small for splitting.")
    train = series.iloc[:-(val_years + test_years)]
    val   = series.iloc[-(val_years + test_years):-test_years]
    test  = series.iloc[-test_years:]
    return train, val, test


# ---- ARIMA Tuning ----
def tune_arima(train, val, p_vals=(0,1,2,3), d_vals=(0,1,2), q_vals=(0,1,2,3)):
    best = (1,1,1); best_err = np.inf
    yv = val.values.astype(float)

    for p in p_vals:
        for d in d_vals:
            for q in q_vals:
                try:
                    model = ARIMA(train, order=(p,d,q)).fit()
                    pred = model.forecast(len(val)).values.astype(float)
                    err = rmse(yv, pred)
                    if err < best_err:
                        best = (p,d,q); best_err = err
                except:
                    pass
    return best


# ---- Main function ----
def evaluate_models_for_country(series, country, test_years=5, val_years=5):
    """
    ARIMA-only evaluation.
    """
    train, val, test = split_series(series, test_years, val_years)

    best_order = tune_arima(train, val)
    model = ARIMA(pd.concat([train, val]), order=best_order).fit()
    pred  = model.forecast(len(test)).values.astype(float)
    ytest = test.values.astype(float)

    return {
        "ARIMA": {
            "order": best_order,
            "MAE": mae(ytest, pred),
            "RMSE": rmse(ytest, pred),
            "MAPE": mape(ytest, pred),
            "SMAPE": smape(ytest, pred),
            "pred": pred,
        }
    }
