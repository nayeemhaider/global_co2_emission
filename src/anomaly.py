import pandas as pd
from sklearn.ensemble import IsolationForest

from .data_cleaning import prepare_core_frame
from .utils import print_section

def compute_yearly_series(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.groupby("year")["emission"]
        .sum()
        .reset_index()
        .sort_values("year")
    )
    return yearly

def zscore_anomalies(yearly: pd.DataFrame, threshold: float = 2.5) -> pd.DataFrame:
    mean = yearly["emission"].mean()
    std = yearly["emission"].std()
    yearly["z_score"] = (yearly["emission"] - mean) / std
    yearly["is_anomaly_z"] = yearly["z_score"].abs() > threshold
    return yearly

def isolation_forest_anomalies(yearly: pd.DataFrame) -> pd.DataFrame:
    iso = IsolationForest(contamination=0.05, random_state=42)
    yearly = yearly.copy()
    yearly["iso_label"] = iso.fit_predict(yearly[["emission"]])
    yearly["is_anomaly_iso"] = yearly["iso_label"] == -1
    return yearly

def run_anomaly_detection():
    print_section("Anomaly Detection")
    df = prepare_core_frame()
    yearly = compute_yearly_series(df)
    yearly = zscore_anomalies(yearly, threshold=2.5)
    yearly = isolation_forest_anomalies(yearly)

    print("\nAnomalies by z-score:")
    print(yearly[yearly["is_anomaly_z"]])

    print("\nAnomalies by Isolation Forest:")
    print(yearly[yearly["is_anomaly_iso"]])

    return yearly

if __name__ == "__main__":
    run_anomaly_detection()
