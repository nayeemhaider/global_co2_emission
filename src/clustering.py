import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data_cleaning import prepare_core_frame
from .utils import print_section

def build_country_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per country for clustering."""
    agg = (
        df.groupby("country")
        .agg(
            total_emission=("emission", "sum"),
            mean_emission=("emission", "mean"),
            max_emission=("emission", "max"),
            min_emission=("emission", "min"),
            first_year=("year", "min"),
            last_year=("year", "max"),
        )
        .reset_index()
    )
    agg["emission_trend"] = agg["max_emission"] - agg["min_emission"]
    return agg

def kmeans_cluster_countries(df_features: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    feats = df_features.drop(columns=["country"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feats)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)

    df_features = df_features.copy()
    df_features["cluster"] = labels
    return df_features

def run_clustering(n_clusters: int = 4) -> pd.DataFrame:
    print_section("Country Clustering")
    df = prepare_core_frame()
    df_features = build_country_features(df)
    df_clustered = kmeans_cluster_countries(df_features, n_clusters=n_clusters)
    print(df_clustered.head(20))
    return df_clustered

if __name__ == "__main__":
    run_clustering()
