import pandas as pd

from .data_cleaning import prepare_core_frame
from .utils import print_section

def build_country_risk(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("country")
        .agg(
            total_emission=("emission", "sum"),
            last_emission=("emission", "last"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            population2022=("population2022", "max"),
        )
        .reset_index()
    )

    agg["years_span"] = agg["last_year"] - agg["first_year"] + 1
    agg["emission_per_year"] = agg["total_emission"] / agg["years_span"]
    agg["emission_per_capita"] = agg["last_emission"] / agg["population2022"].replace(0, pd.NA)
    agg["emission_per_capita"] = agg["emission_per_capita"].fillna(0)

    # Simple growth proxy
    agg["emission_growth_rate"] = agg["last_emission"] / agg["emission_per_year"].replace(0, pd.NA)
    agg["emission_growth_rate"] = agg["emission_growth_rate"].fillna(1.0)

    # Normalise features to [0,1] for scoring
    def norm(col):
        cmin, cmax = col.min(), col.max()
        if cmax == cmin:
            return col * 0
        return (col - cmin) / (cmax - cmin)

    agg["score_level"] = norm(agg["last_emission"])
    agg["score_per_capita"] = norm(agg["emission_per_capita"])
    agg["score_growth"] = norm(agg["emission_growth_rate"])

    # Weighted risk index
    agg["climate_risk_index"] = (
        0.5 * agg["score_level"]
        + 0.3 * agg["score_per_capita"]
        + 0.2 * agg["score_growth"]
    ) * 100

    agg.sort_values("climate_risk_index", ascending=False, inplace=True)
    return agg

def run_risk_index():
    print_section("Climate Risk Index")
    df = prepare_core_frame()
    risk_df = build_country_risk(df)
    print(risk_df.head(20))
    return risk_df

if __name__ == "__main__":
    run_risk_index()
