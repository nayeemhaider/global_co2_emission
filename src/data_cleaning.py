import os
import pandas as pd
from .config import RAW_CO2_FILE, COL_COUNTRY, COL_YEAR, COL_EMISSION
from .utils import print_section

def load_raw_co2() -> pd.DataFrame:
    if not RAW_CO2_FILE.exists():
        raise FileNotFoundError(f"Raw CO2 file not found at: {RAW_CO2_FILE}")
    print_section("Loading Raw CO2 Dataset")
    df = pd.read_csv(RAW_CO2_FILE, encoding="ISO-8859-1")
    return df

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace(r"[^0-9a-zA-Z_]", "", regex=True)
        .str.lower()
    )
    return df

def prepare_core_frame() -> pd.DataFrame:
    """Return a cleaned dataframe with core columns standardized."""
    df_raw = load_raw_co2()
    df = clean_columns(df_raw)

    # Basic sanity print
    print("Cleaned columns:", df.columns.tolist())

    # Ensure required columns exist
    required = [COL_COUNTRY, COL_YEAR, COL_EMISSION]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {df.columns.tolist()}"
        )

    df = df[[COL_COUNTRY, COL_YEAR, COL_EMISSION, "population2022", "area", "_of_world", "densitykm2"]].copy()

    # Types
    df[COL_YEAR] = pd.to_numeric(df[COL_YEAR], errors="coerce")
    df[COL_EMISSION] = pd.to_numeric(df[COL_EMISSION], errors="coerce")

    df = df.dropna(subset=[COL_COUNTRY, COL_YEAR, COL_EMISSION])

    # Standard names for downstream modules
    df.rename(
        columns={
            COL_COUNTRY: "country",
            COL_YEAR: "year",
            COL_EMISSION: "emission",
            "population2022": "population2022",
            "area": "area",
            "_of_world": "share_of_world",
            "densitykm2": "density_km2"
        },
        inplace=True,
    )

    return df
