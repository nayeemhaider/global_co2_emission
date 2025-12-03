from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"

RAW_CO2_FILE = DATA_RAW / "CO2 emission by countries.csv"

# Core columns (after cleaning)
COL_COUNTRY = "country"
COL_YEAR = "year"
COL_EMISSION = "co2_emission_tons"
COL_POP = "population2022"
COL_AREA = "area"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
ARIMA_MODELS_DIR = MODELS_DIR / "arima"
PROPHET_MODELS_DIR = MODELS_DIR / "prophet"
LSTM_MODELS_DIR = MODELS_DIR / "lstm"

# Ensure directories exist (optional convenience)
for d in [MODELS_DIR, ARIMA_MODELS_DIR, PROPHET_MODELS_DIR, LSTM_MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
