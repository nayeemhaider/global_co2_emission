# CO₂ Emission Intelligence Suite

This project is an end-to-end analytics and forecasting toolkit built on a global **CO₂ emissions by country** dataset from Kaggle.

It is designed as a portfolio-ready, consultancy-style project that shows how to go from **raw CSV** to:

- Cleaned and standardized dataset  
- Exploratory data analysis (EDA)  
- Time-series forecasting (ARIMA, Prophet, LSTM)  
- Country-level clustering  
- Global anomaly detection  
- Climate risk scoring  
- Automated country reports  
- An interactive Streamlit dashboard  

# 1. Project Structure

global_co2_emission/
    data/
        raw/
            CO2 emission by countries.csv
        processed/
        external/
        dashboard_image
    src/
        config.py
        utils.py
        data_cleaning.py
        eda.py
        forecasting.py
        clustering.py
        anomaly.py
        risk_index.py
        report_generator.py
        train_models.py
        train_prophet_models.py
        train_lstm_models.py
        dashboard_app/
            app.py
    models/
        arima/
            arima_model.py
            <Country>_arima.pkl   (after training)
        prophet/
            prophet_model.py
            <Country>_prophet.pkl (after training)
        lstm/
            lstm_model.py
            <Country>_lstm.pt     (after training)
    notebooks/
        01 eda.ipynb
        02 forecasting.ipynb
        03 clustering.ipynb
        04 anomaly_detection.ipynb
        05 risk_index.ipynb
    reports/
        country_reports/
        global_summary/
    README.md

# 2. Installation

Create and activate a virtual environment (optional but recommended), then install dependencies:

pip install -r requirements.txt

A minimal requirements.txt could contain:

pandas
numpy
matplotlib
seaborn
statsmodels
joblib
scikit-learn
streamlit
torch
prophet

prophet and torch are only required if you want Prophet and LSTM forecasting.

# 3. Data

Place your CO₂ dataset in:

data/raw/CO2 emission by countries.csv

The cleaning script expects columns that can be transformed into:

country

year

co2_emission_tons

population2022

area

_of_world

densitykm2

After cleaning, these are standardized to:

country

year

emission

population2022

area

share_of_world

density_km2

# 4. Core Features
## 4.1 Data Cleaning

src/data_cleaning.py

from src.data_cleaning import prepare_core_frame

df = prepare_core_frame()
print(df.head())


This function:

Loads the raw CSV

Cleans column names

Standardizes key columns

Ensures numeric types for year and emission

## 4.2 Exploratory Data Analysis (EDA)

Run via script:

python -m src.eda


Or use the notebook:

notebooks/01_eda.ipynb


Functions in src/eda.py provide:

Basic descriptive statistics

Missing values and duplicates report

Global emission trend

Top emitters over time

Heatmap of emissions for top countries

## 4.3 Forecasting (ARIMA / Prophet / LSTM)

ARIMA and utility functions are in src/forecasting.py.
Model wrappers live in models/arima/, models/prophet/, models/lstm/.

Example (notebook 02_forecasting.ipynb):

from src.data_cleaning import prepare_core_frame
from src.forecasting import prepare_country_series, arima_forecast

df = prepare_core_frame()
series = prepare_country_series(df, "Germany")
mean, lower, upper = arima_forecast(series, steps=10)


To pre-train and save models for all countries:

python -m src.train_models           # ARIMA per country
python -m src.train_prophet_models   # Prophet per country (requires prophet)
python -m src.train_lstm_models      # LSTM global


Saved models appear in:

models/arima/<Country>_arima.pkl
models/prophet/<Country>_prophet.pkl
models/lstm/<Country>_lstm.pt


The Streamlit dashboard will use these if present, otherwise it will train a model on-the-fly for the selected country.

## 4.4 Clustering

Notebook: 03_clustering.ipynb
Script: src/clustering.py

from src.clustering import run_clustering

df_clustered = run_clustering(n_clusters=4)


This:

Aggregates country features (total, mean, min, max emission, trend)

Standardizes features

Runs KMeans clustering

Returns a table of countries with cluster labels

## 4.5 Anomaly Detection

Notebook: 04_anomaly_detection.ipynb
Script: src/anomaly.py

from src.anomaly import run_anomaly_detection

yearly = run_anomaly_detection()


This module:

Aggregates global emissions per year

Computes z-score anomalies

Uses an IsolationForest model

Flags anomalous years

## 4.6 Climate Risk Index

Notebook: 05_risk_index.ipynb
Script: src/risk_index.py

from src.risk_index import run_risk_index

risk_df = run_risk_index()


It computes for each country:

Emission level score

Per-capita-like score

Growth score

Combined climate_risk_index (0–100)

## 4.7 Country Reports

src/report_generator.py

from src.report_generator import generate_country_report

generate_country_report("Germany", horizon=10)


This creates a Markdown report in:

reports/country_reports/Germany_CO2_report.md


The report includes:

Overview of emissions and time span

Climate risk index and its components

ARIMA forecast table for future years

## 4.8 Streamlit Dashboard

src/dashboard_app/app.py

Start the dashboard from the project root:

streamlit run src/dashboard_app/app.py


Dashboard features:

Sidebar controls:

Select country

Forecast horizon (years)

Forecast model: ARIMA / Prophet / LSTM (depending on availability)

Main view:

Basic stats for selected country

Historical emission line chart

Forecast line chart (history + forecast)

Forecast result table

Country clustering table (KMeans)

Top climate risk index table

Risk details for the selected country

# 5. Notebooks Overview

All notebooks live in notebooks/ and call into the src/ modules:

    01 eda.ipynb – data exploration and plots

    02 forecasting.ipynb – ARIMA/Prophet forecasting for a chosen country

    03 clustering.ipynb – clustering countries and exploring cluster profiles

    04 anomaly_detection.ipynb – global anomaly detection over time

    05 risk_index.ipynb – climate risk ranking and visualization

They are meant to be readable, teaching-style analyses that reuse the core code.

# Dashboard Overview
![Dashboard Preview](./data/dashboard_image/1.JPG)
![Top 10 Emitters](./data/dashboard_image/2.JPG)
![Particular Country Emission History](./data/dashboard_image/3.JPG)
![Forecast Preview](./data/dashboard_image/4.JPG)
![Risk Index](./data/dashboard_image/5.JPG)


# To use
# For complete EDA: 
    python -m src.eda from terminal 
    or via notebook notebooks/01_eda.ipynb

# Forecasting (ARIMA / Prophet / LSTM)
    ARIMA models:
        python -m src.train_models
 
    Prophet models:
        python -m src.train_prophet_models

    LSTM models:
        python -m src.train_lstm_models
    
    Or via notebook notebooks/02_forecasting.ipynb

# Clustering countries
    Notebook: notebooks/03_clustering.ipynb

# Anomaly detection
    Notebook: notebooks/04_anomaly_detection.ipynb

# Generate country reports
    python -m src.report_generator
    To view: reports/country_reports/Germany_CO2_report.md

# Run the Streamlit dashboard
    streamlit run src/dashboard_app/app.py







