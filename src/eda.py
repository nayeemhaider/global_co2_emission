import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .data_cleaning import prepare_core_frame
from .utils import print_section, wrap_print, set_plot_style

def basic_overview(df: pd.DataFrame):
    print_section("Basic Overview")
    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)
    print("\nHead:\n", df.head(10))
    print("\nDescribe:\n", df.describe().T)

def missing_values_report(df: pd.DataFrame):
    print_section("Missing Values")
    missing = df.isna().sum()
    pct = (missing / len(df)) * 100
    report = (
        pd.DataFrame(
            {"missing_count": missing, "missing_percent": pct.round(2)}
        )
        .sort_values("missing_percent", ascending=False)
    )
    print(report)

def duplicates_report(df: pd.DataFrame):
    print_section("Duplicates")
    dup = df.duplicated().sum()
    print(f"Duplicate rows: {dup}")

def general_stats(df: pd.DataFrame):
    print_section("General Stats")

    n_countries = df["country"].nunique()
    yr_min = df["year"].min()
    yr_max = df["year"].max()
    total_em = df["emission"].sum()

    print(f"Unique countries: {n_countries}")
    print(f"Year range: {yr_min} to {yr_max}")
    print(f"Total emissions: {total_em:,.2f}")

    top_countries = (
        df.groupby("country")["emission"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop 10 countries by total emissions:\n", top_countries)

    by_year = (
        df.groupby("year")["emission"]
        .sum()
        .sort_values()
    )
    print("\nTotal emissions by year (first 10 rows):\n", by_year.head(10))

def plot_global_trend(df: pd.DataFrame):
    print_section("Plot: Global Trend")
    yearly = (
        df.groupby("year")["emission"]
        .sum()
        .reset_index()
        .sort_values("year")
    )
    plt.figure()
    plt.plot(yearly["year"], yearly["emission"], marker="o")
    plt.title("Global CO2 Emissions Over Time")
    plt.xlabel("Year")
    plt.ylabel("Total CO2 Emissions (tons)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_top_countries_trend(df: pd.DataFrame, top_n: int = 5):
    print_section(f"Plot: Top {top_n} Countries Trends")
    total_by_country = (
        df.groupby("country")["emission"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    top_countries = total_by_country.index.tolist()
    df_top = df[df["country"].isin(top_countries)].copy()
    df_top = (
        df_top.groupby(["year", "country"])["emission"]
        .sum()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_top,
        x="year",
        y="emission",
        hue="country",
        marker="o",
    )
    plt.title(f"CO2 Emissions Over Time for Top {top_n} Countries")
    plt.xlabel("Year")
    plt.ylabel("CO2 Emissions (tons)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_latest_year_top_emitters(df: pd.DataFrame, top_n: int = 10):
    print_section("Plot: Top Emitters in Latest Year")
    latest_year = df["year"].max()
    df_latest = df[df["year"] == latest_year].copy()
    by_country = (
        df_latest.groupby("country")["emission"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=by_country,
        x="emission",
        y="country"
    )
    plt.title(f"Top {top_n} Emitters in {latest_year}")
    plt.xlabel("CO2 Emissions (tons)")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.show()

def heatmap_country_year(df: pd.DataFrame, top_n: int = 15):
    print_section("Heatmap: Country vs Year")
    total_by_country = (
        df.groupby("country")["emission"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )
    top_countries = total_by_country.index.tolist()
    df_top = df[df["country"].isin(top_countries)].copy()
    pivot = df_top.pivot_table(
        index="country",
        columns="year",
        values="emission",
        aggfunc="sum"
    )

    plt.figure(figsize=(14, 6))
    sns.heatmap(pivot, cmap="viridis", linewidths=0.1)
    plt.title(f"Heatmap of CO2 emissions (tons) for Top {top_n} Countries")
    plt.xlabel("Year")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.show()

def run_full_eda():
    set_plot_style()
    df = prepare_core_frame()
    wrap_print("Running full EDA pipeline for CO2 dataset.")
    basic_overview(df)
    missing_values_report(df)
    duplicates_report(df)
    general_stats(df)
    plot_global_trend(df)
    plot_top_countries_trend(df, top_n=5)
    plot_latest_year_top_emitters(df, top_n=10)
    heatmap_country_year(df, top_n=15)

if __name__ == "__main__":
    run_full_eda()
