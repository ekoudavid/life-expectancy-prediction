"""
Data preparation for the WHO Life Expectancy dataset (Kaggle, 2000-2015).

The cleaning steps are the same ones developed and documented in the
exploratory analysis. They are collected here as functions so every
notebook uses exactly the same, tested pipeline.

Usage
-----
    from src.data_prep import load_raw, clean

    df_raw = load_raw("data/Life Expectancy Data.csv")
    df, log = clean(df_raw)                           # fully cleaned (for analysis)
    df, log = clean(df_raw, fill_group_median=False,   # leaves cross-country gaps for a
                    fix_alcohol_placeholders=True)     # model pipeline to fill (avoids leakage)
"""

import numpy as np
import pandas as pd

# Columns removed as unreliable (see README: "Data cleaning")
UNRELIABLE_COLUMNS = ["population", "bmi", "percentage_expenditure", "measles"]


def load_raw(path):
    """Read the original CSV without changing anything."""
    return pd.read_csv(path)


def standardise_column_names(df):
    """Strip stray spaces, lower-case, and replace spaces/hyphens/slashes with '_'."""
    df = df.copy()
    new_cols = df.columns.str.strip()
    new_cols = new_cols.str.lower()
    new_cols = new_cols.str.replace(r"[\s\-/]+", "_", regex=True)
    df.columns = new_cols
    # The dataset documentation describes this column as ages 10-19, not 1-19
    df = df.rename(columns={"thinness_1_19_years": "thinness_10_19_years"})
    return df


def set_impossible_values_to_missing(df):
    """Replace values that are impossible given each variable's definition with NaN.

    Returns the updated table and a dictionary of how many values each rule changed.
    """
    df = df.copy()
    count_cols = ["infant_deaths", "under_five_deaths", "measles"]
    df[count_cols] = df[count_cols].astype(float)   # whole-number columns cannot hold NaN

    rules = {
        "infant_deaths": df["infant_deaths"] > 1000,          # a rate per 1,000 cannot exceed 1,000
        "under_five_deaths": df["under_five_deaths"] > 1000,
        "measles": df["measles"] > 1000,
        "adult_mortality": df["adult_mortality"] < 20,        # far below any real country's rate
        "schooling": df["schooling"] == 0,                    # 0 means "not recorded"
        "income_composition_of_resources": df["income_composition_of_resources"] == 0,
        "hepatitis_b": df["hepatitis_b"] > 100,               # coverage cannot exceed 100%
        "polio": df["polio"] > 100,
        "diphtheria": df["diphtheria"] > 100,
    }
    changed = {}
    for column, is_invalid in rules.items():
        changed[column] = int(is_invalid.sum())
        df.loc[is_invalid, column] = np.nan
    return df, changed


def set_alcohol_placeholders_to_missing(df):
    """Treat alcohol = 0.01 as a missing-value code in countries that normally drink more.

    The raw data records many 2012-2015 values as exactly 0.01 litres even for countries
    that consumed around 10 litres a year before (e.g. Uganda). A country whose typical
    (median) consumption is above 0.5 litres cannot genuinely drop to 0.01, so those
    values are set to missing. Countries that genuinely drink almost nothing keep 0.01.
    """
    df = df.copy()
    typical = df.groupby("country")["alcohol"].transform("median")
    is_placeholder = (df["alcohol"] <= 0.01) & (typical > 0.5)
    df.loc[is_placeholder, "alcohol"] = np.nan
    return df, int(is_placeholder.sum())


def fill_from_neighbouring_years(values):
    """Linear interpolation across a single country's years (also fills the ends)."""
    return values.interpolate(limit_direction="both")


def clean(df_raw, fill_group_median=True, fix_alcohol_placeholders=False):
    """Run the full cleaning pipeline.

    Parameters
    ----------
    df_raw : the original table from load_raw()
    fill_group_median : if True, any value still missing after within-country
        interpolation is filled with the Developed/Developing median (as in the
        exploratory analysis). Set to False for modelling, so the remaining gaps
        are filled inside the model pipeline using training data only.
    fix_alcohol_placeholders : if True, alcohol values of 0.01 in countries that
        normally drink more are treated as missing (see set_alcohol_placeholders_to_missing).
        Off by default so the output matches the exploratory analysis exactly.

    Returns
    -------
    df  : cleaned table
    log : list of sentences describing each step and its effect
    """
    log = []

    df = standardise_column_names(df_raw)
    log.append("Standardised column names")

    df["country"] = df["country"].str.strip()
    df["status"] = df["status"].str.strip().str.title()

    n_dups = int(df.duplicated(subset=["country", "year"]).sum())
    df = df.drop_duplicates(subset=["country", "year"])
    log.append(f"Removed {n_dups} duplicate country-year rows")

    df["year"] = df["year"].astype(int)

    rows_per_country = df.groupby("country").size()
    single_year = rows_per_country[rows_per_country == 1].index
    df = df[df["country"].isin(single_year) == False].copy()
    log.append(f"Removed {len(single_year)} countries with only one year of data")

    df, changed = set_impossible_values_to_missing(df)
    log.append(f"Set {sum(changed.values())} impossible values to missing")

    if fix_alcohol_placeholders:
        df, n_alcohol = set_alcohol_placeholders_to_missing(df)
        log.append(f"Set {n_alcohol} alcohol placeholder values (0.01) to missing")

    df = df.drop(columns=UNRELIABLE_COLUMNS)
    log.append("Dropped unreliable columns: " + ", ".join(UNRELIABLE_COLUMNS))

    df = df.dropna(subset=["life_expectancy"])

    num_cols = df.select_dtypes(include="number").columns.drop("year")
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    df[num_cols] = df.groupby("country")[num_cols].transform(fill_from_neighbouring_years)
    log.append(f"Interpolated within each country; {int(df[num_cols].isnull().sum().sum())} values still missing")

    if fill_group_median:
        medians = df.groupby("status", observed=True)[num_cols].transform("median")
        df[num_cols] = df[num_cols].fillna(medians)
        log.append("Filled remaining gaps with the Developed/Developing median")

    df["log_gdp"] = np.log10(df["gdp"])
    df["log_hiv_aids"] = np.log10(df["hiv_aids"])
    df["status"] = df["status"].astype("category")
    log.append("Added log10 versions of GDP and HIV/AIDS deaths")

    return df, log
