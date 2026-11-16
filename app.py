"""
Life Expectancy Explorer: a Streamlit app for the life expectancy model.

Run locally:   streamlit run app.py
"""
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from src.data_prep import clean, load_raw

st.set_page_config(page_title="Life Expectancy Explorer", page_icon="🌍", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Life Expectancy Data.csv")
REPO_URL = "https://github.com/ekoudavid/life-expectancy-prediction"

# The development indicators the model uses (no mortality variables: see "About")
FEATURES = ["schooling", "income_composition_of_resources", "log_gdp", "total_expenditure",
            "alcohol", "hepatitis_b", "polio", "diphtheria", "thinness_10_19_years", "developed"]

NICE = {"schooling": "Schooling (years)", "income_composition_of_resources": "Income composition (HDI, 0-1)",
        "log_gdp": "GDP per capita", "total_expenditure": "Health spending (% of govt. spending)",
        "alcohol": "Alcohol consumption (litres/person)", "hepatitis_b": "Hepatitis B coverage (%)",
        "polio": "Polio coverage (%)", "diphtheria": "Diphtheria coverage (%)",
        "thinness_10_19_years": "Thinness, ages 10-19 (%)", "developed": "Development status"}

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"


# ---------------------------------------------------------------- data and model
@st.cache_data
def load_data():
    """Clean the WHO data with the same pipeline as the notebook."""
    df, _ = clean(load_raw(DATA_PATH), fill_group_median=False, fix_alcohol_placeholders=True)
    df["developed"] = (df["status"] == "Developed").astype(int)
    return df


@st.cache_resource
def train_models():
    """Train 5 random forests, each on 80% of countries (5-fold grouped cross-validation).

    Every country falls in exactly one held-out fold, so the estimate shown for it comes
    from a model that never saw that country: an honest, out-of-sample prediction.
    """
    df = load_data()
    models, fold_of_country = [], {}
    honest_prediction = pd.Series(np.nan, index=df.index)

    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=5).split(df, groups=df["country"])):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
        )
        model.fit(df.iloc[train_idx][FEATURES], df.iloc[train_idx]["life_expectancy"])
        honest_prediction.iloc[test_idx] = model.predict(df.iloc[test_idx][FEATURES])
        models.append(model)
        for country in df.iloc[test_idx]["country"].unique():
            fold_of_country[country] = fold

    return models, fold_of_country, honest_prediction


df = load_data()
models, fold_of_country, honest_prediction = train_models()
overall_error = mean_absolute_error(df["life_expectancy"], honest_prediction)
typical = df[FEATURES].median()   # "typical" value of each indicator across all countries and years


def predict(model, values):
    """Predict life expectancy for one set of indicator values (a dictionary)."""
    return float(model.predict(pd.DataFrame([values])[FEATURES])[0])


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Choose a country")
countries = sorted(df["country"].unique())
country = st.sidebar.selectbox("Country", countries, index=countries.index("Uganda"))
years = sorted(df.loc[df["country"] == country, "year"].unique())
year = st.sidebar.select_slider("Year", options=years, value=years[-1])

row = df[(df["country"] == country) & (df["year"] == year)].iloc[0]
model = models[fold_of_country[country]]   # the model that never saw this country
key = f"{country}_{year}"                  # sliders reset when the country or year changes

if st.sidebar.button("Reset sliders to actual values"):
    for k in list(st.session_state.keys()):
        if k.endswith(key):
            del st.session_state[k]

st.sidebar.caption(
    f"Each estimate comes from a model trained on other countries only. "
    f"Across all {df['country'].nunique()} countries, its average error is {overall_error:.1f} years."
)

# ---------------------------------------------------------------- header
st.title("🌍 Life Expectancy Explorer")
st.markdown(
    "How well do a country's **development indicators** predict its life expectancy? "
    "Pick a country, then move the sliders to explore *what if* its indicators were different. "
    "Model trained on WHO data for 183 countries, 2000–2015."
)


# ---------------------------------------------------------------- what-if inputs
def starting_value(feature):
    """The country's actual value, or the typical value if it was not recorded."""
    value = row[feature]
    return (float(typical[feature]), True) if pd.isna(value) else (float(value), False)


st.subheader(f"What if? {country}, {year}")
st.caption("Sliders start at the country's real values. Values marked * were not recorded "
           "and start at the typical value across all countries.")

inputs, missing_marks = {}, []
col_a, col_b = st.columns(2)

with col_a:
    v, miss = starting_value("income_composition_of_resources")
    inputs["income_composition_of_resources"] = st.slider(
        NICE["income_composition_of_resources"] + (" *" if miss else ""), 0.0, 1.0, round(v, 3), 0.005, key="icr_" + key)
    v, miss = starting_value("schooling")
    inputs["schooling"] = st.slider(NICE["schooling"] + (" *" if miss else ""), 0.0, 21.0, round(v, 1), 0.1, key="sch_" + key)
    v, miss = starting_value("log_gdp")
    gdp = st.number_input(NICE["log_gdp"] + " (US$)" + (" *" if miss else ""), min_value=10, max_value=150000,
                          value=int(round(10 ** v)), step=100, key="gdp_" + key)
    inputs["log_gdp"] = np.log10(gdp)   # the model uses GDP on a log scale
    v, miss = starting_value("total_expenditure")
    inputs["total_expenditure"] = st.slider(NICE["total_expenditure"] + (" *" if miss else ""), 0.0, 18.0, round(v, 1), 0.1, key="exp_" + key)
    status = st.radio(NICE["developed"], ["Developing", "Developed"], index=int(row["developed"]),
                      horizontal=True, key="dev_" + key)
    inputs["developed"] = 1 if status == "Developed" else 0

with col_b:
    for feature, top, step in [("hepatitis_b", 100.0, 1.0), ("polio", 100.0, 1.0), ("diphtheria", 100.0, 1.0),
                               ("thinness_10_19_years", 30.0, 0.1), ("alcohol", 18.0, 0.1)]:
        v, miss = starting_value(feature)
        inputs[feature] = st.slider(NICE[feature] + (" *" if miss else ""), 0.0, top, round(v, 1), step, key=feature + "_" + key)

# ---------------------------------------------------------------- results
actual = float(row["life_expectancy"])
estimate = float(honest_prediction.loc[row.name])
what_if = predict(model, inputs)

m1, m2, m3 = st.columns(3)
m1.metric("Actual life expectancy", f"{actual:.1f} years")
m2.metric("Model estimate (real indicators)", f"{estimate:.1f} years", f"{estimate - actual:+.1f} vs actual",
          delta_color="off")
m3.metric("What-if estimate (your sliders)", f"{what_if:.1f} years", f"{what_if - estimate:+.1f} vs model estimate")

left, right = st.columns([3, 2])

# Chart: actual vs model estimate over time, with the what-if point
with left:
    history = df[df["country"] == country][["year", "life_expectancy"]].copy()
    history["Model estimate"] = honest_prediction.loc[history.index].values
    history = history.rename(columns={"life_expectancy": "Actual"})
    long = history.melt("year", var_name="Series", value_name="Life expectancy (years)")
    lines = alt.Chart(long).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("year:O", title="Year"),
        y=alt.Y("Life expectancy (years):Q", scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=alt.Scale(domain=["Actual", "Model estimate"], range=[BLUE, GREY]),
                        legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=["Actual", "Model estimate"], range=[[1, 0], [5, 4]]),
                                  legend=None),
        tooltip=["year:O", "Series:N", alt.Tooltip("Life expectancy (years):Q", format=".1f")],
    )
    point = alt.Chart(pd.DataFrame({"year": [year], "What-if": [what_if]})).mark_point(
        shape="diamond", size=180, filled=True, color=ORANGE).encode(
        x="year:O", y=alt.Y("What-if:Q", title="Life expectancy (years)"),
        tooltip=[alt.Tooltip("What-if:Q", format=".1f")])
    st.markdown(f"**{country}: actual vs model estimate, 2000–2015** (orange diamond = your what-if)")
    st.altair_chart(lines + point, width="stretch")

# Chart: which indicators push this what-if estimate up or down
with right:
    effects = []
    for feature in FEATURES:
        changed = dict(inputs)
        changed[feature] = typical[feature]   # put just this indicator back to the typical value
        effects.append({"Indicator": NICE[feature].split(" (")[0],
                        "Effect (years)": what_if - predict(model, changed)})
    effects = pd.DataFrame(effects)
    effects["Direction"] = np.where(effects["Effect (years)"] >= 0, "Raises estimate", "Lowers estimate")
    bars = alt.Chart(effects).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Effect (years):Q", title="Years added (+) or removed (−)"),
        y=alt.Y("Indicator:N", sort=alt.EncodingSortField("Effect (years)", op="sum", order="descending"),
                title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("Direction:N", scale=alt.Scale(domain=["Raises estimate", "Lowers estimate"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Indicator:N", alt.Tooltip("Effect (years):Q", format="+.2f")],
    )
    st.markdown("**What drives the what-if estimate**")
    st.altair_chart(bars, width="stretch")
    st.caption("Each bar shows how the estimate would change if that one indicator were at the "
               "typical value for all countries, with everything else as set.")

st.info("This model shows **associations, not cause and effect**. Raising one indicator on a slider "
        "does not mean that change alone would add those years in reality.", icon="ℹ️")

# ---------------------------------------------------------------- about
with st.expander("About this model"):
    st.markdown(f"""
**Data.** WHO Life Expectancy dataset (Kaggle): 183 countries, 2000–2015, cleaned to fix impossible values,
missing-value codes and unreliable columns.

**Model.** A random forest using ten development indicators only. Each country's estimate comes from a model
trained on the *other* countries (5-fold grouped cross-validation), so it is a genuine prediction for a country
the model has not seen. Average error across all countries: **{overall_error:.1f} years**.

**Why no mortality indicators?** Life expectancy is *calculated from* death rates. Adding adult mortality,
child deaths or HIV/AIDS deaths makes the model look more accurate (about 2 years of error) only because those
inputs contain part of the answer. This is called *target leakage*.

**Limitations.** The data ends in 2015 and has known recording errors (for example, very noisy GDP figures);
results are national averages and describe associations, not causes.

Full analysis and code: [{REPO_URL.replace('https://', '')}]({REPO_URL})
""")
