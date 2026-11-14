"""
Insurance Charge Predictor - a simple web app.

How to run (in the VS Code terminal, inside your project folder):
    pip install streamlit
    streamlit run app.py

Keep Health_Insurance.csv in the same folder as this file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Insurance Charge Predictor", page_icon="🏥")

# ---------------------------------------------------------------
# 1) Train the model (runs once, then Streamlit remembers it)
# ---------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "Health_Insurance.csv"


@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_FILE)

    # Same preparation as in the notebook:
    # text columns -> 0/1 numbers, and log of charges to reduce skewness
    X = pd.get_dummies(df.drop(columns="charges"), drop_first=True, dtype=int)
    y = np.log1p(df["charges"])

    # Random Forest with the best settings found by GridSearchCV
    model = RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_split=2, random_state=42
    )
    model.fit(X, y)
    return model, X.columns


model, training_columns = train_model()

# ---------------------------------------------------------------
# 2) The page: title and input widgets
# ---------------------------------------------------------------
st.title("🏥 Health Insurance Charge Predictor")
st.write("Fill in the details below and click **Predict** to estimate the yearly insurance charge.")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=64, value=35)
    bmi = st.number_input("BMI", min_value=15.0, max_value=55.0, value=28.0, step=0.1)
    children = st.number_input("Number of children", min_value=0, max_value=5, value=0, step=1)

with col2:
    sex = st.radio("Sex", ["female", "male"], horizontal=True)
    smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# ---------------------------------------------------------------
# 3) When the button is clicked: build the input row and predict
# ---------------------------------------------------------------
if st.button("Predict", type="primary"):
    # Turn the answers into the same 0/1 columns the model was trained on
    row = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": int(sex == "male"),
        "smoker_yes": int(smoker == "yes"),
        "region_northwest": int(region == "northwest"),
        "region_southeast": int(region == "southeast"),
        "region_southwest": int(region == "southwest"),
    }])
    row = row.reindex(columns=training_columns, fill_value=0)  # same column order as training

    # The model predicts log(charges), so convert back to dollars
    prediction = np.expm1(model.predict(row)[0])

    st.success("Prediction ready!")
    st.metric("Estimated yearly insurance charge", f"${prediction:,.0f}")
    st.caption(
        "This is an estimate from a machine learning model, not a real insurance quote. "
        "On unseen test data the model was typically off by about $2,100."
    )
