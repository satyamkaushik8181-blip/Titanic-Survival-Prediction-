import streamlit as st
import pandas as pd
import pickle

# load model

with open("titanic_model_and_preprocessors.pkl", "rb") as f:
    model_components = pickle.load(f)

model = model_components["model"]
scaler = model_components["scaler"]

# page configuration

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("🚢 Titanic Survival Prediction")
st.markdown("Predict whether a passenger would survive the Titanic disaster.")

st.divider()

# Slidebar

st.sidebar.header("Passenger Details")

# inputs

pclass = st.sidebar.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.sidebar.selectbox(
    "Gender",
    ["male", "female"]
)

age = st.sidebar.slider(
    "Age",
    min_value=0,
    max_value=80,
    value=25
)

sibsp = st.sidebar.number_input(
    "Siblings / Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.sidebar.number_input(
    "Parents / Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.sidebar.number_input(
    "Fare",
    min_value=0.0,
    value=500.0,
    step=10.0
)

embarked = st.sidebar.selectbox(
    "Port of Embarkation",
    (
        "Cherbourg (C)",
        "Queenstown (Q)",
        "Southampton (S)"
    )
)

# feature engineering

sex = 0 if sex == "male" else 1

family_size = sibsp + parch + 1

alone = 1 if family_size == 1 else 0

embarked_C = 1 if embarked == "Cherbourg (C)" else 0
embarked_Q = 1 if embarked == "Queenstown (Q)" else 0
embarked_S = 1 if embarked == "Southampton (S)" else 0

# DataFrame

input_df = pd.DataFrame({
    "pclass": [pclass],
    "sex": [sex],
    "age": [age],
    "fare": [fare],
    "alone": [alone],
    "family_size": [family_size],
    "embarked_C": [embarked_C],
    "embarked_Q": [embarked_Q],
    "embarked_S": [embarked_S]
})

# Match Training Features

try:
    input_df = input_df[scaler.feature_names_in_]
except AttributeError:
    pass

# show input data

st.subheader("Passenger Information")

st.dataframe(input_df, use_container_width=True)

# Prediction

if st.button("Predict Survival", use_container_width=True):

    scaled_data = scaler.transform(input_df)

    prediction = model.predict(scaled_data)[0]

    probability = model.predict_proba(scaled_data)[0]

    survive_prob = probability[1] * 100
    die_prob = probability[0] * 100

    st.divider()

    if prediction == 1:

        st.success("🎉 Passenger is likely to SURVIVE!")

        st.balloons()

    else:

        st.error("❌ Passenger is NOT likely to survive.")

    st.subheader("Prediction Probability")

    st.write(f"✅ Survival Probability : **{survive_prob:.2f}%**")

    st.write(f"❌ Death Probability : **{die_prob:.2f}%**")

    st.progress(int(survive_prob))