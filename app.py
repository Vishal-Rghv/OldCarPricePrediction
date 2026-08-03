import streamlit as st
import pandas as pd
import pickle
import os


BASE_DIR = os.path.dirname(__file__)


# Load trained pipeline model
with open(os.path.join(BASE_DIR, "car_model.pkl"), "rb") as f:
    model = pickle.load(f)


# Load dataset for dropdown options
data = pd.read_csv(os.path.join(BASE_DIR, "OldCarPriceData.csv"))


st.title("🚗 Old Car Price Prediction")


# Dropdown inputs

car_name = st.selectbox(
    "Car Name",
    sorted(data["Car_name"].unique())
)


location = st.selectbox(
    "Location",
    sorted(data["Location"].unique())
)


year = st.number_input(
    "Year",
    min_value=int(data["Year"].min()),
    max_value=int(data["Year"].max())
)


kms = st.number_input(
    "Kilometers Driven",
    min_value=0,
    max_value=500000
)


owner = st.selectbox(
    "Owner Type",
    sorted(data["Owner_Type"].unique())
)


engine = st.selectbox(
    "Engine",
    sorted(data["Engine"].dropna().unique())
)


fuel = st.selectbox(
    "Fuel Type",
    sorted(data["Fuel_Type"].unique())
)


transmission = st.selectbox(
    "Transmission",
    sorted(data["Transmission"].unique())
)



if st.button("Predict Price"):


    input_data = pd.DataFrame({

        "Car_name": [car_name],
        "Location": [location],
        "Year": [year],
        "Kilometers_Driven": [kms],
        "Owner_Type": [owner],
        "Engine": [engine],
        "Fuel_Type": [fuel],
        "Transmission": [transmission]

    })


    prediction = model.predict(input_data)[0]


    st.success(
        f"🚗 Estimated Price: ₹ {prediction:.2f} Lakh"
    )