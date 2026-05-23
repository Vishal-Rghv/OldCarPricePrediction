import streamlit as st
import pandas as pd
import pickle
import os

# correct path handling
BASE_DIR = os.path.dirname(__file__)

# load model + files
model = pickle.load(open(os.path.join(BASE_DIR, "car_model.pkl"), "rb"))
columns = pickle.load(open(os.path.join(BASE_DIR, "model_columns.pkl"), "rb"))

le_car = pickle.load(open(os.path.join(BASE_DIR, "car_encoder.pkl"), "rb"))
le_location = pickle.load(open(os.path.join(BASE_DIR, "location_encoder.pkl"), "rb"))
le_owner = pickle.load(open(os.path.join(BASE_DIR, "owner_encoder.pkl"), "rb"))
le_engine = pickle.load(open(os.path.join(BASE_DIR, "engine_encoder.pkl"), "rb"))


st.title("Old Car Price Prediction App")
# st.write("Predict car price using Machine Learning")

# inputs
car_name = st.selectbox("Car Name", le_car.classes_)
location = st.selectbox("Location", le_location.classes_)
year = st.number_input("Year", 2000, 2025)
kms = st.number_input("Kilometers Driven", 0, 500000)
owner = st.selectbox("Owner Type", le_owner.classes_)
engine = st.selectbox("Engine", le_engine.classes_)

fuel = st.selectbox("Fuel Type", ["Diesel", "Petrol", "CNG"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

# prediction
if st.button("Predict Price"):

    data = {
        "Car_name": [le_car.transform([car_name])[0]],
        "Location": [le_location.transform([location])[0]],
        "Year": [year],
        "Kilometers_Driven": [kms],
        "Owner_Type": [le_owner.transform([owner])[0]],
        "Engine": [le_engine.transform([engine])[0]]
    }

    df = pd.DataFrame(data)

    # one-hot encoding
    for col in ["Fuel_Type_CNG", "Fuel_Type_Diesel", "Fuel_Type_Petrol"]:
        df[col] = 0
    df[f"Fuel_Type_{fuel}"] = 1

    for col in ["Transmission_Automatic", "Transmission_Manual"]:
        df[col] = 0
    df[f"Transmission_{transmission}"] = 1

    # match training columns
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    st.success(f"🚗 Predicted Price: ₹ {prediction:.2f} Lakh")