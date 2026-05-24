#  Old Car Price Prediction

This is a Machine Learning project that predicts the resale price of old cars using different car features.

---

##  Features
- Predict old car prices
- Interactive Streamlit web app
- Machine Learning based prediction
- User-friendly interface

---

##  Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn

---

##  Machine Learning Algorithm
- Random Forest Regressor

---

##  Project Structure

OldCarPricePrediction/
│
├── app.py
├── train_model.py
├── car_model.pkl
├── model_columns.pkl
├── car_encoder.pkl
├── location_encoder.pkl
├── owner_encoder.pkl
├── engine_encoder.pkl
├── README.md
└── requirements.txt

---

##  Dataset Features
- Car Name
- Location
- Year
- Kilometers Driven
- Fuel Type
- Transmission
- Owner Type
- Engine

---

## ▶ How to Run

Install required libraries:

```bash
pip install -r requirements.txt

Run Streamlit app:
streamlit run app.py
