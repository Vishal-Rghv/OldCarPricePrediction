import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


dataset = pd.read_csv("CarPrediction.csv")
dataset["Engine"] = dataset["Engine"].fillna(dataset["Engine"].mode()[0])

Car_name_le = LabelEncoder()
Location_le = LabelEncoder()
Owner_Type_le = LabelEncoder()
Engine_le = LabelEncoder()

dataset["Car_name"] = Car_name_le.fit_transform(dataset["Car_name"])
dataset["Location"] = Location_le.fit_transform(dataset["Location"])
dataset["Owner_Type"] = Owner_Type_le.fit_transform(dataset["Owner_Type"])
dataset["Engine"] = Engine_le.fit_transform(dataset["Engine"])

dataset = pd.get_dummies(dataset, columns=["Fuel_Type", "Transmission"])

X = dataset.drop(columns=["Price"])
Y = dataset["Price"]

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(random_state=42)
model.fit(x_train, y_train)

print("Test Score:", model.score(x_test, y_test) * 100)
print("Train Score:", model.score(x_train, y_train) * 100)


prediction = model.predict(x_test)
print("MSE:", mean_squared_error(y_test, prediction))
print("MAE:", mean_absolute_error(y_test, prediction))

pickle.dump(model, open(r"D:\CarpredictionApp\car_model.pkl", "wb"))
pickle.dump(X.columns, open(r"D:\CarpredictionApp\model_columns.pkl", "wb"))

pickle.dump(Car_name_le, open(r"D:\CarpredictionApp\car_encoder.pkl", "wb"))
pickle.dump(Location_le, open(r"D:\CarpredictionApp\location_encoder.pkl", "wb"))
pickle.dump(Owner_Type_le, open(r"D:\CarpredictionApp\owner_encoder.pkl", "wb"))
pickle.dump(Engine_le, open(r"D:\CarpredictionApp\engine_encoder.pkl", "wb"))

print("All files saved successfully")