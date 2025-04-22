import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["air_quality"]
collection = db["beijing_data"]

# Load data from MongoDB
data = pd.DataFrame(list(collection.find()))

# Drop unwanted columns
data.drop(columns=["_id", "station"], inplace=True, errors="ignore")

# Convert time column to datetime
data["time"] = pd.to_datetime(data["time"], errors="coerce")

# Drop rows with missing essential values
required_columns = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
data.dropna(subset=required_columns, inplace=True)

# Rename columns to lowercase
data.columns = [col.lower() for col in data.columns]

# ---- AQI BREAKPOINTS FOR EACH POLLUTANT (INDIA CPCB) ---- #
breakpoints = {
    "pm2.5": [
        {"bp_lo": 0, "bp_hi": 30, "aqi_lo": 0, "aqi_hi": 50},
        {"bp_lo": 31, "bp_hi": 60, "aqi_lo": 51, "aqi_hi": 100},
        {"bp_lo": 61, "bp_hi": 90, "aqi_lo": 101, "aqi_hi": 200},
        {"bp_lo": 91, "bp_hi": 120, "aqi_lo": 201, "aqi_hi": 300},
        {"bp_lo": 121, "bp_hi": 250, "aqi_lo": 301, "aqi_hi": 400},
        {"bp_lo": 251, "bp_hi": 500, "aqi_lo": 401, "aqi_hi": 500},
    ],
    "pm10": [
        {"bp_lo": 0, "bp_hi": 50, "aqi_lo": 0, "aqi_hi": 50},
        {"bp_lo": 51, "bp_hi": 100, "aqi_lo": 51, "aqi_hi": 100},
        {"bp_lo": 101, "bp_hi": 250, "aqi_lo": 101, "aqi_hi": 200},
        {"bp_lo": 251, "bp_hi": 350, "aqi_lo": 201, "aqi_hi": 300},
        {"bp_lo": 351, "bp_hi": 430, "aqi_lo": 301, "aqi_hi": 400},
        {"bp_lo": 431, "bp_hi": 500, "aqi_lo": 401, "aqi_hi": 500},
    ],
    "so2": [
        {"bp_lo": 0, "bp_hi": 40, "aqi_lo": 0, "aqi_hi": 50},
        {"bp_lo": 41, "bp_hi": 80, "aqi_lo": 51, "aqi_hi": 100},
        {"bp_lo": 81, "bp_hi": 380, "aqi_lo": 101, "aqi_hi": 200},
        {"bp_lo": 381, "bp_hi": 800, "aqi_lo": 201, "aqi_hi": 300},
        {"bp_lo": 801, "bp_hi": 1600, "aqi_lo": 301, "aqi_hi": 400},
        {"bp_lo": 1601, "bp_hi": 2000, "aqi_lo": 401, "aqi_hi": 500},
    ],
    "no2": [
        {"bp_lo": 0, "bp_hi": 40, "aqi_lo": 0, "aqi_hi": 50},
        {"bp_lo": 41, "bp_hi": 80, "aqi_lo": 51, "aqi_hi": 100},
        {"bp_lo": 81, "bp_hi": 180, "aqi_lo": 101, "aqi_hi": 200},
        {"bp_lo": 181, "bp_hi": 280, "aqi_lo": 201, "aqi_hi": 300},
        {"bp_lo": 281, "bp_hi": 400, "aqi_lo": 301, "aqi_hi": 400},
        {"bp_lo": 401, "bp_hi": 500, "aqi_lo": 401, "aqi_hi": 500},
    ],
    "co": [
        {"bp_lo": 0, "bp_hi": 1.0, "aqi_lo": 0, "aqi_hi": 50},
        {"bp_lo": 1.1, "bp_hi": 2.0, "aqi_lo": 51, "aqi_hi": 100},
        {"bp_lo": 2.1, "bp_hi": 10, "aqi_lo": 101, "aqi_hi": 200},
        {"bp_lo": 10.1, "bp_hi": 17, "aqi_lo": 201, "aqi_hi": 300},
        {"bp_lo": 17.1, "bp_hi": 34, "aqi_lo": 301, "aqi_hi": 400},
        {"bp_lo": 34.1, "bp_hi": 50, "aqi_lo": 401, "aqi_hi": 500},
    ],
    "o3": [
        {"bp_lo": 0, "bp_hi": 50, "aqi_lo": 0, "aqi_hi": 50},
        {"bp_lo": 51, "bp_hi": 100, "aqi_lo": 51, "aqi_hi": 100},
        {"bp_lo": 101, "bp_hi": 168, "aqi_lo": 101, "aqi_hi": 200},
        {"bp_lo": 169, "bp_hi": 208, "aqi_lo": 201, "aqi_hi": 300},
        {"bp_lo": 209, "bp_hi": 748, "aqi_lo": 301, "aqi_hi": 400},
        {"bp_lo": 749, "bp_hi": 1000, "aqi_lo": 401, "aqi_hi": 500},
    ]
}

# Compute AQI sub-index for each pollutant
def compute_sub_index(value, breakpoints):
    for bp in breakpoints:
        if bp["bp_lo"] <= value <= bp["bp_hi"]:
            return round(((bp["aqi_hi"] - bp["aqi_lo"]) / (bp["bp_hi"] - bp["bp_lo"])) *
                         (value - bp["bp_lo"]) + bp["aqi_lo"], 2)
    return np.nan

# Calculate AQI as max of sub-indices
def compute_overall_aqi(row):
    sub_indices = []
    for pollutant, bp_list in breakpoints.items():
        if pd.notnull(row[pollutant]):
            sub_index = compute_sub_index(row[pollutant], bp_list)
            sub_indices.append(sub_index)
    return max(sub_indices) if sub_indices else np.nan

# Add AQI column
data["aqi"] = data.apply(compute_overall_aqi, axis=1)

# Drop rows where AQI couldn't be calculated
data.dropna(subset=["aqi"], inplace=True)

# Features and target
feature_cols = ["pm2.5", "pm10", "so2", "no2", "co", "o3"]
X = data[feature_cols]
y = data["aqi"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure X_train and X_test have no NaNs before scaling
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training with GridSearchCV
estimator = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(
    estimator, param_grid, cv=3, n_jobs=-1, scoring='r2', verbose=2
)

# Fit the model
grid_search.fit(X_train_scaled, y_train)

# Print best parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate the model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# Save model and scaler
joblib.dump(best_model, "models/aqi_model.pkl")
joblib.dump(scaler, "models/aqi_scaler.pkl")

print("✅ Model and scaler saved successfully!")