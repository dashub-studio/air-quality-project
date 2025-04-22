import joblib
import pandas as pd
from datetime import datetime

# Load the trained AQI model
model = joblib.load("models/aqi_model.pkl")

# Sample input (replace with actual values or user input from UI)
sample_input = {
    "pm2.5": 23,
    "pm10": 63,
    "no2": 5,
    "so2": 2,
    "co": 664,
    "o3": 12
}

# Convert to DataFrame
input_df = pd.DataFrame([sample_input])

# Predict AQI
predicted_aqi = model.predict(input_df)[0]

# Health category logic
def get_health_category(aqi):
    if aqi <= 50:
        return "Good 😊 - Air quality is considered satisfactory."
    elif aqi <= 100:
        return "Satisfactory 🙂 - Acceptable air quality."
    elif aqi <= 200:
        return "Moderate 😐 - Unhealthy for sensitive groups."
    elif aqi <= 300:
        return "Poor 😷 - Everyone may experience health effects."
    elif aqi <= 400:
        return "Very Poor 😫 - Health warnings of emergency conditions."
    else:
        return "Severe ☠️ - Serious health effects, avoid outdoor activities."

# Display result
print(f"\n📈 Predicted AQI: {predicted_aqi:.2f}")
print(f"🏥 Health Advisory: {get_health_category(predicted_aqi)}")
