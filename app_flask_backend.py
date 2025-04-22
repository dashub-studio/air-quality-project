from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained AQI model
model = joblib.load("models/aqi_model.pkl")

def get_health_advisory(aqi):
    """Classify AQI and provide health advisory."""
    if aqi <= 50:
        return "Good 😊 - Air quality is considered satisfactory for everyone."
    elif aqi <= 100:
        return "Satisfactory 🙂 - Air quality is acceptable, but sensitive groups should take precautions."
    elif aqi <= 200:
        return "Moderate 😐 - Unhealthy for sensitive groups like children and senior citizens."
    elif aqi <= 300:
        return "Poor 😷 - Everyone may experience health effects; visitors, children, and seniors should avoid outdoor activities."
    elif aqi <= 400:
        return "Very Poor 😫 - Health warnings of emergency conditions; avoid outdoor activities."
    else:
        return "Severe ☠️ - Serious health effects; everyone should stay indoors."

@app.route("/")
def index():
    # List of stations to populate the dropdown
    stations = [
        "Aotizhongxin", "Changping", "Dingling", "Dongsi",
        "Guanyuan", "Gucheng", "Huairou", "Nongzhanguan",
        "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"
    ]
    return render_template("index.html", stations=stations)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.form
        station = data.get("station")  # Selected station
        input_data = {
            "pm2.5": float(data.get("pm2_5")),
            "pm10": float(data.get("pm10")),
            "no2": float(data.get("no2")),
            "so2": float(data.get("so2")),
            "co": float(data.get("co")),
            "o3": float(data.get("o3"))
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict AQI
        predicted_aqi = model.predict(input_df)[0]

        # Get health advisory
        health_advisory = get_health_advisory(predicted_aqi)

        # Return result
        return jsonify({
            "station": station,
            "aqi": round(predicted_aqi, 2),
            "health_advisory": health_advisory
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)