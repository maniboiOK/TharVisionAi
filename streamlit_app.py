import numpy as np
import pandas as pd
import json
import urllib.parse
import urllib.request
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

data = pd.read_csv("Thar vision AI/rajasthan_heatwave_dataset_1000_rows.csv")


X = data[
    [
        "Temperature_C",
        "Humidity_Percent",
        "AirPressure_hPa",
        "WindSpeed_kmh",
        "Radiation_Wm2"
    ]
]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = data["HeatwaveLikelihood_Percent"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    reg_alpha=0.1,        # L1
    reg_lambda=1.0,       # L2

    
)


# Train model
model.fit(X_train, y_train)



train_pred = model.predict(X_train)

train_cost = mean_squared_error(y_train, train_pred)

print("Train Cost (MSE):", train_cost)


test_pred = model.predict(X_test)

test_cost = mean_squared_error(y_test, test_pred)

print("Test Cost (MSE):", test_cost)



r2 = r2_score(y_test, test_pred)

print("R2 Score:", r2)



cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring='neg_mean_squared_error'
)

cv_costs = -cv_scores

print("\nCross Validation Costs:", cv_costs)

print("Final Train Cost:", train_cost)


# ---------------- LIVE WEATHER HEATWAVE PREDICTION ----------------
# Put your OpenWeatherMap API key here:
# https://openweathermap.org/api
OPENWEATHER_API_KEY = "3991d46f1f35cceb3b9e38d44fd039dd"

# Change this city whenever you want to test another location.
USER_CITY = "Jodhpur"
COUNTRY_CODE = "IN"


def estimate_radiation_from_clouds(cloud_cover_percent):
    """
    OpenWeatherMap's free current weather endpoint does not provide solar radiation.
    This gives a practical estimate using cloud cover so the trained model still
    receives all 5 inputs it expects.
    """
    clear_sky_radiation = 850
    cloud_reduction = 0.75 * (cloud_cover_percent / 100)
    radiation = clear_sky_radiation * (1 - cloud_reduction)
    return round(max(150, min(radiation, 900)), 2)


def get_live_weather(city, api_key, country_code="IN"):
    if not api_key or api_key == "PUT_YOUR_API_KEY_HERE":
        raise ValueError("Please add your OpenWeatherMap API key first.")

    query = urllib.parse.quote(f"{city},{country_code}")
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?q={query}&appid={api_key}&units=metric"
    )

    with urllib.request.urlopen(url, timeout=15) as response:
        weather_data = json.loads(response.read().decode("utf-8"))

    cloud_cover = weather_data.get("clouds", {}).get("all", 20)

    return {
        "Temperature_C": weather_data["main"]["temp"],
        "Humidity_Percent": weather_data["main"]["humidity"],
        "AirPressure_hPa": weather_data["main"]["pressure"],
        "WindSpeed_kmh": weather_data["wind"]["speed"] * 3.6,
        "Radiation_Wm2": estimate_radiation_from_clouds(cloud_cover),
        "City": weather_data.get("name", city),
        "CloudCover_Percent": cloud_cover
    }


def get_risk_level(heatwave_likelihood):
    if heatwave_likelihood >= 76:
        return "Very High Risk"
    if heatwave_likelihood >= 56:
        return "High Risk"
    if heatwave_likelihood >= 31:
        return "Moderate Risk"
    return "Low Risk"


def predict_heatwave_from_weather(weather_values):
    live_input = pd.DataFrame(
        [
            {
                "Temperature_C": weather_values["Temperature_C"],
                "Humidity_Percent": weather_values["Humidity_Percent"],
                "AirPressure_hPa": weather_values["AirPressure_hPa"],
                "WindSpeed_kmh": weather_values["WindSpeed_kmh"],
                "Radiation_Wm2": weather_values["Radiation_Wm2"]
            }
        ]
    )

    prediction = model.predict(live_input)[0]
    prediction = max(0, min(100, prediction))

    return round(prediction, 2), get_risk_level(prediction)


def run_live_heatwave_prediction():
    weather_values = get_live_weather(USER_CITY, OPENWEATHER_API_KEY, COUNTRY_CODE)
    heatwave_percent, risk_level = predict_heatwave_from_weather(weather_values)

    print("\nLive Weather Prediction")
    print("City:", weather_values["City"])
    print("Temperature (C):", weather_values["Temperature_C"])
    print("Humidity (%):", weather_values["Humidity_Percent"])
    print("Air Pressure (hPa):", weather_values["AirPressure_hPa"])
    print("Wind Speed (km/h):", round(weather_values["WindSpeed_kmh"], 2))
    print("Estimated Radiation (W/m2):", weather_values["Radiation_Wm2"])
    print("Cloud Cover (%):", weather_values["CloudCover_Percent"])
    print("Heatwave Likelihood (%):", heatwave_percent)
    print("Risk Level:", risk_level)



