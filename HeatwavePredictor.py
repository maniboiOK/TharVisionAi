import json
import os
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "rajasthan_heatwave_dataset_1000_rows.csv"

FEATURE_COLUMNS = [
    "Temperature_C",
    "Humidity_Percent",
    "AirPressure_hPa",
    "WindSpeed_kmh",
    "Radiation_Wm2",
]

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
USER_CITY = "Jodhpur"
COUNTRY_CODE = "IN"


@lru_cache(maxsize=1)
def load_dataset():
    return pd.read_csv(DATASET_PATH)


@lru_cache(maxsize=1)
def train_heatwave_model():
    data = load_dataset()
    x = data[FEATURE_COLUMNS]
    y = data["HeatwaveLikelihood_Percent"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )

    model.fit(x_train, y_train)
    return model, x, y, x_train, x_test, y_train, y_test


def evaluate_model():
    model, x, y, x_train, x_test, y_train, y_test = train_heatwave_model()

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    cv_scores = cross_val_score(
        model,
        x,
        y,
        cv=5,
        scoring="neg_mean_squared_error",
    )

    return {
        "train_mse": mean_squared_error(y_train, train_pred),
        "test_mse": mean_squared_error(y_test, test_pred),
        "r2_score": r2_score(y_test, test_pred),
        "cv_costs": -cv_scores,
    }


def estimate_radiation_from_clouds(cloud_cover_percent):
    """
    OpenWeatherMap's free current weather endpoint does not provide solar radiation.
    This estimates radiation from cloud cover so the model receives all expected inputs.
    """
    clear_sky_radiation = 850
    cloud_reduction = 0.75 * (cloud_cover_percent / 100)
    radiation = clear_sky_radiation * (1 - cloud_reduction)
    return round(max(150, min(radiation, 900)), 2)


def get_live_weather(city, api_key=None, country_code="IN"):
    api_key = api_key or OPENWEATHER_API_KEY
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
        "CloudCover_Percent": cloud_cover,
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
    model = train_heatwave_model()[0]
    live_input = pd.DataFrame([{column: weather_values[column] for column in FEATURE_COLUMNS}])

    prediction = float(model.predict(live_input)[0])
    prediction = max(0, min(100, prediction))

    return round(prediction, 2), get_risk_level(prediction)


def get_heatwave_report(city, api_key=None):
    weather_values = get_live_weather(city, api_key, COUNTRY_CODE)
    heatwave_percent, risk_level = predict_heatwave_from_weather(weather_values)

    return {
        "city": weather_values["City"],
        "weather": weather_values,
        "heatwave_percent": heatwave_percent,
        "risk_level": risk_level,
    }


def run_live_heatwave_prediction():
    city = input("\nEnter Rajasthan city for heatwave prediction: ").strip()
    if not city:
        city = USER_CITY

    report = get_heatwave_report(city)
    weather_values = report["weather"]

    print("\nLive Weather Prediction")
    print("City:", weather_values["City"])
    print("Temperature (C):", weather_values["Temperature_C"])
    print("Humidity (%):", weather_values["Humidity_Percent"])
    print("Air Pressure (hPa):", weather_values["AirPressure_hPa"])
    print("Wind Speed (km/h):", round(weather_values["WindSpeed_kmh"], 2))
    print("Estimated Radiation (W/m2):", weather_values["Radiation_Wm2"])
    print("Cloud Cover (%):", weather_values["CloudCover_Percent"])
    print("Heatwave Likelihood (%):", report["heatwave_percent"])
    print("Risk Level:", report["risk_level"])


if __name__ == "__main__":
    metrics = evaluate_model()
    print("Train Cost (MSE):", metrics["train_mse"])
    print("Test Cost (MSE):", metrics["test_mse"])
    print("R2 Score:", metrics["r2_score"])
    print("\nCross Validation Costs:", metrics["cv_costs"])
    print("Final Train Cost:", metrics["train_mse"])
    run_live_heatwave_prediction()
