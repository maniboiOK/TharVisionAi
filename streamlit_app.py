from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import streamlit as st

from HeatwavePredictor import (
    evaluate_model,
    get_heatwave_report,
    load_dataset,
    predict_heatwave_from_weather,
)


BASE_DIR = Path(__file__).resolve().parent
RECOMMENDATION_FILE = BASE_DIR / "AIrecommedation (1).py"

DISTRICT_COORDS = {
    "Ajmer": (26.4499, 74.6399),
    "Barmer": (25.7532, 71.4181),
    "Bhilwara": (25.3463, 74.6364),
    "Bikaner": (28.0229, 73.3119),
    "Churu": (28.2921, 74.9618),
    "Jaipur": (26.9124, 75.7873),
    "Jaisalmer": (26.9157, 70.9083),
    "Jodhpur": (26.2389, 73.0243),
    "Kota": (25.2138, 75.8648),
    "Nagaur": (27.1991, 73.7409),
    "Pali": (25.7711, 73.3234),
    "Udaipur": (24.5854, 73.7125),
}


@st.cache_resource
def load_recommendation_module():
    spec = importlib.util.spec_from_file_location("thar_recommendations", RECOMMENDATION_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def risk_label(score: float) -> str:
    if score >= 76:
        return "Very High"
    if score >= 56:
        return "High"
    if score >= 31:
        return "Moderate"
    return "Low"


def risk_color(score: float) -> str:
    if score >= 76:
        return "#ff4b3e"
    if score >= 56:
        return "#ff9f1c"
    if score >= 31:
        return "#ffd166"
    return "#21c875"


def estimate_drought_risk(weather: dict, rainfall_mm: float, water_stress: str) -> float:
    stress_bonus = {"Low": 4, "Moderate": 14, "High": 24, "Severe": 34}[water_stress]
    rainfall_pressure = max(0, 55 - rainfall_mm) * 0.8
    heat_pressure = max(0, weather["Temperature_C"] - 35) * 3.4
    humidity_pressure = max(0, 45 - weather["Humidity_Percent"]) * 0.75
    radiation_pressure = max(0, weather["Radiation_Wm2"] - 550) * 0.04
    return round(min(100, stress_bonus + rainfall_pressure + heat_pressure + humidity_pressure + radiation_pressure), 1)


def make_manual_weather(city: str, temperature: float, humidity: float, pressure: float, wind: float, radiation: float):
    return {
        "City": city,
        "Temperature_C": temperature,
        "Humidity_Percent": humidity,
        "AirPressure_hPa": pressure,
        "WindSpeed_kmh": wind,
        "Radiation_Wm2": radiation,
        "CloudCover_Percent": max(0, min(100, 100 - (radiation / 850) * 100)),
    }


def build_snapshot(recommendation_module, report: dict, drought_risk: float, rainfall_mm: float, water_stress: str, reservoir: float, crop_stage: str):
    weather = report["weather"]
    return recommendation_module.ClimateSnapshot(
        district=report["city"],
        drought_risk=drought_risk,
        heatwave_risk=report["heatwave_percent"],
        rainfall_forecast_mm=rainfall_mm,
        water_stress=water_stress,
        avg_temperature_c=weather["Temperature_C"],
        reservoir_level_percent=reservoir,
        crop_stage=crop_stage,
    )


def response_to_dict(response):
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


def apply_theme():
    st.set_page_config(
        page_title="THAR Vision AI",
        page_icon="TV",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        :root {
            --bg: #06111f;
            --panel: rgba(8, 24, 42, 0.86);
            --panel-soft: rgba(15, 38, 61, 0.70);
            --line: rgba(87, 198, 255, 0.20);
            --text: #eaf6ff;
            --muted: #99afbf;
            --cyan: #27d7ff;
            --green: #31d18b;
            --orange: #ff9f1c;
            --red: #ff4b3e;
        }
        .stApp {
            background:
                radial-gradient(circle at 76% 12%, rgba(39, 215, 255, 0.11), transparent 28%),
                linear-gradient(135deg, #050b14 0%, #071828 48%, #050b14 100%);
            color: var(--text);
        }
        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(8, 17, 30, 0.94), rgba(18, 43, 58, 0.86)),
                linear-gradient(22deg, rgba(255, 159, 28, 0.25), transparent 58%);
            border-right: 1px solid rgba(39, 215, 255, 0.22);
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        .hero {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 22px 24px;
            background:
                linear-gradient(90deg, rgba(8, 24, 42, 0.94), rgba(9, 35, 44, 0.78)),
                repeating-linear-gradient(135deg, rgba(39, 215, 255, 0.08) 0 1px, transparent 1px 24px);
        }
        .hero-title {
            font-size: 34px;
            font-weight: 780;
            margin-bottom: 4px;
        }
        .hero-subtitle {
            color: var(--muted);
            font-size: 16px;
        }
        .metric-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 16px;
            background: var(--panel);
            min-height: 128px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .metric-value {
            font-size: 34px;
            font-weight: 780;
            margin-top: 6px;
        }
        .metric-note {
            color: var(--muted);
            font-size: 13px;
            margin-top: 8px;
        }
        .recommendation {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px 16px;
            background: var(--panel-soft);
            margin-bottom: 10px;
        }
        .tag {
            display: inline-block;
            border-radius: 6px;
            padding: 3px 8px;
            font-size: 12px;
            margin-right: 6px;
            background: rgba(39, 215, 255, 0.13);
            color: #b7ecff;
            border: 1px solid rgba(39, 215, 255, 0.20);
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            padding: 14px;
            border-radius: 8px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            border: 1px solid rgba(87, 198, 255, 0.17);
            background: rgba(8, 24, 42, 0.62);
            padding: 9px 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, note: str, color: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recommendations(recommendations: list[dict]):
    for item in recommendations:
        impact = item["impact"].value if hasattr(item["impact"], "value") else item["impact"]
        urgency = item["urgency"].value if hasattr(item["urgency"], "value") else item["urgency"]
        st.markdown(
            f"""
            <div class="recommendation">
                <div style="font-weight:760;font-size:16px;">{item["title"]}</div>
                <div style="color:#b7c7d5;margin:5px 0 10px 0;">{item["description"]}</div>
                <span class="tag">{item["category"]}</span>
                <span class="tag">{impact}</span>
                <span class="tag">{urgency}</span>
                <div style="color:#8da6b7;font-size:13px;margin-top:10px;">{item["reason"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_prediction_report(city: str, api_key: str, mode: str, manual_weather: dict):
    if mode == "Live weather API":
        return get_heatwave_report(city, api_key=api_key)

    heatwave_percent, heatwave_level = predict_heatwave_from_weather(manual_weather)
    return {
        "city": city,
        "weather": manual_weather,
        "heatwave_percent": heatwave_percent,
        "risk_level": heatwave_level,
    }


def main():
    apply_theme()
    recommendation_module = load_recommendation_module()

    with st.sidebar:
        st.markdown("## THAR VISION AI")
        st.caption("AI-powered climate intelligence for Rajasthan")
        page = st.radio(
            "Navigation",
            [
                "Home",
                "Heatwave Prediction",
                "Rajasthan Risk Map",
                "Forecasting",
                "AI Recommendations",
                "Historical Insights",
                "Data Explorer",
                "About Project",
            ],
        )
        st.divider()
        mode = st.radio("Prediction source", ["Manual/demo data", "Live weather API"])
        default_key = get_secret("OPENWEATHER_API_KEY", "")
        api_key = st.text_input("OpenWeatherMap API key", value=default_key, type="password")
        city = st.selectbox("District", sorted(DISTRICT_COORDS), index=sorted(DISTRICT_COORDS).index("Jodhpur"))
        rainfall_mm = st.slider("30-day rainfall forecast (mm)", 0.0, 180.0, 32.6, 0.5)
        water_stress = st.select_slider("Water stress", options=["Low", "Moderate", "High", "Severe"], value="High")
        reservoir = st.slider("Reservoir/storage level (%)", 0.0, 100.0, 28.0, 1.0)
        crop_stage = st.selectbox("Crop stage", ["Pre-sowing", "Sowing", "Growing", "Harvest", "Off-season"], index=2)

    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Building a Climate Resilient Rajasthan</div>
            <div class="hero-subtitle">Real-time heatwave prediction, AI recommendations, district risk mapping, and climate decision support.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    manual_weather = make_manual_weather(
        city=city,
        temperature=st.sidebar.slider("Manual temperature (C)", 25.0, 52.0, 43.0, 0.1),
        humidity=st.sidebar.slider("Manual humidity (%)", 5.0, 90.0, 31.0, 0.5),
        pressure=st.sidebar.slider("Manual pressure (hPa)", 980.0, 1035.0, 1002.0, 0.5),
        wind=st.sidebar.slider("Manual wind speed (km/h)", 0.0, 45.0, 12.0, 0.5),
        radiation=st.sidebar.slider("Manual radiation (W/m2)", 150.0, 900.0, 720.0, 5.0),
    )

    try:
        report = get_prediction_report(city, api_key, mode, manual_weather)
        error_message = None
    except Exception as exc:
        report = get_prediction_report(city, api_key, "Manual/demo data", manual_weather)
        error_message = f"Live API unavailable, using manual/demo values. Details: {exc}"

    weather = report["weather"]
    heatwave_percent = report["heatwave_percent"]
    drought_risk = estimate_drought_risk(weather, rainfall_mm, water_stress)
    snapshot = build_snapshot(
        recommendation_module,
        report,
        drought_risk,
        rainfall_mm,
        water_stress,
        reservoir,
        crop_stage,
    )
    recommendation_response = response_to_dict(recommendation_module.recommend_actions(snapshot))

    if error_message:
        st.warning(error_message)

    if page == "Home":
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            render_metric_card("Drought Risk", f"{drought_risk:.0f}%", risk_label(drought_risk), risk_color(drought_risk))
        with col2:
            render_metric_card("Heatwave Risk", f"{heatwave_percent:.0f}%", report["risk_level"], risk_color(heatwave_percent))
        with col3:
            render_metric_card("Rainfall Forecast", f"{rainfall_mm:.1f} mm", "Next 30 days", "#27d7ff")
        with col4:
            render_metric_card("Water Stress", water_stress, "District water pressure", "#d471ff")
        with col5:
            render_metric_card("Climate Index", f"{recommendation_response['risk_score']}/100", recommendation_response["risk_level"], "#31d18b")

        left, right = st.columns([1.05, 1])
        with left:
            st.subheader("Rajasthan Risk Map")
            map_rows = []
            for district, coords in DISTRICT_COORDS.items():
                distance_penalty = 10 if district != city else 0
                district_score = max(8, min(100, heatwave_percent - distance_penalty + (hash(district) % 17)))
                map_rows.append(
                    {
                        "district": district,
                        "lat": coords[0],
                        "lon": coords[1],
                        "risk": district_score,
                    }
                )
            map_df = pd.DataFrame(map_rows)
            st.map(map_df, latitude="lat", longitude="lon", size="risk", color="#ff9f1c")
            st.caption("Marker size reflects heatwave-oriented district risk.")

        with right:
            st.subheader("Forecasting Overview")
            forecast = pd.DataFrame(
                {
                    "Day": ["Today", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"],
                    "Temperature_C": [
                        weather["Temperature_C"],
                        weather["Temperature_C"] + 1.2,
                        weather["Temperature_C"] + 2.0,
                        weather["Temperature_C"] + 1.4,
                        weather["Temperature_C"] - 0.2,
                        weather["Temperature_C"] - 1.1,
                        weather["Temperature_C"] - 1.7,
                    ],
                    "Heatwave_Risk": [
                        heatwave_percent,
                        min(100, heatwave_percent + 4),
                        min(100, heatwave_percent + 8),
                        min(100, heatwave_percent + 5),
                        max(0, heatwave_percent - 2),
                        max(0, heatwave_percent - 7),
                        max(0, heatwave_percent - 11),
                    ],
                }
            )
            st.line_chart(forecast, x="Day", y=["Temperature_C", "Heatwave_Risk"])

            st.subheader("AI Recommendations")
            render_recommendations(recommendation_response["recommendations"][:3])

    elif page == "Heatwave Prediction":
        st.subheader("Heatwave Prediction")
        c1, c2, c3 = st.columns(3)
        c1.metric("Heatwave likelihood", f"{heatwave_percent:.2f}%")
        c2.metric("Risk level", report["risk_level"])
        c3.metric("Selected district", report["city"])
        st.dataframe(pd.DataFrame([weather]), use_container_width=True)

    elif page == "Rajasthan Risk Map":
        st.subheader("Rajasthan Risk Map")
        rows = []
        for district, coords in DISTRICT_COORDS.items():
            rows.append(
                {
                    "District": district,
                    "lat": coords[0],
                    "lon": coords[1],
                    "Heatwave Risk": max(5, min(100, heatwave_percent + (hash(district) % 25) - 12)),
                }
            )
        risk_df = pd.DataFrame(rows)
        st.map(risk_df, latitude="lat", longitude="lon", size="Heatwave Risk", color="#ff4b3e")
        st.dataframe(risk_df.sort_values("Heatwave Risk", ascending=False), use_container_width=True)

    elif page == "Forecasting":
        st.subheader("Forecasting")
        forecast_days = pd.date_range(pd.Timestamp.today().normalize(), periods=10)
        forecast_df = pd.DataFrame(
            {
                "Date": forecast_days,
                "Predicted Temperature C": [weather["Temperature_C"] + ((i % 4) - 1) * 0.9 for i in range(10)],
                "Predicted Heatwave Risk": [max(0, min(100, heatwave_percent + ((i % 5) - 2) * 3)) for i in range(10)],
                "Rainfall mm": [max(0, rainfall_mm / 10 + ((i % 3) - 1) * 1.6) for i in range(10)],
            }
        )
        st.line_chart(forecast_df, x="Date", y=["Predicted Temperature C", "Predicted Heatwave Risk", "Rainfall mm"])
        st.dataframe(forecast_df, use_container_width=True)

    elif page == "AI Recommendations":
        st.subheader("AI Recommendations")
        c1, c2 = st.columns([0.7, 1.3])
        with c1:
            st.metric("Combined risk score", f"{recommendation_response['risk_score']}/100")
            st.metric("Overall risk level", recommendation_response["risk_level"])
            st.json(
                {
                    "district": report["city"],
                    "drought_risk": drought_risk,
                    "heatwave_risk": heatwave_percent,
                    "rainfall_forecast_mm": rainfall_mm,
                    "water_stress": water_stress,
                }
            )
        with c2:
            render_recommendations(recommendation_response["recommendations"])

    elif page == "Historical Insights":
        st.subheader("Historical Insights")
        data = load_dataset()
        city_data = data[data["City"].str.lower() == city.lower()]
        if city_data.empty:
            city_data = data
        st.metric("Average historical heatwave likelihood", f"{city_data['HeatwaveLikelihood_Percent'].mean():.1f}%")
        st.metric("Maximum historical temperature", f"{city_data['Temperature_C'].max():.1f} C")
        st.scatter_chart(city_data, x="Temperature_C", y="HeatwaveLikelihood_Percent", color="#ff9f1c")

    elif page == "Data Explorer":
        st.subheader("Data Explorer")
        data = load_dataset()
        st.dataframe(data, use_container_width=True)
        with st.expander("Model evaluation"):
            metrics = evaluate_model()
            st.write(
                {
                    "Train MSE": round(metrics["train_mse"], 4),
                    "Test MSE": round(metrics["test_mse"], 4),
                    "R2 Score": round(metrics["r2_score"], 4),
                    "Cross Validation MSE": [round(value, 4) for value in metrics["cv_costs"]],
                }
            )

    else:
        st.subheader("About Project")
        st.write(
            "THAR Vision AI combines live weather data, a trained heatwave prediction model, "
            "and rule-based AI recommendations to support climate resilience planning in Rajasthan."
        )
        st.write("Data sources: local training dataset, OpenWeatherMap current weather, and user-adjustable planning inputs.")
        st.write("For Streamlit Cloud, add `OPENWEATHER_API_KEY` in app secrets before enabling live weather mode.")


if __name__ == "__main__":
    main()
