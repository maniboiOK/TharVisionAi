from __future__ import annotations

import importlib.util
import sys
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
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for model_name in ["ClimateSnapshot", "Recommendation", "RecommendationResponse"]:
        model = getattr(module, model_name, None)
        if model and hasattr(model, "model_rebuild"):
            model.model_rebuild(_types_namespace=module.__dict__)

    return module


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
                linear-gradient(90deg, rgba(255, 134, 28, 0.10), transparent 18%),
                radial-gradient(circle at 76% 12%, rgba(39, 215, 255, 0.11), transparent 28%),
                linear-gradient(135deg, #050b14 0%, #071828 48%, #050b14 100%);
            color: var(--text);
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            padding-top: 22px;
            padding-bottom: 24px;
            max-width: 1580px;
        }
        section[data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(4, 12, 23, 0.97), rgba(17, 31, 42, 0.92)),
                linear-gradient(24deg, rgba(255, 143, 27, 0.50), transparent 66%);
            border-right: 1px solid rgba(39, 215, 255, 0.22);
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 18px;
        }
        .brand {
            text-align: center;
            padding: 18px 6px 22px 6px;
            border-bottom: 1px solid rgba(39, 215, 255, 0.18);
            margin-bottom: 12px;
        }
        .brand-mark {
            color: #ffad32;
            font-size: 42px;
            line-height: 40px;
            font-weight: 900;
            letter-spacing: 2px;
        }
        .brand-sub {
            color: #5de8ff;
            font-size: 21px;
            letter-spacing: 4px;
            font-weight: 700;
            margin-top: 2px;
        }
        .brand-caption {
            color: #c5d4de;
            font-size: 11px;
            text-transform: uppercase;
            line-height: 1.5;
            margin-top: 15px;
        }
        .side-status {
            margin-top: 20px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 8px;
            padding: 16px;
            background: rgba(255, 255, 255, 0.07);
        }
        .online-dot {
            color: #34e082;
            font-size: 13px;
            margin-top: 12px;
        }
        div[role="radiogroup"] label {
            border-radius: 8px;
            padding: 8px 10px;
            margin-bottom: 4px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            align-items: end;
            margin-bottom: 18px;
        }
        .greeting {
            color: #f4fbff;
            font-size: 15px;
            margin-bottom: 18px;
        }
        .top-controls {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            justify-content: end;
        }
        .control-pill {
            border: 1px solid rgba(87, 198, 255, 0.24);
            border-radius: 8px;
            background: rgba(3, 15, 28, 0.78);
            padding: 12px 18px;
            min-width: 180px;
            color: #eaf6ff;
        }
        .hero {
            padding: 0;
            background: transparent;
        }
        .hero-title {
            font-size: 28px;
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
            background:
                linear-gradient(160deg, rgba(9, 27, 48, 0.96), rgba(5, 14, 26, 0.86)),
                repeating-linear-gradient(90deg, rgba(39, 215, 255, 0.04) 0 1px, transparent 1px 32px);
            min-height: 128px;
            box-shadow: inset 0 0 22px rgba(39, 215, 255, 0.03);
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
        div[data-testid="stVerticalBlock"] > div:has(.panel-title) {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(5, 18, 32, 0.72);
            padding: 14px 16px;
        }
        .panel-title {
            color: #eaf6ff;
            font-weight: 760;
            font-size: 18px;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .risk-panel, .forecast-strip {
            border: 1px solid var(--line);
            border-radius: 8px;
            background:
                radial-gradient(circle at 50% 45%, rgba(39, 215, 255, 0.08), transparent 45%),
                rgba(4, 17, 31, 0.82);
            padding: 16px;
            min-height: 320px;
        }
        .panel-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .live-badge {
            font-size: 11px;
            color: #c8ffe2;
            border: 1px solid rgba(49, 209, 139, 0.35);
            border-radius: 6px;
            padding: 2px 7px;
            margin-left: 8px;
            background: rgba(49, 209, 139, 0.12);
        }
        .map-toggle {
            border: 1px solid rgba(39, 215, 255, 0.38);
            border-radius: 16px;
            padding: 5px 18px;
            color: #5de8ff;
            background: rgba(39, 215, 255, 0.11);
            font-size: 13px;
        }
        .map-stage {
            position: relative;
            height: 280px;
            overflow: hidden;
        }
        .legend {
            position: absolute;
            z-index: 4;
            left: 0;
            top: 18px;
            width: 118px;
            border: 1px solid rgba(87, 198, 255, 0.18);
            border-radius: 8px;
            background: rgba(5, 18, 32, 0.85);
            padding: 12px;
            font-size: 12px;
            color: #c7d8e5;
        }
        .legend p {
            margin: 8px 0 0 0;
        }
        .legend span {
            display: inline-block;
            width: 11px;
            height: 11px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .state-shape {
            position: absolute;
            left: 21%;
            top: 0;
            width: 70%;
            height: 100%;
            filter: drop-shadow(0 0 22px rgba(255, 159, 28, 0.22));
        }
        .zone {
            position: absolute;
            border: 1px solid rgba(255, 220, 151, 0.32);
            background: linear-gradient(135deg, #ba321e, #ff8c1a);
            opacity: 0.95;
        }
        .z1 {
            left: 2%;
            top: 24%;
            width: 45%;
            height: 45%;
            clip-path: polygon(12% 18%, 42% 0, 88% 16%, 95% 62%, 65% 100%, 15% 83%, 0 48%);
        }
        .z2 {
            left: 33%;
            top: 8%;
            width: 42%;
            height: 50%;
            background: linear-gradient(135deg, #ef6f16, #ffad24);
            clip-path: polygon(16% 0, 78% 6%, 100% 42%, 82% 88%, 30% 100%, 0 55%);
        }
        .z3 {
            left: 34%;
            top: 46%;
            width: 42%;
            height: 45%;
            background: linear-gradient(135deg, #f47c16, #ffc233);
            clip-path: polygon(4% 8%, 56% 0, 100% 38%, 77% 100%, 28% 88%, 0 56%);
        }
        .z4 {
            left: 65%;
            top: 48%;
            width: 30%;
            height: 34%;
            background: linear-gradient(135deg, #57b94e, #2e934b);
            clip-path: polygon(0 18%, 55% 0, 100% 24%, 88% 72%, 46% 100%, 8% 78%);
        }
        .map-marker {
            position: absolute;
            z-index: 5;
            transform: translate(-50%, -50%);
            color: white;
            font-size: 13px;
            font-weight: 760;
            text-shadow: 0 1px 6px #000;
        }
        .map-marker span {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid white;
            border-radius: 50%;
            margin-right: 5px;
            vertical-align: -2px;
        }
        .active-marker {
            background: rgba(0, 0, 0, 0.34);
            border-radius: 18px;
            padding: 5px 8px;
            box-shadow: 0 0 0 8px rgba(255, 75, 62, 0.16);
        }
        .map-hint {
            position: absolute;
            left: 0;
            bottom: 0;
            border: 1px solid rgba(87, 198, 255, 0.18);
            border-radius: 8px;
            background: rgba(5, 18, 32, 0.82);
            padding: 10px 12px;
            color: #d9e8f2;
            font-size: 13px;
        }
        .forecast-strip {
            min-height: 178px;
            margin-top: 16px;
        }
        .mini-label {
            float: right;
            color: #9db1c1;
            font-size: 11px;
            font-weight: 500;
        }
        .days {
            display: grid;
            grid-template-columns: repeat(7, minmax(68px, 1fr));
            gap: 8px;
            margin-top: 10px;
        }
        .day-card {
            border: 1px solid rgba(87, 198, 255, 0.10);
            border-radius: 8px;
            background: rgba(8, 24, 42, 0.75);
            text-align: center;
            padding: 10px 6px;
            min-height: 96px;
        }
        .sun-dot {
            width: 24px;
            height: 24px;
            border-radius: 50%;
            margin: 9px auto 6px auto;
            box-shadow: 0 0 18px currentColor;
        }
        .day-card span {
            display: block;
            color: #b6c8d5;
            font-size: 12px;
            margin-top: 2px;
        }
        .risk-bar {
            height: 5px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.08);
            margin-top: 12px;
            overflow: hidden;
        }
        .risk-bar span {
            display: block;
            height: 100%;
            background: linear-gradient(90deg, #ff4b3e, #ff9f1c, #31d18b);
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


def render_risk_map_panel(city: str):
    district_points = {
        "Bikaner": (50, 18, "#ff8c1a"),
        "Jaipur": (67, 35, "#ff9f1c"),
        "Ajmer": (58, 52, "#ff9f1c"),
        "Jodhpur": (35, 43, "#ff4b3e"),
        "Udaipur": (44, 76, "#ff7a1a"),
        "Bhilwara": (61, 72, "#ffbf2e"),
        "Kota": (79, 69, "#39c86a"),
    }
    markers = []
    for name, (left, top, color) in district_points.items():
        active = "active-marker" if name == city else ""
        markers.append(
            f"""
            <div class="map-marker {active}" style="left:{left}%;top:{top}%;">
                <span style="background:{color};"></span>{name}
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="risk-panel">
            <div class="panel-head">
                <div class="panel-title">RAJASTHAN RISK MAP <span class="live-badge">LIVE</span></div>
                <div class="map-toggle">Risk View</div>
            </div>
            <div class="map-stage">
                <div class="legend">
                    <div>RISK LEVEL</div>
                    <p><span style="background:#ff4b3e"></span>Very High</p>
                    <p><span style="background:#ff8c1a"></span>High</p>
                    <p><span style="background:#ffd166"></span>Moderate</p>
                    <p><span style="background:#39c86a"></span>Low</p>
                    <p><span style="background:#2aa6c8"></span>Very Low</p>
                </div>
                <div class="state-shape">
                    <div class="zone z1"></div>
                    <div class="zone z2"></div>
                    <div class="zone z3"></div>
                    <div class="zone z4"></div>
                    {''.join(markers)}
                </div>
                <div class="map-hint">Click on a district to view detailed analytics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_heatwave_days(base_temp: float, heatwave_percent: float):
    days = [
        ("Mon", base_temp, "High", "#ff9f1c"),
        ("Tue", base_temp + 2, "Very High", "#ff4b3e"),
        ("Wed", base_temp + 3, "Very High", "#ff4b3e"),
        ("Thu", base_temp + 1, "High", "#ff9f1c"),
        ("Fri", base_temp, "High", "#ffbf2e"),
        ("Sat", base_temp - 1, "Moderate", "#ffd166"),
        ("Sun", base_temp - 2, "Moderate", "#31d18b"),
    ]
    cards = []
    for day, temp, level, color in days:
        cards.append(
            f"""
            <div class="day-card">
                <div>{day}</div>
                <div class="sun-dot" style="background:{color};"></div>
                <strong style="color:{color};">{temp:.0f}C</strong>
                <span>{level}</span>
            </div>
            """
        )
    st.markdown(
        f"""
        <div class="forecast-strip">
            <div class="panel-title">HEATWAVE PREDICTION <span class="mini-label">7 DAYS FORECAST</span></div>
            <div class="days">{''.join(cards)}</div>
            <div class="risk-bar">
                <span style="width:{max(8, min(100, heatwave_percent))}%;"></span>
            </div>
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


def get_prediction_report(city: str, fallback_weather: dict):
    try:
        return get_heatwave_report(city)
    except Exception as exc:
        heatwave_percent, heatwave_level = predict_heatwave_from_weather(fallback_weather)
        return {
            "city": city,
            "weather": fallback_weather,
            "heatwave_percent": heatwave_percent,
            "risk_level": heatwave_level,
            "warning": f"Live weather API unavailable, showing demo values. {exc}",
        }


def main():
    apply_theme()
    recommendation_module = load_recommendation_module()

    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="brand-mark">THAR</div>
                <div class="brand-sub">VISION AI</div>
                <div class="brand-caption">AI-powered climate intelligence<br>for Rajasthan</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
        city = st.selectbox("District", sorted(DISTRICT_COORDS), index=sorted(DISTRICT_COORDS).index("Jodhpur"))
        selected_date = st.date_input("Date", value=pd.Timestamp("2025-05-26"))
        rainfall_mm = st.slider("30-day rainfall forecast (mm)", 0.0, 180.0, 32.6, 0.5)
        water_stress = st.select_slider("Water stress", options=["Low", "Moderate", "High", "Severe"], value="High")
        reservoir = st.slider("Reservoir/storage level (%)", 0.0, 100.0, 28.0, 1.0)
        crop_stage = st.selectbox("Crop stage", ["Pre-sowing", "Sowing", "Growing", "Harvest", "Off-season"], index=2)
        st.markdown(
            f"""
            <div class="side-status">
                <div style="font-size:12px;color:#dbefff;">TODAY'S OVERVIEW</div>
                <div style="font-weight:760;margin-top:12px;">{city}</div>
                <div style="color:#b8c7d4;margin-top:8px;">{selected_date.strftime("%d %b %Y")}</div>
                <div class="online-dot">● Operational</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="topbar">
            <div class="hero">
                <div class="greeting">Namaste, Arjun!</div>
                <div class="hero-title">Building a Climate Resilient Rajasthan</div>
                <div class="hero-subtitle">Real-time Predictions. Smarter Decisions. Sustainable Tomorrow.</div>
            </div>
            <div class="top-controls">
                <div class="control-pill">{selected_date.strftime("%d %b %Y")}</div>
                <div class="control-pill">{city}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    fallback_weather = make_manual_weather(
        city=city,
        temperature=43.0,
        humidity=31.0,
        pressure=1002.0,
        wind=12.0,
        radiation=720.0,
    )

    report = get_prediction_report(city, fallback_weather)
    error_message = report.get("warning")

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
            render_risk_map_panel(city)
            render_heatwave_days(weather["Temperature_C"], heatwave_percent)

        with right:
            st.markdown('<div class="panel-title">FORECASTING OVERVIEW</div>', unsafe_allow_html=True)
            forecast = pd.DataFrame(
                {
                    "Day": ["26 May", "2 Jun", "9 Jun", "16 Jun", "23 Jun", "30 Jun"],
                    "Temperature_C": [
                        weather["Temperature_C"],
                        weather["Temperature_C"] + 1.2,
                        weather["Temperature_C"] + 2.0,
                        weather["Temperature_C"] + 1.4,
                        weather["Temperature_C"] - 0.2,
                        weather["Temperature_C"] - 1.1,
                    ],
                    "Heatwave_Risk": [
                        heatwave_percent,
                        min(100, heatwave_percent + 4),
                        min(100, heatwave_percent + 8),
                        min(100, heatwave_percent + 5),
                        max(0, heatwave_percent - 2),
                        max(0, heatwave_percent - 7),
                    ],
                }
            )
            st.line_chart(forecast, x="Day", y=["Temperature_C", "Heatwave_Risk"])

            st.markdown('<div class="panel-title">AI RECOMMENDATIONS</div>', unsafe_allow_html=True)
            render_recommendations(recommendation_response["recommendations"][:3])

        bottom1, bottom2, bottom3, bottom4 = st.columns(4)
        with bottom1:
            render_metric_card("Quick Insights", f"{city}", "High heat risk due to temperature and radiation.", "#eaf6ff")
        with bottom2:
            render_metric_card("Top At Risk District", city, f"{heatwave_percent:.0f}% heatwave risk", "#ff9f1c")
        with bottom3:
            render_metric_card("Historical Comparison", "+2.4C", "Hotter than recent seasonal average", "#27d7ff")
        with bottom4:
            render_metric_card("Data Sources", "Live", "OpenWeatherMap, local model, project dataset", "#31d18b")

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
