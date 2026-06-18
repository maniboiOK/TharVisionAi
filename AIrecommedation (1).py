from __future__ import annotations

from enum import Enum
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator


class ImpactLevel(str, Enum):
    high = "High Impact"
    medium = "Medium Impact"
    low = "Low Impact"


class UrgencyLevel(str, Enum):
    immediate = "Immediate"
    this_week = "This Week"
    this_month = "This Month"
    seasonal = "Seasonal"


class ClimateSnapshot(BaseModel):
    district: str = Field(..., examples=["Jodhpur"])
    drought_risk: float = Field(..., ge=0, le=100, examples=[82])
    heatwave_risk: float = Field(..., ge=0, le=100, examples=[64])
    rainfall_forecast_mm: float = Field(..., ge=0, examples=[32.6])
    water_stress: Literal["Low", "Moderate", "High", "Severe"] = "High"
    avg_temperature_c: float | None = Field(default=None, examples=[43.5])
    reservoir_level_percent: float | None = Field(default=None, ge=0, le=100, examples=[28])
    crop_stage: Literal["Pre-sowing", "Sowing", "Growing", "Harvest", "Off-season"] = (
        "Growing"
    )

    @field_validator("district")
    @classmethod
    def normalize_district(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("district cannot be empty")
        return cleaned.title()


class Recommendation(BaseModel):
    title: str
    description: str
    impact: ImpactLevel
    urgency: UrgencyLevel
    category: Literal[
        "Water",
        "Agriculture",
        "Heat Safety",
        "Governance",
        "Monitoring",
    ]
    confidence: float = Field(..., ge=0, le=1)
    reason: str


class RecommendationResponse(BaseModel):
    district: str
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: Literal["Low", "Moderate", "High", "Very High"]
    recommendations: list[Recommendation]


app = FastAPI(
    title="THAR VisionAI Recommendation API",
    description="AI-style climate action recommendations for Rajasthan districts.",
    version="1.0.0",
)


WATER_STRESS_SCORE = {
    "Low": 15,
    "Moderate": 45,
    "High": 75,
    "Severe": 95,
}

ARID_DISTRICTS = {
    "Jodhpur",
    "Jaisalmer",
    "Barmer",
    "Bikaner",
    "Nagaur",
    "Churu",
    "Pali",
}


def calculate_risk_score(snapshot: ClimateSnapshot) -> int:
    rainfall_deficit_score = max(0, 100 - (snapshot.rainfall_forecast_mm / 75) * 100)
    water_score = WATER_STRESS_SCORE[snapshot.water_stress]

    score = (
        snapshot.drought_risk * 0.38
        + snapshot.heatwave_risk * 0.27
        + rainfall_deficit_score * 0.2
        + water_score * 0.15
    )

    if snapshot.reservoir_level_percent is not None and snapshot.reservoir_level_percent < 35:
        score += 5

    return round(min(score, 100))


def get_risk_level(score: int) -> Literal["Low", "Moderate", "High", "Very High"]:
    if score >= 75:
        return "Very High"
    if score >= 55:
        return "High"
    if score >= 35:
        return "Moderate"
    return "Low"


def build_recommendations(snapshot: ClimateSnapshot) -> list[Recommendation]:
    district_is_arid = snapshot.district in ARID_DISTRICTS
    recommendations: list[Recommendation] = []

    if snapshot.drought_risk >= 75 or snapshot.water_stress in {"High", "Severe"}:
        recommendations.append(
            Recommendation(
                title="Prioritize rainwater harvesting and recharge points",
                description=(
                    "Map public buildings, school roofs, and village ponds for recharge "
                    "work before the next rainfall window."
                ),
                impact=ImpactLevel.high,
                urgency=UrgencyLevel.this_month,
                category="Water",
                confidence=0.91,
                reason=(
                    f"Drought risk is {snapshot.drought_risk:.0f}% and water stress is "
                    f"{snapshot.water_stress.lower()}."
                ),
            )
        )

    if snapshot.rainfall_forecast_mm < 40:
        recommendations.append(
            Recommendation(
                title="Shift irrigation to deficit mode",
                description=(
                    "Use drip scheduling, night irrigation, and alternate-furrow watering "
                    "for non-critical crop stages."
                ),
                impact=ImpactLevel.high,
                urgency=UrgencyLevel.this_week,
                category="Agriculture",
                confidence=0.88,
                reason=(
                    f"Only {snapshot.rainfall_forecast_mm:.1f} mm rainfall is forecast "
                    "for the next 30 days."
                ),
            )
        )

    if snapshot.heatwave_risk >= 60:
        recommendations.append(
            Recommendation(
                title="Activate heatwave work-hour advisory",
                description=(
                    "Avoid outdoor labor during peak afternoon hours and prepare cooling "
                    "points near markets, farms, and bus stands."
                ),
                impact=ImpactLevel.medium,
                urgency=UrgencyLevel.immediate,
                category="Heat Safety",
                confidence=0.86,
                reason=f"Heatwave risk is {snapshot.heatwave_risk:.0f}%.",
            )
        )

    if district_is_arid and snapshot.crop_stage in {"Pre-sowing", "Sowing", "Growing"}:
        recommendations.append(
            Recommendation(
                title="Prefer drought-resistant crop choices",
                description=(
                    "Recommend bajra, moth bean, guar, sesame, or cluster bean where "
                    "soil and market conditions support the switch."
                ),
                impact=ImpactLevel.medium,
                urgency=UrgencyLevel.seasonal,
                category="Agriculture",
                confidence=0.82,
                reason=f"{snapshot.district} is in the arid risk belt.",
            )
        )

    if snapshot.reservoir_level_percent is not None and snapshot.reservoir_level_percent < 30:
        recommendations.append(
            Recommendation(
                title="Trigger drinking-water contingency planning",
                description=(
                    "Prepare tanker routes, check borewell status, and protect drinking "
                    "water allocation from non-essential use."
                ),
                impact=ImpactLevel.high,
                urgency=UrgencyLevel.immediate,
                category="Governance",
                confidence=0.9,
                reason=(
                    f"Reservoir/storage level is only "
                    f"{snapshot.reservoir_level_percent:.0f}%."
                ),
            )
        )

    if snapshot.drought_risk >= 60 or snapshot.heatwave_risk >= 50:
        recommendations.append(
            Recommendation(
                title="Increase district monitoring frequency",
                description=(
                    "Update satellite vegetation index, rainfall anomaly, and water-point "
                    "status at least twice a week."
                ),
                impact=ImpactLevel.low,
                urgency=UrgencyLevel.this_week,
                category="Monitoring",
                confidence=0.79,
                reason="Multiple climate indicators are above normal watch thresholds.",
            )
        )

    if not recommendations:
        recommendations.append(
            Recommendation(
                title="Maintain normal climate watch",
                description=(
                    "Keep weekly monitoring active and preserve water-saving advisories "
                    "for vulnerable blocks."
                ),
                impact=ImpactLevel.low,
                urgency=UrgencyLevel.seasonal,
                category="Monitoring",
                confidence=0.72,
                reason="Current risk indicators are below alert thresholds.",
            )
        )

    return sorted(
        recommendations,
        key=lambda item: (
            {
                ImpactLevel.high: 3,
                ImpactLevel.medium: 2,
                ImpactLevel.low: 1,
            }[item.impact],
            item.confidence,
        ),
        reverse=True,
    )


def recommend_actions(snapshot: ClimateSnapshot) -> RecommendationResponse:
    risk_score = calculate_risk_score(snapshot)
    return RecommendationResponse(
        district=snapshot.district,
        risk_score=risk_score,
        risk_level=get_risk_level(risk_score),
        recommendations=build_recommendations(snapshot),
    )


@app.get("/")
def health_check() -> dict[str, str]:
    return {
        "project": "THAR VisionAI",
        "module": "AI Recommendations",
        "status": "ready",
    }


@app.post("/recommendations", response_model=RecommendationResponse)
def create_recommendations(snapshot: ClimateSnapshot) -> RecommendationResponse:
    return recommend_actions(snapshot)


@app.get("/demo/jodhpur", response_model=RecommendationResponse)
def demo_jodhpur() -> RecommendationResponse:
    return recommend_actions(
        ClimateSnapshot(
            district="Jodhpur",
            drought_risk=82,
            heatwave_risk=64,
            rainfall_forecast_mm=32.6,
            water_stress="High",
            avg_temperature_c=43.5,
            reservoir_level_percent=28,
            crop_stage="Growing",
        )
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("AIrecommedation:app", host="127.0.0.1", port=8000, reload=True)
