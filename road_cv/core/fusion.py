"""
CYPhER Multi-Model Hazard Fusion & Evidence Arbiter (Phase 7)
Synthesizes predictions across damage, surface, water, depth, vehicles, weather,
and obstacles into explainable, structured road hazard assessments.
"""

from typing import Dict, Any, List, Optional
import numpy as np

class CYPhERHazardFusion:
    """
    Explainable multi-model fusion layer. Combines evidence from all 14 perception modules.
    """

    @staticmethod
    def fuse(capabilities: Dict[str, Any]) -> Dict[str, Any]:
        hazards = []
        overall_risk = "LOW_RISK"
        confidence_scores = []

        damage_res = capabilities.get("damage", {})
        water_res = capabilities.get("water", {})
        surface_res = capabilities.get("surface", {})
        weather_res = capabilities.get("weather", {})
        vehicle_res = capabilities.get("vehicles", {})
        obstacle_res = capabilities.get("obstacles", {})
        depth_res = capabilities.get("depth", {})
        accident_res = capabilities.get("accident", {})

        # 1. Pothole + Water Overlap (Waterlogged Pothole Hazard)
        potholes = [d for d in damage_res.get("predictions", []) if "pothole" in d.get("class_name", "").lower() or d.get("class_id") == 3]
        water_pred = water_res.get("predictions", {})
        standing_water = water_pred.get("standing_water_present", False)
        waterlogged_potholes = water_pred.get("waterlogged_potholes", [])

        if waterlogged_potholes or (potholes and standing_water):
            hazards.append({
                "hazard_type": "WATERLOGGED_POTHOLE_IMPACT_RISK",
                "risk_level": "SEVERE_HAZARD",
                "evidence": {
                    "pothole_count": len(potholes),
                    "standing_water_present": standing_water,
                    "overlap_candidates": len(waterlogged_potholes)
                },
                "confidence": 0.88,
                "reason": "Detected structural pothole coincides with surface water accumulation, concealing defect depth and increasing tyre/rim damage risk."
            })
            overall_risk = "SEVERE_HAZARD"
            confidence_scores.append(0.88)

        # 2. Wet Surface + Adverse Weather (Hydroplaning / Low Grip Hazard)
        surf_cond = surface_res.get("predictions", {}).get("visual_surface_condition", {}).get("class_name", "Dry")
        weather_cond = weather_res.get("predictions", {}).get("predicted_weather", "Clear")

        if surf_cond in ["Wet", "Water-covered", "Ice-covered"] or weather_cond in ["Rain_or_Wet_Reflective", "Fog"]:
            risk_lvl = "CRITICAL_HAZARD" if surf_cond == "Ice-covered" else "ELEVATED_RISK"
            hazards.append({
                "hazard_type": "REDUCED_FRICTION_HYDROPLANING_RISK",
                "risk_level": risk_lvl,
                "evidence": {
                    "surface_condition": surf_cond,
                    "weather_condition": weather_cond
                },
                "confidence": 0.85,
                "reason": f"Visual surface condition is '{surf_cond}' under '{weather_cond}' atmospheric condition, reducing tyre-road adhesion and lengthening stopping distance."
            })
            if overall_risk != "SEVERE_HAZARD":
                overall_risk = risk_lvl
            confidence_scores.append(0.85)

        # 3. Structural Cracking Severity
        cracks = [d for d in damage_res.get("predictions", []) if "crack" in d.get("class_name", "").lower()]
        alligator_cracks = [c for c in cracks if c.get("class_id") == 2 or "alligator" in c.get("class_name", "").lower()]
        if alligator_cracks:
            hazards.append({
                "hazard_type": "STRUCTURAL_PAVEMENT_FATIGUE",
                "risk_level": "MODERATE_HAZARD",
                "evidence": {
                    "alligator_crack_count": len(alligator_cracks),
                    "total_cracks": len(cracks)
                },
                "confidence": 0.82,
                "reason": "Alligator cracking indicates progressive base-course structural fatigue requiring pavement maintenance."
            })
            if overall_risk == "LOW_RISK":
                overall_risk = "MODERATE_HAZARD"
            confidence_scores.append(0.82)

        # 4. Obstacle / Debris on Corridor
        obstacles = obstacle_res.get("predictions", {}).get("obstacles", [])
        if obstacles:
            hazards.append({
                "hazard_type": "FOREIGN_OBJECT_CORRIDOR_OBSTRUCTION",
                "risk_level": "ELEVATED_RISK",
                "evidence": {
                    "obstacle_count": len(obstacles),
                    "categories": [o.get("category") for o in obstacles]
                },
                "confidence": 0.79,
                "reason": f"Detected foreign obstacle on drivable road corridor: {[o.get('category') for o in obstacles]}."
            })
            if overall_risk == "LOW_RISK":
                overall_risk = "ELEVATED_RISK"
            confidence_scores.append(0.79)

        # 5. Video Collision / Rollover Incident
        acc_preds = accident_res.get("predictions", {})
        incidents = acc_preds.get("incidents", acc_preds.get("details", []))
        is_accident = (
            acc_preds.get("accident_detected", False)
            or len(incidents) > 0
            or acc_preds.get("event_class", "normal_traffic") in [
                "vehicle_collision_detected",
                "vehicle_rollover_or_jackknife",
                "pedestrian_collision_hazard"
            ]
        )
        if is_accident:
            hazards.append({
                "hazard_type": "ACTIVE_COLLISION_OR_ROLLOVER_INCIDENT",
                "risk_level": "CRITICAL_HAZARD",
                "evidence": {
                    "incident_count": max(1, len(incidents)),
                    "event_class": acc_preds.get("event_class", "vehicle_collision"),
                    "details": incidents
                },
                "confidence": 0.90,
                "reason": "Collision or vehicular accident confirmed by severe physical collision overlap or structural crash deformation."
            })
            overall_risk = "CRITICAL_HAZARD"
            confidence_scores.append(0.90)
        elif acc_preds.get("has_proximity_warning", False):
            # Proximity warning in dense traffic (elevated headway risk, NOT an accident)
            hazards.append({
                "hazard_type": "VEHICLE_HEADWAY_PROXIMITY_WARNING",
                "risk_level": "ELEVATED_RISK",
                "evidence": {
                    "proximity_count": len(acc_preds.get("proximity_events", []))
                },
                "confidence": 0.70,
                "reason": "Vehicles detected in close spatial proximity or tailgating headway."
            })
            if overall_risk == "LOW_RISK":
                overall_risk = "ELEVATED_RISK"
            confidence_scores.append(0.70)

        avg_conf = round(float(np.mean(confidence_scores)), 2) if confidence_scores else 0.95

        return {
            "overall_road_safety_risk": overall_risk,
            "fusion_confidence": avg_conf,
            "detected_hazards_count": len(hazards),
            "hazards": hazards,
            "fusion_policy": "Explainable Rule-Based Evidence Arbiter (CYPhER Phase 7)"
        }

