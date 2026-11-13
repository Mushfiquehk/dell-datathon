"""Susceptibility Rating, read directly from an Intersection Observation (see CONTEXT.md)."""
from __future__ import annotations

from enum import Enum


class Rating(str, Enum):
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    UNDETERMINED = "Undetermined"
    UNRATED = "Unrated"        # outside the Covered Area: nothing is known
    PENDING = "Not yet observed"  # inside the Covered Area, intersection pass has not run


def rate(roadway_visible: bool, new_water_on_roadway: bool, new_water_or_residue_close_by: bool, fema_sfha_tf: str) -> Rating:
    """
    High:         roadway visibly under New Water in the after image.
    Moderate:     not submerged, but New Water / Flood Residue Close By, or in a FEMA High-Risk Zone.
    Low:          no evidence Close By AND known to be outside the FEMA High-Risk Zone (fema_sfha_tf == "F").
    Undetermined: roadway cannot be seen in one or both images.

    A blank FEMA value means "not known to be outside", so it can never give Low; with no
    imagery evidence such an Intersection is Moderate.
    """
    if not roadway_visible:
        return Rating.UNDETERMINED
    if new_water_on_roadway:
        return Rating.HIGH
    if new_water_or_residue_close_by:
        return Rating.MODERATE
    if (fema_sfha_tf or "").strip().upper() == "F":
        return Rating.LOW
    return Rating.MODERATE


def rating_reason(roadway_visible: bool, new_water_on_roadway: bool, new_water_or_residue_close_by: bool, fema_sfha_tf: str) -> str:
    fema = (fema_sfha_tf or "").strip().upper()
    if not roadway_visible:
        return "Roadway not visible in one or both images"
    if new_water_on_roadway:
        return "Roadway under water after Harvey"
    if new_water_or_residue_close_by:
        return "New water or flood residue close by after Harvey"
    if fema == "T":
        return "In a FEMA High-Risk Zone"
    if fema == "F":
        return "No flood evidence nearby; outside FEMA High-Risk Zone"
    return "No flood evidence nearby, but FEMA mapping unavailable"
