"""Alerts: turn a Flash Flood Warning into a message for one Alert Location.

The Alert names the Nearby (within NEARBY_KM) Intersections rated High and Moderate.
Low and Undetermined Intersections are shown on the map only. Nothing is sent anywhere.
"""
from __future__ import annotations

import json
import logging
import math
import urllib.parse
import urllib.request
from typing import Any

from .config import Settings
from .dataset import in_covered_area, load_pairs
from .geo import haversine_m
from .nws_warnings import warnings_covering
from .rating import Rating, rate, rating_reason
from .store import Store
from .vlm import VLMClient, VLMError

log = logging.getLogger(__name__)

NAMED_RATINGS = (Rating.HIGH, Rating.MODERATE)
COMPASS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]


def intersection_ratings(settings: Settings, store: Store) -> list[dict]:
    """Every Intersection with its Susceptibility Rating (or Unrated / not yet observed)."""
    pairs = {p.pair_id: p for p in load_pairs(settings.dataset_dir)}
    observations = store.observations()
    out = []
    for r in store.intersections():
        item: dict[str, Any] = {"id": r["id"], "name": r["name"], "roads": json.loads(r["roads"]),
                                "lat": r["lat"], "lon": r["lon"], "pair_id": r["pair_id"]}
        if r["pair_id"] is None:
            item.update(rating=Rating.UNRATED.value, reason="Outside the imagery's footprint")
        else:
            fema = pairs[r["pair_id"]].fema_sfha_tf
            item["fema_sfha_tf"] = fema
            obs = observations.get(r["id"])
            if obs is None:
                item.update(rating=Rating.PENDING.value, reason="Intersection pass has not run here yet")
            else:
                args = (bool(obs["roadway_visible"]), bool(obs["new_water_on_roadway"]),
                        bool(obs["new_water_or_residue_close_by"]), fema)
                item.update(rating=rate(*args).value, reason=rating_reason(*args),
                            observation={"roadway_visible": args[0], "new_water_on_roadway": args[1],
                                         "new_water_or_residue_close_by": args[2], "notes": obs["notes"],
                                         "model": obs["model"]})
        out.append(item)
    return out


def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    deg = math.degrees(math.atan2(math.sin(dl) * math.cos(p2),
                                  math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)))
    return COMPASS[round(((deg + 360) % 360) / 45) % 8]


def geocode(address: str, settings: Settings) -> tuple[float, float, str] | None:
    """Address -> (lat, lon, display name) via Nominatim. The address is sent to that service."""
    q = urllib.parse.urlencode({"q": address, "format": "json", "limit": 1, "countrycodes": "us"})
    req = urllib.request.Request(f"{settings.nominatim_url}?{q}", headers={"User-Agent": settings.user_agent})
    with urllib.request.urlopen(req, timeout=30) as resp:
        results = json.loads(resp.read().decode())
    if not results:
        return None
    return float(results[0]["lat"]), float(results[0]["lon"]), results[0].get("display_name", address)


def _warning_summary(w: dict) -> dict:
    p = w["properties"]
    return {"id": w["id"], "wfo": w["wfo"], "issued": w["issued"], "expires": w["expires"],
            "counties": p.get("locations", ""), "label": p.get("event_label", "Flash Flood Warning")}


def template_text(warning: dict, covered: bool, named: list[dict], location_label: str) -> str:
    head = (f"FLASH FLOOD WARNING for {location_label}: the National Weather Service warning issued "
            f"{warning['issued']} (until {warning['expires']}) covers this location.")
    if not covered:
        return head + " No intersection ratings are available here: this location is outside the aerial imagery's coverage."
    if not named:
        return head + (" No nearby intersections are rated High or Moderate, but conditions can change quickly; "
                       "avoid driving through any water.")
    lines = [f"- {i['name']} ({i['rating']}, {i['distance_km']:.1f} km {i['direction']}): {i['reason']}" for i in named]
    return head + " Intersections near you that flooded or are flood-prone:\n" + "\n".join(lines) + \
        "\nAvoid these intersections and never drive through water on the road."


ALERT_SYSTEM = (
    "You write short, calm, plain-language flash flood alerts for residents. Use only the facts given. "
    "Never invent places, depths, times or advice beyond the facts. Never call any place safe."
)


def compose_text(client: VLMClient | None, warning: dict, covered: bool, named: list[dict], location_label: str) -> tuple[str, str]:
    """Returns (text, composed_by). Falls back to a template if the language model is unavailable."""
    if client is None:
        return template_text(warning, covered, named, location_label), "template"
    facts = {
        "alert_location": location_label,
        "flash_flood_warning": {"issued_utc": warning["issued"], "expires_utc": warning["expires"],
                                "counties": warning["counties"]},
        "location_has_ratings": covered,
        "nearby_intersections_to_name": [
            {"name": i["name"], "rating": i["rating"], "distance_km": round(i["distance_km"], 1),
             "direction_from_location": i["direction"], "why": i["reason"]} for i in named],
    }
    if not covered:
        task = ("Write the alert in under 70 words. Say the warning applies to this location and that no "
                "intersection ratings are available for it because it is outside the aerial imagery coverage.")
    elif not named:
        task = ("Write the alert in under 70 words. Say the warning applies and that no nearby intersections "
                "are rated High or Moderate; still advise never driving into water.")
    else:
        task = ("Write the alert in under 160 words. Say the warning applies, then name EVERY intersection in "
                "nearby_intersections_to_name exactly as written, High ones first, with its distance and direction. "
                "Explain that High means the road was under water after Hurricane Harvey and Moderate means "
                "flood evidence close by or a FEMA high-risk zone. End with: never drive through flooded roads.")
    prompt = f"Facts (JSON):\n{json.dumps(facts, indent=1)}\n\n{task} Plain text, no markdown headings."
    try:
        text = client.ask_text(prompt, system=ALERT_SYSTEM, max_tokens=4096)
    except VLMError as e:
        log.warning("Alert composition failed, using template: %s", e)
        return template_text(warning, covered, named, location_label), "template (model unavailable)"
    missing = [i["name"] for i in named if i["name"].lower() not in text.lower()]
    if not text or missing:
        # The model must name every High/Moderate Intersection; append any it dropped.
        text = (text + "\n\nAlso flagged nearby: " + "; ".join(missing)) if text else template_text(warning, covered, named, location_label)
    return text, f"model:{client.model}"


def build_alert(settings: Settings, store: Store, lat: float, lon: float, warning_id: str | None = None,
                location_label: str | None = None, client: VLMClient | None = None,
                ratings: list[dict] | None = None) -> dict:
    location_label = location_label or f"{lat:.5f}, {lon:.5f}"
    all_warnings = store.warnings()
    covering = warnings_covering(lat, lon, all_warnings)
    if warning_id:
        covering = [w for w in covering if w["id"] == warning_id]
    covered = in_covered_area(lat, lon, load_pairs(settings.dataset_dir))
    result: dict[str, Any] = {"location": {"lat": lat, "lon": lon, "label": location_label},
                              "in_covered_area": covered,
                              "covering_warnings": [_warning_summary(w) for w in covering]}
    if not covering:
        result.update(status="no_warning", text="No Flash Flood Warning covers this location, so no Alert is produced.")
        return result

    warning = _warning_summary(covering[0])
    nearby = []
    if covered:
        for item in ratings if ratings is not None else intersection_ratings(settings, store):
            d = haversine_m(lat, lon, item["lat"], item["lon"]) / 1000
            if d <= settings.nearby_km:
                nearby.append({**item, "distance_km": d, "direction": bearing(lat, lon, item["lat"], item["lon"])})
    order = {Rating.HIGH.value: 0, Rating.MODERATE.value: 1}
    named = sorted((i for i in nearby if i["rating"] in order), key=lambda i: (order[i["rating"]], i["distance_km"]))
    text, composed_by = compose_text(client, warning, covered, named, location_label)
    counts: dict[str, int] = {}
    for i in nearby:
        counts[i["rating"]] = counts.get(i["rating"], 0) + 1
    result.update(status="alert", warning=warning, text=text, composed_by=composed_by,
                  named_intersections=named, nearby_counts=counts,
                  nearby_ids=[i["id"] for i in nearby])
    return result
