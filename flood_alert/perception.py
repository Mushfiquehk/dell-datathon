"""What the model sees in ONE image, asked neutrally.

The 8B model is very suggestible: shown a before/after pair and told it is Hurricane
Harvey, it reports flooding almost everywhere; given a vivid description of flooded
streets, it sees water on dry roads. So each image is described on its own, with no
mention of floods or hurricanes, and the before/after comparison that defines New
Water is done in code (see image_pass.py and intersection_pass.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .vlm import VLMClient, yes_no

RING_SURFACES = ["paved_road", "standing_water", "mud", "trees", "roof", "grass_or_bare_soil", "cannot_tell"]

_ELSEWHERE = (
    "standing_water = is there standing water on streets, driveways, yards, parking lots, fields or around "
    "buildings? Swimming pools, ponds, lakes, rivers, bayous, reservoirs and drainage channels do not count. "
    "mud_or_debris = are there brown mud or sediment films on pavement or lawns, debris lines, or piles of "
    "debris at the curb? Answer true only if you can point to it."
)

RING_PROMPT = (
    "Aerial photo, north up, about {size:.0f} m across. Look only at the area inside the red ring. "
    "Describe in one sentence what you see inside the ring, then classify it: paved_road (grey/white "
    "asphalt or concrete with visible edges, lane lines or curbs), standing_water (a smooth sheet of "
    "water with no road surface visible), mud, trees, roof, grass_or_bare_soil, or cannot_tell. "
    "Then, looking at the rest of the photo OUTSIDE the ring: " + _ELSEWHERE +
    " Reply JSON {{\"description\": ..., \"ring_surface\": ..., \"standing_water\": true|false, "
    "\"mud_or_debris\": true|false}}."
)

SCENE_PROMPT = (
    "Aerial photo, north up, about 530 m across. Describe in one sentence what the area is and anything "
    "notable about water or ground conditions. Then: " + _ELSEWHERE +
    " Reply JSON {\"description\": ..., \"standing_water\": true|false, \"mud_or_debris\": true|false}."
)

RING_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "ring_surface": {"type": "string", "enum": RING_SURFACES},
        "standing_water": {"type": "boolean"},
        "mud_or_debris": {"type": "boolean"},
    },
    "required": ["description", "ring_surface", "standing_water", "mud_or_debris"],
}

SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "standing_water": {"type": "boolean"},
        "mud_or_debris": {"type": "boolean"},
    },
    "required": ["description", "standing_water", "mud_or_debris"],
}


@dataclass
class Seen:
    description: str
    standing_water: bool
    mud_or_debris: bool
    ring_surface: str | None = None
    raw: str = ""


def _seen(answer: dict[str, Any], raw: str, ring: bool) -> Seen:
    surface = None
    if ring:
        surface = str(answer.get("ring_surface", "cannot_tell")).strip().lower()
        if surface not in RING_SURFACES:
            surface = "cannot_tell"
    return Seen(
        description=str(answer.get("description", "")).strip().replace("\n", " ")[:300],
        standing_water=yes_no(answer.get("standing_water")) == "yes",
        mud_or_debris=yes_no(answer.get("mud_or_debris")) == "yes",
        ring_surface=surface,
        raw=raw,
    )


def look_at_crop(client: VLMClient, crop, size_m: float) -> Seen:
    answer, result = client.ask_json(RING_PROMPT.format(size=size_m), [crop], json_schema=RING_SCHEMA)
    return _seen(answer, result.text, ring=True)


def look_at_scene(client: VLMClient, image) -> Seen:
    answer, result = client.ask_json(SCENE_PROMPT, [image], json_schema=SCENE_SCHEMA)
    return _seen(answer, result.text, ring=False)
