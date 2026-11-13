"""Intersection pass: one Intersection Observation per Intersection in the Covered Area.

Each Intersection is judged from a 150 m square crop centred on it, cut from both
images of its Image Pair. A crop clipped by the image edge is still used, so the
Intersection is marked with a ring rather than assumed to be at the crop centre.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Callable

from PIL import Image as PILImage, ImageDraw

from .config import Settings
from .dataset import Image, ImagePair, load_pairs
from .store import Store
from .vlm import VLMClient, VLMError, yes_no

log = logging.getLogger(__name__)

MARKER_RADIUS_M = 12.0

OBSERVATION_SCHEMA = {
    "type": "object",
    "properties": {
        "roadway_visible": {"type": "boolean"},
        "new_water_on_roadway": {"type": "boolean"},
        "new_water_or_residue_close_by": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["roadway_visible", "new_water_on_roadway", "new_water_or_residue_close_by", "notes"],
}

PROMPT = (
    "Two aerial image crops of the same 150 m square of ground near Houston, Texas, north up.\n"
    "Image 1: BEFORE Hurricane Harvey (2016). Image 2: AFTER Harvey (days after the heaviest rain, 2017).\n"
    "A red ring in each marks the road intersection of {name}. 'The roadway' means the road "
    "surface inside the red ring. 'Close by' means anywhere else in the crop.\n\n"
    "Definitions:\n"
    "- New Water: water in the AFTER image where there was none in the BEFORE image. Permanent "
    "rivers, bayous, lakes, ponds, pools and retention basins are NOT New Water.\n"
    "- Flood Residue: traces that water was recently present and has receded: mud, brown sediment "
    "films, debris lines, piles of flood-damaged belongings at the curb.\n\n"
    "Answer three questions:\n"
    "1. roadway_visible: can the roadway inside the ring be seen in BOTH images (not hidden by trees, "
    "cloud, deep shadow, or blank/black image area)?\n"
    "2. new_water_on_roadway: in the AFTER image, is the roadway inside the ring under New Water?\n"
    "3. new_water_or_residue_close_by: in the AFTER image, is New Water or Flood Residue visible close "
    "by (elsewhere in the crop, not on the roadway inside the ring)?\n"
    "Reply with JSON: {{\"roadway_visible\": true|false, \"new_water_on_roadway\": true|false, "
    "\"new_water_or_residue_close_by\": true|false, \"notes\": one short sentence of what you saw}}."
)


@lru_cache(maxsize=64)
def _open(path: Path) -> PILImage.Image:
    img = PILImage.open(path)
    img.load()
    return img.convert("RGB")


def crop_around(image: Image, lat: float, lon: float, size_m: float, out_px: int, marker: bool = True) -> PILImage.Image:
    """Square crop of ground size_m centred on (lat, lon), clipped at the image edge, scaled to out_px."""
    src = _open(image.path)
    cx, cy = image.to_pixel(lat, lon)
    half = image.metres_to_pixels(size_m / 2)
    box = (max(0, round(cx - half)), max(0, round(cy - half)),
           min(src.width, round(cx + half)), min(src.height, round(cy + half)))
    crop = src.crop(box)
    scale = out_px / (2 * half)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), PILImage.LANCZOS)
    if marker:
        draw = ImageDraw.Draw(crop)
        mx, my = (cx - box[0]) * scale, (cy - box[1]) * scale
        r = image.metres_to_pixels(MARKER_RADIUS_M) * scale
        draw.ellipse((mx - r, my - r, mx + r, my + r), outline=(255, 0, 0), width=max(2, out_px // 150))
    return crop


def observe(client: VLMClient, settings: Settings, row, pair: ImagePair) -> tuple[dict, str]:
    roads = json.loads(row["roads"])
    before = crop_around(pair.before, row["lat"], row["lon"], settings.crop_size_m, settings.crop_output_px)
    after = crop_around(pair.after, row["lat"], row["lon"], settings.crop_size_m, settings.crop_output_px)
    answer, result = client.ask_json(PROMPT.format(name=" and ".join(roads)), [before, after], json_schema=OBSERVATION_SCHEMA)
    obs = {
        "roadway_visible": yes_no(answer.get("roadway_visible")) == "yes",
        "new_water_on_roadway": yes_no(answer.get("new_water_on_roadway")) == "yes",
        "new_water_or_residue_close_by": yes_no(answer.get("new_water_or_residue_close_by")) == "yes",
        "notes": str(answer.get("notes", "")).strip().replace("\n", " ")[:500],
    }
    return obs, result.text


def run_intersection_pass(settings: Settings, store: Store, limit: int | None = None, ids: list[str] | None = None,
                          pair_ids: list[str] | None = None, redo: bool = False,
                          progress: Callable[[str], None] = print) -> int:
    pairs = {p.pair_id: p for p in load_pairs(settings.dataset_dir)}
    rows = store.intersections(rated_only=True)
    if ids:
        rows = [r for r in rows if r["id"] in set(ids)]
    if pair_ids:
        rows = [r for r in rows if r["pair_id"] in set(pair_ids)]
    if not redo:
        done = store.observed_ids()
        rows = [r for r in rows if r["id"] not in done]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        progress("Intersection pass: nothing to do.")
        return 0

    client = VLMClient(settings)
    model = client.model
    progress(f"Intersection pass: observing {len(rows)} intersections with {model} at {settings.vlm_base_url}")
    count = 0
    with ThreadPoolExecutor(max_workers=max(1, settings.vlm_concurrency)) as pool:
        futures = {pool.submit(observe, client, settings, r, pairs[r["pair_id"]]): r for r in rows}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                obs, raw = fut.result()
            except (VLMError, OSError) as e:
                progress(f"  ! {r['id']} {r['name']}: {e}")
                continue
            store.save_observation(r["id"], r["pair_id"], obs, model, raw)
            count += 1
            progress(f"  [{count}/{len(rows)}] {r['name']}: visible={obs['roadway_visible']} "
                     f"on_road={obs['new_water_on_roadway']} close_by={obs['new_water_or_residue_close_by']} | {obs['notes'][:80]}")
    return count
