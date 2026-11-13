"""Intersection pass: one Intersection Observation per Intersection in the Covered Area.

Each Intersection is judged from a 150 m square crop centred on it, cut from both
images of its Image Pair. A crop clipped by the image edge is still used, so the
Intersection is marked with a ring rather than assumed to be at the crop centre.
Each crop is described on its own (see perception.py); the comparison is done here.
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
from .perception import Seen, look_at_crop
from .vlm import VLMClient, VLMError

log = logging.getLogger(__name__)

MARKER_RADIUS_M = 12.0

def to_observation(before: Seen, after: Seen) -> dict:
    """Combine what the model saw in each crop into an Intersection Observation (CONTEXT.md).

    - Roadway visible: the ring shows paved road in the before crop, and in the after crop it shows
      paved road, standing water or mud (a submerged road is visibly under water, so it counts as seen).
    - New Water on the roadway: the after ring shows standing water.
    - New Water or Flood Residue close by: standing water elsewhere in the after crop that the
      before crop did not have, mud or debris elsewhere in the after crop, or mud on the roadway itself
      (residue without standing water is not High, but it is evidence).
    """
    b, a = before.ring_surface, after.ring_surface
    visible = b == "paved_road" and a in ("paved_road", "standing_water", "mud")
    on_road = visible and a == "standing_water"
    new_water_elsewhere = after.standing_water and not before.standing_water
    close_by = new_water_elsewhere or after.mud_or_debris or a == "mud"
    return {
        "roadway_visible": visible,
        "new_water_on_roadway": on_road,
        "new_water_or_residue_close_by": close_by,
        "notes": (f"[ring before: {b}; after: {a}; water elsewhere before/after: "
                  f"{before.standing_water}/{after.standing_water}; mud/debris after: {after.mud_or_debris}] "
                  f"{after.description}"),
    }


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
    size, px = settings.crop_size_m, settings.crop_output_px
    before = look_at_crop(client, crop_around(pair.before, row["lat"], row["lon"], size, px), size)
    after = look_at_crop(client, crop_around(pair.after, row["lat"], row["lon"], size, px), size)
    raw = json.dumps({"before": before.raw, "after": after.raw})
    return to_observation(before, after), raw


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
