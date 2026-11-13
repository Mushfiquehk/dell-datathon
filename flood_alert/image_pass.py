"""Image pass: fill the dataset's blank Image Label fields for the 1,150 Harvey images.

Labels go into this project's sqlite store and an exported copy of the metadata CSV;
nothing is written back to the dataset.
"""
from __future__ import annotations

import csv
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from .config import Settings
from .dataset import Image, ImagePair, load_pairs, read_metadata
from .store import Store
from .perception import Seen, look_at_scene
from .vlm import VLMClient, VLMError

log = logging.getLogger(__name__)

def _yn(flag: bool) -> str:
    return "yes" if flag else "no"


def labels_for_pair(before: Seen, after: Seen) -> tuple[dict, dict]:
    """Image Labels for both images of a pair from what the model saw in each.

    New Water is water in the after image where there was none before, so for the after image:
    water seen after and not before -> yes; water seen in both -> uncertain (it may be permanent or
    the same wet area); no water after -> no. The before image has no earlier reference, so its
    labels are what was seen in it.
    """
    before_label = {"flood_visible": _yn(before.standing_water), "flood_aftermath_visible": _yn(before.mud_or_debris),
                    "notes": before.description}
    if after.standing_water and before.standing_water:
        water = "uncertain"
    else:
        water = _yn(after.standing_water)
    after_label = {"flood_visible": water, "flood_aftermath_visible": _yn(after.mud_or_debris),
                   "notes": after.description}
    return before_label, after_label


def label_pair(client: VLMClient, pair: ImagePair) -> list[tuple[Image, dict, str]]:
    before = look_at_scene(client, pair.before.path)
    after = look_at_scene(client, pair.after.path)
    b, a = labels_for_pair(before, after)
    return [(pair.before, b, before.raw), (pair.after, a, after.raw)]


def run_image_pass(settings: Settings, store: Store, limit: int | None = None, image_ids: list[str] | None = None,
                   redo: bool = False, progress: Callable[[str], None] = print) -> int:
    """Labels images a pair at a time (the after label needs the before image), so a requested image
    also labels its partner. `limit` counts pairs."""
    pairs = list(load_pairs(settings.dataset_dir))
    if image_ids:
        wanted = set(image_ids)
        pairs = [p for p in pairs if p.before.image_id in wanted or p.after.image_id in wanted]
    if not redo:
        done = store.labelled_image_ids()
        pairs = [p for p in pairs if not (p.before.image_id in done and p.after.image_id in done)]
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        progress("Image pass: nothing to do.")
        return 0

    client = VLMClient(settings)
    annotator = f"vlm:{client.model}"
    progress(f"Image pass: labelling {len(pairs)} pairs ({2 * len(pairs)} images) with {client.model} at {settings.vlm_base_url}")
    count = 0
    with ThreadPoolExecutor(max_workers=max(1, settings.vlm_concurrency)) as pool:
        futures = {pool.submit(label_pair, client, p): p for p in pairs}
        for fut in as_completed(futures):
            pair = futures[fut]
            try:
                results = fut.result()
            except (VLMError, OSError) as e:
                progress(f"  ! {pair.pair_id}: {e}")
                continue
            for img, label, raw in results:
                store.save_image_label(img.image_id, label, annotator, raw)
                count += 1
                progress(f"  [{count}/{2 * len(pairs)}] {img.image_id}: water={label['flood_visible']} "
                         f"residue={label['flood_aftermath_visible']} | {label['notes'][:90]}")
    return count


def export_labelled_metadata(settings: Settings, store: Store) -> int:
    """Write this project's own copy of image_metadata.csv with the label fields filled in."""
    fields, rows = read_metadata(settings.metadata_csv)
    labels = store.image_labels()
    filled = 0
    for row in rows:
        lab = labels.get(row["image_id"])
        if lab is None:
            continue
        row["flood_visible"] = lab["flood_visible"]
        row["flood_aftermath_visible"] = lab["flood_aftermath_visible"]
        row["annotator"] = lab["annotator"]
        row["annotation_notes"] = lab["annotation_notes"]
        row["annotation_status"] = "model_labelled"
        filled += 1
    settings.labelled_metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.labelled_metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return filled
