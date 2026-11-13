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
from .vlm import VLMClient, VLMError, yes_no

log = logging.getLogger(__name__)

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "flood_visible": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "flood_aftermath_visible": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "notes": {"type": "string"},
    },
    "required": ["flood_visible", "flood_aftermath_visible", "notes"],
}

DEFINITIONS = (
    "Definitions:\n"
    "- New Water: water present where there was none before. Permanent rivers, bayous, lakes, ponds, "
    "pools and retention basins are NOT New Water.\n"
    "- Flood Residue: visible traces that water was recently present and has receded, such as mud, "
    "brown sediment films, debris lines, or piles of flood-damaged belongings at the curb.\n"
)

AFTER_PROMPT = (
    "You are labelling aerial imagery (about 530 m across, north up) from Hurricane Harvey, Texas.\n"
    "The FIRST image was taken before the hurricane (reference). The SECOND image is the same ground "
    "taken in the days after it. Label the SECOND image only, comparing it with the first.\n\n"
    + DEFINITIONS +
    "\nAnswer with JSON: {\"flood_visible\": yes|no|uncertain (is New Water visible anywhere in the "
    "second image?), \"flood_aftermath_visible\": yes|no|uncertain (is Flood Residue visible?), "
    "\"notes\": one short sentence saying what you saw and where}."
)

BEFORE_PROMPT = (
    "You are labelling an aerial image (about 530 m across, north up) of the Houston, Texas area, "
    "taken in 2016 as a pre-hurricane baseline.\n\n"
    + DEFINITIONS +
    "For this image, treat New Water as water clearly out of place: on roads, in yards, around "
    "buildings or spread over fields outside any channel or pond.\n"
    "\nAnswer with JSON: {\"flood_visible\": yes|no|uncertain (is New Water visible?), "
    "\"flood_aftermath_visible\": yes|no|uncertain (is Flood Residue visible?), "
    "\"notes\": one short sentence saying what you saw}."
)


def label_image(client: VLMClient, image: Image, pair: ImagePair) -> tuple[dict, str]:
    if image.role == "after":
        prompt, images = AFTER_PROMPT, [pair.before.path, image.path]
    else:
        prompt, images = BEFORE_PROMPT, [image.path]
    answer, result = client.ask_json(prompt, images, json_schema=LABEL_SCHEMA)
    label = {
        "flood_visible": yes_no(answer.get("flood_visible")),
        "flood_aftermath_visible": yes_no(answer.get("flood_aftermath_visible")),
        "notes": str(answer.get("notes", "")).strip().replace("\n", " ")[:500],
    }
    return label, result.text


def run_image_pass(settings: Settings, store: Store, limit: int | None = None, image_ids: list[str] | None = None,
                   redo: bool = False, progress: Callable[[str], None] = print) -> int:
    pairs = load_pairs(settings.dataset_dir)
    work: list[tuple[Image, ImagePair]] = []
    for p in pairs:
        work += [(p.before, p), (p.after, p)]
    if image_ids:
        wanted = set(image_ids)
        work = [w for w in work if w[0].image_id in wanted]
    if not redo:
        done = store.labelled_image_ids()
        work = [w for w in work if w[0].image_id not in done]
    if limit is not None:
        work = work[:limit]
    if not work:
        progress("Image pass: nothing to do.")
        return 0

    client = VLMClient(settings)
    annotator = f"vlm:{client.model}"
    progress(f"Image pass: labelling {len(work)} images with {client.model} at {settings.vlm_base_url}")
    done_count = 0
    with ThreadPoolExecutor(max_workers=max(1, settings.vlm_concurrency)) as pool:
        futures = {pool.submit(label_image, client, img, pair): img for img, pair in work}
        for fut in as_completed(futures):
            img = futures[fut]
            try:
                label, raw = fut.result()
            except VLMError as e:
                progress(f"  ! {img.image_id}: {e}")
                continue
            store.save_image_label(img.image_id, label, annotator, raw)
            done_count += 1
            progress(f"  [{done_count}/{len(work)}] {img.image_id}: water={label['flood_visible']} "
                     f"residue={label['flood_aftermath_visible']} | {label['notes'][:90]}")
    return done_count


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
