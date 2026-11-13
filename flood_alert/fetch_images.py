"""Download the Harvey images when the dataset clone holds only Git LFS pointers.

Files go to data/images/ and are checked against the dataset's sha256 column.
The dataset checkout itself is left untouched.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from .config import Settings
from .dataset import Image, is_real_image, load_pairs

HF_RESOLVE = os.environ.get(
    "FLOOD_IMAGES_BASE_URL",
    "https://huggingface.co/datasets/jennifer-fowler/UTSA-flood-images/resolve/main/images/",
)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(img: Image, settings: Settings) -> str:
    dest = img.cache_path
    if dest.exists() and (not img.sha256 or _sha256(dest) == img.sha256):
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(HF_RESOLVE + dest.name, headers={"User-Agent": settings.user_agent})
    last: Exception | None = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
                while chunk := resp.read(1 << 16):
                    out.write(chunk)
            if img.sha256 and _sha256(tmp) != img.sha256:
                raise ValueError("sha256 mismatch")
            tmp.replace(dest)
            return "downloaded"
        except (OSError, ValueError) as e:
            last = e
    raise RuntimeError(f"{dest.name}: {last}")


def fetch_images(settings: Settings, pair_ids: list[str] | None = None, workers: int = 8,
                 progress: Callable[[str], None] = print) -> int:
    pairs = [p for p in load_pairs(settings.dataset_dir) if not pair_ids or p.pair_id in set(pair_ids)]
    images = [img for p in pairs for img in (p.before, p.after)]
    todo = [img for img in images if not is_real_image(img.dataset_path)]
    if not todo:
        progress("Images: dataset already holds real image files.")
        return 0
    progress(f"Images: {len(todo)} of {len(images)} are LFS pointers; fetching into {todo[0].cache_path.parent}")
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download, img, settings) for img in todo]
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
            except RuntimeError as e:
                failed += 1
                progress(f"  ! {e}")
            if (done + failed) % 100 == 0:
                progress(f"  {done + failed}/{len(todo)}")
    progress(f"Images: {done} ready, {failed} failed")
    return done
