"""Loads the Hurricane Harvey Image Pairs from the dataset's image_metadata.csv.

Only the 575 Harvey before/after pairs are in scope; the Hill Country images and
Coast Guard video frames are ignored.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .geo import to_mercator

HARVEY_EVENT_ID = "harvey_2017"


@dataclass(frozen=True)
class Image:
    image_id: str
    dataset_path: Path   # may be a Git LFS pointer if the dataset was cloned without git-lfs
    cache_path: Path     # where `fetch-images` puts the real file
    sha256: str
    role: str  # "before" or "after"
    acquisition_date: str
    west: float
    south: float
    east: float
    north: float
    width_px: int
    height_px: int
    fema_sfha_tf: str  # "T", "F" or "" (blank = not known to be outside)
    locality: str

    @property
    def path(self) -> Path:
        if is_real_image(self.dataset_path):
            return self.dataset_path
        return self.cache_path

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east

    @property
    def center(self) -> tuple[float, float]:
        return ((self.south + self.north) / 2, (self.west + self.east) / 2)

    def to_pixel(self, lat: float, lon: float) -> tuple[float, float]:
        """WGS84 -> pixel (x right, y down). Images are on an EPSG:3857 grid."""
        x0, y0 = to_mercator(self.north, self.west)
        x1, y1 = to_mercator(self.south, self.east)
        x, y = to_mercator(lat, lon)
        return ((x - x0) / (x1 - x0) * self.width_px, (y0 - y) / (y0 - y1) * self.height_px)

    def metres_to_pixels(self, metres: float) -> float:
        """Ground metres -> pixels at the image centre."""
        lat, lon = self.center
        from .geo import haversine_m

        width_m = haversine_m(lat, self.west, lat, self.east)
        return metres / width_m * self.width_px


@dataclass(frozen=True)
class ImagePair:
    pair_id: str
    before: Image
    after: Image

    @property
    def fema_sfha_tf(self) -> str:
        # Both images share a footprint; prefer the after image's value, fall back to before.
        return self.after.fema_sfha_tf or self.before.fema_sfha_tf

    @property
    def locality(self) -> str:
        return self.after.locality

    def contains(self, lat: float, lon: float) -> bool:
        return self.after.contains(lat, lon)

    def centre_offset(self, lat: float, lon: float) -> float:
        """Normalised distance from the footprint centre (0 = centre, 1 = edge midpoint)."""
        a = self.after
        cy, cx = a.center
        return max(abs(lat - cy) / ((a.north - a.south) / 2), abs(lon - cx) / ((a.east - a.west) / 2))


def is_real_image(path: Path) -> bool:
    """False for a missing file or a Git LFS pointer (a ~130 byte text stub)."""
    try:
        return path.stat().st_size > 1024
    except OSError:
        return False


def _image(row: dict, dataset_dir: Path, cache_dir: Path) -> Image:
    return Image(
        image_id=row["image_id"],
        dataset_path=dataset_dir / row["image_path"],
        cache_path=cache_dir / Path(row["image_path"]).name,
        sha256=row.get("sha256", ""),
        role="before" if row["image_role"] == "before" else "after",
        acquisition_date=row["acquisition_date"],
        west=float(row["west"]),
        south=float(row["south"]),
        east=float(row["east"]),
        north=float(row["north"]),
        width_px=int(row["width_px"]),
        height_px=int(row["height_px"]),
        fema_sfha_tf=(row.get("fema_sfha_tf") or "").strip().upper(),
        locality=row.get("locality", ""),
    )


def read_metadata(metadata_csv: Path) -> tuple[list[str], list[dict]]:
    with open(metadata_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def default_cache_dir(dataset_dir: Path) -> Path:
    from .config import get_settings

    return get_settings().data_dir / "images"


@lru_cache(maxsize=4)
def load_pairs(dataset_dir: Path, cache_dir: Path | None = None) -> tuple[ImagePair, ...]:
    cache_dir = cache_dir or default_cache_dir(dataset_dir)
    _, rows = read_metadata(dataset_dir / "image_metadata.csv")
    harvey = {r["image_id"]: r for r in rows if r["event_id"] == HARVEY_EVENT_ID and r["paired_image_id"]}
    pairs: dict[str, ImagePair] = {}
    for row in harvey.values():
        other = harvey.get(row["paired_image_id"])
        if other is None or row["pair_id"] in pairs:
            continue
        before, after = (row, other) if row["image_role"] == "before" else (other, row)
        pairs[row["pair_id"]] = ImagePair(row["pair_id"], _image(before, dataset_dir, cache_dir), _image(after, dataset_dir, cache_dir))
    return tuple(sorted(pairs.values(), key=lambda p: p.pair_id))


def best_pair_for(lat: float, lon: float, pairs: tuple[ImagePair, ...]) -> ImagePair | None:
    """The Image Pair whose footprint holds the point nearest its centre, or None (Unrated)."""
    candidates = [p for p in pairs if p.contains(lat, lon)]
    if not candidates:
        return None
    return min(candidates, key=lambda p: p.centre_offset(lat, lon))


def in_covered_area(lat: float, lon: float, pairs: tuple[ImagePair, ...]) -> bool:
    return any(p.contains(lat, lon) for p in pairs)
