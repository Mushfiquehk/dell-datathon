"""Intersections and road names from OpenStreetMap (Overpass API).

An Intersection is an OSM node shared by drivable ways that carry at least two
different road names (or numbers). Nodes for the same road pair within MERGE_M
metres (dual carriageways, wide junctions) are merged into one Intersection.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

from .config import Settings
from .dataset import ImagePair, best_pair_for, load_pairs
from .geo import haversine_m
from .store import Store

log = logging.getLogger(__name__)

DRIVABLE = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "living_street", "road")
HIGHWAY_REGEX = "^(" + "|".join(DRIVABLE) + ")(_link)?$"
MERGE_M = 60.0
FOOTPRINT_PAD_DEG = 0.001  # ~100 m, so roads crossing the image edge still meet


def road_label(tags: dict) -> str | None:
    name, ref = (tags.get("name") or "").strip(), (tags.get("ref") or "").strip()
    if name and ref and ref not in name:
        return f"{name} ({ref})"
    return name or ref or None


def road_identity(tags: dict) -> str | None:
    """What makes two ways 'the same road': the name, else the route number."""
    return ((tags.get("name") or tags.get("ref") or "").strip().lower()) or None


def build_query(bboxes: Iterable[tuple[float, float, float, float]]) -> str:
    parts = "".join(f'way["highway"~"{HIGHWAY_REGEX}"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});' for s, w, n, e in bboxes)
    return f"[out:json][timeout:300];({parts});out body;>;out skel qt;"


def fetch_overpass(query: str, settings: Settings, cache_dir: Path, progress: Callable[[str], None] = print) -> dict:
    key = hashlib.sha1(query.encode()).hexdigest()[:16]
    cache = cache_dir / f"overpass_{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    body = urllib.parse.urlencode({"data": query}).encode()
    delay = 5.0
    last: Exception | None = None
    for attempt in range(4):
        for url in settings.overpass_urls:
            req = urllib.request.Request(url, data=body, headers={"User-Agent": settings.user_agent})
            try:
                with urllib.request.urlopen(req, timeout=360) as resp:
                    data = json.loads(resp.read().decode())
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(data))
                return data
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ConnectionError) as e:
                last = e
                progress(f"  overpass {url} failed ({e}); trying next")
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"All Overpass endpoints failed: {last}")


def extract_intersections(osm: dict) -> list[dict]:
    """Turn an Overpass 'out body; >; out skel' response into Intersections."""
    nodes = {el["id"]: (el["lat"], el["lon"]) for el in osm.get("elements", []) if el["type"] == "node"}
    node_roads: dict[int, dict[str, str]] = defaultdict(dict)  # node -> {identity: label}
    for el in osm.get("elements", []):
        if el["type"] != "way":
            continue
        tags = el.get("tags", {})
        if tags.get("area") == "yes":
            continue
        ident, label = road_identity(tags), road_label(tags)
        if not ident or not label:
            continue  # unnamed roads cannot name an Intersection
        for nid in el.get("nodes", []):
            node_roads[nid].setdefault(ident, label)

    raw = []
    for nid, roads in node_roads.items():
        if len(roads) >= 2 and nid in nodes:
            labels = sorted(set(roads.values()), key=str.lower)
            raw.append({"node": nid, "key": tuple(sorted(roads)), "roads": labels, "lat": nodes[nid][0], "lon": nodes[nid][1]})

    # Merge nodes of the same road combination that sit close together.
    merged: list[dict] = []
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in sorted(raw, key=lambda r: r["node"]):
        for m in by_key[r["key"]]:
            if haversine_m(m["lat"], m["lon"], r["lat"], r["lon"]) <= MERGE_M:
                m["osm_node_ids"].append(r["node"])
                n = len(m["osm_node_ids"])
                m["lat"] += (r["lat"] - m["lat"]) / n
                m["lon"] += (r["lon"] - m["lon"]) / n
                break
        else:
            item = {"key": r["key"], "roads": r["roads"], "lat": r["lat"], "lon": r["lon"], "osm_node_ids": [r["node"]]}
            by_key[r["key"]].append(item)
            merged.append(item)
    for m in merged:
        m["id"] = f"osm-{min(m['osm_node_ids'])}"
        m["name"] = " & ".join(m["roads"])
        del m["key"]
    return merged


def run_fetch_intersections(settings: Settings, store: Store, progress: Callable[[str], None] = print) -> tuple[int, int]:
    pairs = load_pairs(settings.dataset_dir)
    by_locality: dict[str, list[ImagePair]] = defaultdict(list)
    for p in pairs:
        by_locality[p.locality].append(p)

    found: dict[str, dict] = {}
    for locality, group in sorted(by_locality.items()):
        bboxes = [(p.after.south - FOOTPRINT_PAD_DEG, p.after.west - FOOTPRINT_PAD_DEG,
                   p.after.north + FOOTPRINT_PAD_DEG, p.after.east + FOOTPRINT_PAD_DEG) for p in group]
        progress(f"OSM: {locality} ({len(group)} pairs)")
        osm = fetch_overpass(build_query(bboxes), settings, settings.data_dir / "cache")
        for item in extract_intersections(osm):
            found.setdefault(item["id"], item)

    rows = []
    for item in found.values():
        pair = best_pair_for(item["lat"], item["lon"], pairs)
        item["pair_id"] = pair.pair_id if pair else None  # None = Unrated
        rows.append(item)
    store.replace_intersections(rows)
    rated = sum(1 for r in rows if r["pair_id"])
    progress(f"OSM: {len(rows)} intersections, {rated} inside the Covered Area, {len(rows) - rated} Unrated")
    return len(rows), rated
