"""Address index for the search bar's autosuggest.

Every OpenStreetMap address (addr:housenumber + addr:street) and every Intersection that lies
inside the Harvey imagery footprints, in a local SQLite FTS5 index. Suggestions never leave the
Covered Area and no lookup is sent to an outside service while typing.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable

from .config import Settings
from .dataset import best_pair_for, load_pairs
from .osm import FOOTPRINT_PAD_DEG, fetch_overpass
from .store import Store

_TOKEN = re.compile(r"[0-9a-z]+")


def build_address_query(bboxes) -> str:
    parts = "".join(f'nwr["addr:housenumber"]["addr:street"]({s:.6f},{w:.6f},{n:.6f},{e:.6f});' for s, w, n, e in bboxes)
    return f"[out:json][timeout:300];({parts});out center tags qt;"


def address_label(tags: dict) -> str | None:
    number, street = (tags.get("addr:housenumber") or "").strip(), (tags.get("addr:street") or "").strip()
    if not number or not street:
        return None
    city = (tags.get("addr:city") or "").strip()
    postcode = (tags.get("addr:postcode") or "").strip()
    tail = ", ".join(x for x in (city, postcode) if x)
    return f"{number} {street}" + (f", {tail}" if tail else "")


def extract_addresses(osm: dict) -> list[dict]:
    out = []
    for el in osm.get("elements", []):
        label = address_label(el.get("tags", {}))
        lat = el.get("lat", (el.get("center") or {}).get("lat"))
        lon = el.get("lon", (el.get("center") or {}).get("lon"))
        if label and lat is not None and lon is not None:
            out.append({"id": f"osm-{el['type'][0]}{el['id']}", "label": label, "lat": lat, "lon": lon})
    return out


CHUNK_PAIRS = 15  # small Overpass queries: the public servers time out on whole localities


def run_fetch_addresses(settings: Settings, store: Store, progress: Callable[[str], None] = print) -> int:
    """Resumable: each chunk is cached, and the index is rebuilt after every chunk, so a run cut
    short still leaves the addresses fetched so far searchable. Failed chunks are skipped; rerun to fill them."""
    pairs = load_pairs(settings.dataset_dir)
    ordered = sorted(pairs, key=lambda p: (p.locality, p.pair_id))
    chunks = [ordered[i:i + CHUNK_PAIRS] for i in range(0, len(ordered), CHUNK_PAIRS)]
    found: dict[str, dict] = {}
    failed = 0
    for n, group in enumerate(chunks, 1):
        bboxes = [(p.after.south - FOOTPRINT_PAD_DEG, p.after.west - FOOTPRINT_PAD_DEG,
                   p.after.north + FOOTPRINT_PAD_DEG, p.after.east + FOOTPRINT_PAD_DEG) for p in group]
        try:
            osm = fetch_overpass(build_address_query(bboxes), settings, settings.data_dir / "cache", progress)
        except RuntimeError as e:
            failed += 1
            progress(f"  chunk {n}/{len(chunks)} skipped: {e}")
            continue
        for a in extract_addresses(osm):
            pair = best_pair_for(a["lat"], a["lon"], pairs)
            if pair is not None:  # strictly inside a footprint
                found.setdefault(a["id"], {**a, "pair_id": pair.pair_id})
        store.replace_addresses(list(found.values()))
        progress(f"Addresses: chunk {n}/{len(chunks)} done, {len(found)} inside the Covered Area so far")
    progress(f"Addresses: {len(found)} indexed; {failed} chunks failed (rerun to retry them)")
    return len(found)


def suggest(store: Store, q: str, limit: int = 8) -> list[dict]:
    """Prefix match on every word typed: '1234 bar' -> '1234 Barker Cypress Rd'."""
    tokens = _TOKEN.findall((q or "").lower())
    if not tokens or len("".join(tokens)) < 2:
        return []
    match = " ".join(f'"{t}"*' for t in tokens[:8])
    rows = store.query(
        "SELECT places.label, kind, lat, lon, pair_id FROM place_fts JOIN places ON places.rowid = place_fts.rowid"
        " WHERE place_fts MATCH ? ORDER BY bm25(place_fts), length(places.label) LIMIT ?",
        (match, limit * 4))
    out, seen = [], set()
    for r in rows:  # an address can be on both a node and its building
        if r["label"] not in seen:
            seen.add(r["label"])
            out.append(dict(r))
    return out[:limit]
