"""Harvey Flash Flood Warning polygons from the Iowa Environmental Mesonet archive."""
from __future__ import annotations

import json
import urllib.request
from typing import Callable

from .config import Settings
from .geo import point_in_geojson
from .store import Store

IEM_URL = ("https://mesonet.agron.iastate.edu/api/1/vtec/sbw_interval.geojson"
           "?begints=2017-08-25T00:00:00Z&endts=2017-09-04T00:00:00Z&wfo={wfo}&only_new=true&ph=FF")
# HGX = Houston/Galveston; LCH = Lake Charles (covers the Bevil Oaks tiles).
WFOS = ("HGX", "LCH")


def fetch_warnings(settings: Settings, store: Store, progress: Callable[[str], None] = print) -> int:
    rows = []
    for wfo in WFOS:
        req = urllib.request.Request(IEM_URL.format(wfo=wfo), headers={"User-Agent": settings.user_agent})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        cache = settings.data_dir / "cache" / f"iem_ffw_{wfo}.geojson"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data))
        for feat in data.get("features", []):
            p = feat["properties"]
            if p.get("phenomena") != "FF" or p.get("significance") != "W" or not feat.get("geometry"):
                continue
            rows.append({
                "id": p.get("product_id") or f"{wfo}-{p.get('eventid')}-{p.get('utc_issue')}",
                "wfo": p.get("wfo", wfo),
                "issued": p.get("utc_issue"),
                "expires": p.get("utc_expire"),
                "properties": p,
                "geometry": feat["geometry"],
            })
        progress(f"Warnings: {wfo}: {sum(1 for r in rows if r['wfo'] == wfo)} Flash Flood Warnings")
    store.replace_warnings(rows)
    return len(rows)


def warnings_covering(lat: float, lon: float, warnings: list[dict]) -> list[dict]:
    return [w for w in warnings if point_in_geojson(lat, lon, w["geometry"])]
