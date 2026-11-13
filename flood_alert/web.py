"""Flask app: the map, on-screen Alerts, and the spot-check review page."""
from __future__ import annotations

import io
import random
from functools import lru_cache

from flask import Flask, abort, jsonify, render_template, request, send_file

from .alerts import build_alert, geocode, intersection_ratings
from .config import Settings, get_settings
from .dataset import load_pairs
from .intersection_pass import crop_around
from .rating import Rating
from .store import Store
from .vlm import VLMClient

REVIEW_SAMPLE_SIZE = 50
REVIEW_SEED = 2017


def create_app(settings: Settings | None = None, store: Store | None = None, use_model: bool = True) -> Flask:
    settings = settings or get_settings()
    store = store or Store(settings.db_path)
    app = Flask(__name__)
    pairs = load_pairs(settings.dataset_dir)
    pairs_by_id = {p.pair_id: p for p in pairs}
    images_by_id = {img.image_id: img for p in pairs for img in (p.before, p.after)}

    def client() -> VLMClient | None:
        if not use_model:
            return None
        return VLMClient(settings, model=settings.alert_model or None)

    @app.get("/")
    def index():
        return render_template("index.html", nearby_km=settings.nearby_km)

    @app.get("/api/data")
    def data():
        return jsonify({
            "footprints": [{"pair_id": p.pair_id, "locality": p.locality, "fema_sfha_tf": p.fema_sfha_tf,
                            "before": p.before.image_id, "after": p.after.image_id,
                            "bounds": [[p.after.south, p.after.west], [p.after.north, p.after.east]]} for p in pairs],
            "intersections": intersection_ratings(settings, store),
            "warnings": [{"id": w["id"], "wfo": w["wfo"], "issued": w["issued"], "expires": w["expires"],
                          "counties": w["properties"].get("locations", ""), "geometry": w["geometry"]}
                         for w in store.warnings()],
        })

    @app.post("/api/alert")
    def alert():
        body = request.get_json(force=True) or {}
        label = None
        if body.get("address"):
            try:
                found = geocode(body["address"], settings)
            except OSError as e:
                return jsonify({"status": "error", "text": f"Address lookup failed: {e}"}), 502
            if not found:
                return jsonify({"status": "error", "text": "Address not found."}), 404
            lat, lon, label = found
        else:
            try:
                lat, lon = float(body["lat"]), float(body["lon"])
            except (KeyError, TypeError, ValueError):
                return jsonify({"status": "error", "text": "Give an address or lat/lon."}), 400
        return jsonify(build_alert(settings, store, lat, lon, warning_id=body.get("warning_id") or None,
                                   location_label=label, client=client()))

    @app.get("/image/<image_id>.png")
    def image(image_id: str):
        img = images_by_id.get(image_id)
        if img is None:
            abort(404)
        return send_file(img.path, mimetype="image/png")

    @app.get("/crop/<intersection_id>/<which>.png")
    def crop(intersection_id: str, which: str):
        rows = [r for r in store.intersections(rated_only=True) if r["id"] == intersection_id]
        if not rows or which not in ("before", "after"):
            abort(404)
        r = rows[0]
        pair = pairs_by_id[r["pair_id"]]
        img = crop_around(pair.before if which == "before" else pair.after, r["lat"], r["lon"],
                          settings.crop_size_m, settings.crop_output_px)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    @lru_cache(maxsize=1)
    def _sample_ids(observed_key: int) -> tuple[str, ...]:
        """~50 observed Intersections, spread across ratings so High ones are not crowded out."""
        rated = [i for i in intersection_ratings(settings, store) if "observation" in i]
        rng = random.Random(REVIEW_SEED)
        buckets: dict[str, list[dict]] = {}
        for i in rated:
            buckets.setdefault(i["rating"], []).append(i)
        for b in buckets.values():
            rng.shuffle(b)
        picked: list[str] = []
        while len(picked) < REVIEW_SAMPLE_SIZE and any(buckets.values()):
            for key in [Rating.HIGH.value, Rating.MODERATE.value, Rating.LOW.value, Rating.UNDETERMINED.value]:
                if buckets.get(key) and len(picked) < REVIEW_SAMPLE_SIZE:
                    picked.append(buckets[key].pop()["id"])
        return tuple(picked)

    @app.get("/review")
    def review():
        ids = _sample_ids(len(store.observed_ids()))
        by_id = {i["id"]: i for i in intersection_ratings(settings, store)}
        reviews = store.reviews()
        items = [{**by_id[i], "review": dict(reviews[i]) if i in reviews else None} for i in ids if i in by_id]
        right = sum(1 for i in items if i["review"] and i["review"]["verdict"] == "right")
        wrong = sum(1 for i in items if i["review"] and i["review"]["verdict"] == "wrong")
        return render_template("review.html", items=items, right=right, wrong=wrong)

    @app.post("/api/review")
    def save_review():
        body = request.get_json(force=True) or {}
        if body.get("verdict") not in ("right", "wrong") or not body.get("id"):
            return jsonify({"ok": False}), 400
        store.save_review(body["id"], body["verdict"], body.get("comment", ""))
        return jsonify({"ok": True})

    return app
