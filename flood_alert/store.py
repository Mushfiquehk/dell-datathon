"""SQLite store for everything this project produces. Nothing is written back to the dataset."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS image_labels (
    image_id TEXT PRIMARY KEY,
    flood_visible TEXT NOT NULL,            -- yes / no / uncertain  (New Water visible)
    flood_aftermath_visible TEXT NOT NULL,  -- yes / no / uncertain  (Flood Residue visible)
    annotator TEXT NOT NULL,
    annotation_notes TEXT NOT NULL,
    raw TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS intersections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,                     -- e.g. "Barker Cypress Rd & Clay Rd"
    roads TEXT NOT NULL,                    -- JSON list of road names
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    pair_id TEXT,                           -- NULL = Unrated (outside Covered Area)
    osm_node_ids TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS intersections_pair ON intersections(pair_id);
CREATE TABLE IF NOT EXISTS observations (
    intersection_id TEXT PRIMARY KEY REFERENCES intersections(id),
    pair_id TEXT NOT NULL,
    roadway_visible INTEGER NOT NULL,
    new_water_on_roadway INTEGER NOT NULL,
    new_water_or_residue_close_by INTEGER NOT NULL,
    notes TEXT NOT NULL,
    model TEXT NOT NULL,
    raw TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS warnings (
    id TEXT PRIMARY KEY,
    wfo TEXT NOT NULL,
    issued TEXT,
    expires TEXT,
    properties TEXT NOT NULL,
    geometry TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS places (              -- search-bar suggestions, inside the Covered Area only
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,                     -- address / intersection
    label TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    pair_id TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS place_fts USING fts5(label, content='places', content_rowid='rowid', tokenize='unicode61');
CREATE TABLE IF NOT EXISTS reviews (
    intersection_id TEXT PRIMARY KEY REFERENCES intersections(id),
    verdict TEXT NOT NULL,                  -- right / wrong
    comment TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def execute(self, sql: str, params: tuple | dict = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def query(self, sql: str, params: tuple | dict = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.conn.execute(sql, params).fetchall()

    # --- image labels -----------------------------------------------------
    def save_image_label(self, image_id: str, label: dict, annotator: str, raw: str) -> None:
        self.execute(
            "INSERT OR REPLACE INTO image_labels (image_id, flood_visible, flood_aftermath_visible, annotator, annotation_notes, raw)"
            " VALUES (?,?,?,?,?,?)",
            (image_id, label["flood_visible"], label["flood_aftermath_visible"], annotator, label["notes"], raw),
        )

    def labelled_image_ids(self) -> set[str]:
        return {r["image_id"] for r in self.query("SELECT image_id FROM image_labels")}

    def image_labels(self) -> dict[str, sqlite3.Row]:
        return {r["image_id"]: r for r in self.query("SELECT * FROM image_labels")}

    # --- intersections ----------------------------------------------------
    def replace_intersections(self, rows: list[dict]) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM intersections")
            self.conn.executemany(
                "INSERT INTO intersections (id, name, roads, lat, lon, pair_id, osm_node_ids) VALUES (?,?,?,?,?,?,?)",
                [(r["id"], r["name"], json.dumps(r["roads"]), r["lat"], r["lon"], r["pair_id"], json.dumps(r["osm_node_ids"])) for r in rows],
            )
            self.conn.commit()

    def intersections(self, rated_only: bool = False) -> list[sqlite3.Row]:
        sql = "SELECT * FROM intersections"
        if rated_only:
            sql += " WHERE pair_id IS NOT NULL"
        return self.query(sql + " ORDER BY id")

    def intersection(self, intersection_id: str) -> sqlite3.Row | None:
        rows = self.query("SELECT * FROM intersections WHERE id = ?", (intersection_id,))
        return rows[0] if rows else None

    # --- search-bar places -------------------------------------------------
    def replace_addresses(self, rows: list[dict]) -> None:
        """Addresses from OSM plus every rated Intersection, re-indexed for prefix search."""
        inters = [(r["id"], "intersection", r["name"], r["lat"], r["lon"], r["pair_id"])
                  for r in self.intersections(rated_only=True)]
        with self._lock:
            self.conn.execute("DELETE FROM places")
            self.conn.executemany("INSERT OR REPLACE INTO places (id, kind, label, lat, lon, pair_id) VALUES (?,?,?,?,?,?)",
                                  [(r["id"], "address", r["label"], r["lat"], r["lon"], r["pair_id"]) for r in rows] + inters)
            self.conn.execute("INSERT INTO place_fts(place_fts) VALUES ('rebuild')")
            self.conn.commit()

    # --- observations -----------------------------------------------------
    def save_observation(self, intersection_id: str, pair_id: str, obs: dict, model: str, raw: str) -> None:
        self.execute(
            "INSERT OR REPLACE INTO observations (intersection_id, pair_id, roadway_visible, new_water_on_roadway,"
            " new_water_or_residue_close_by, notes, model, raw) VALUES (?,?,?,?,?,?,?,?)",
            (intersection_id, pair_id, int(obs["roadway_visible"]), int(obs["new_water_on_roadway"]),
             int(obs["new_water_or_residue_close_by"]), obs.get("notes", ""), model, raw),
        )

    def observed_ids(self) -> set[str]:
        return {r["intersection_id"] for r in self.query("SELECT intersection_id FROM observations")}

    def observations(self) -> dict[str, sqlite3.Row]:
        return {r["intersection_id"]: r for r in self.query("SELECT * FROM observations")}

    # --- warnings ---------------------------------------------------------
    def replace_warnings(self, rows: list[dict]) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM warnings")
            self.conn.executemany(
                "INSERT OR REPLACE INTO warnings (id, wfo, issued, expires, properties, geometry) VALUES (?,?,?,?,?,?)",
                [(r["id"], r["wfo"], r["issued"], r["expires"], json.dumps(r["properties"]), json.dumps(r["geometry"])) for r in rows],
            )
            self.conn.commit()

    def warnings(self) -> list[dict]:
        out = []
        for r in self.query("SELECT * FROM warnings ORDER BY issued"):
            out.append({"id": r["id"], "wfo": r["wfo"], "issued": r["issued"], "expires": r["expires"],
                        "properties": json.loads(r["properties"]), "geometry": json.loads(r["geometry"])})
        return out

    # --- reviews ----------------------------------------------------------
    def save_review(self, intersection_id: str, verdict: str, comment: str = "") -> None:
        self.execute("INSERT OR REPLACE INTO reviews (intersection_id, verdict, comment) VALUES (?,?,?)",
                     (intersection_id, verdict, comment))

    def reviews(self) -> dict[str, sqlite3.Row]:
        return {r["intersection_id"]: r for r in self.query("SELECT * FROM reviews")}
