import json
from pathlib import Path

import pytest

from flood_alert.config import Settings
from flood_alert.dataset import best_pair_for, load_pairs
from flood_alert.geo import haversine_m, point_in_geojson
from flood_alert.osm import extract_intersections, road_label
from flood_alert.rating import Rating, rate
from flood_alert.vlm import VLMError, extract_json, yes_no

DATASET = Path(__file__).resolve().parent.parent / "UTSA-flood-images"
needs_dataset = pytest.mark.skipif(not (DATASET / "image_metadata.csv").exists(), reason="dataset not present")


# --- rating -----------------------------------------------------------------
@pytest.mark.parametrize("visible,on_road,close_by,fema,expected", [
    (False, True, True, "T", Rating.UNDETERMINED),   # can't see the road -> no judgement
    (True, True, False, "F", Rating.HIGH),
    (True, True, True, "", Rating.HIGH),
    (True, False, True, "F", Rating.MODERATE),       # evidence close by
    (True, False, False, "T", Rating.MODERATE),      # FEMA High-Risk Zone
    (True, False, False, "F", Rating.LOW),
    (True, False, False, "", Rating.MODERATE),       # blank FEMA can never be Low
    (True, False, False, " f ", Rating.LOW),
])
def test_rating_rules(visible, on_road, close_by, fema, expected):
    assert rate(visible, on_road, close_by, fema) == expected


# --- geometry -----------------------------------------------------------------
def test_point_in_multipolygon_with_hole():
    square = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    hole = [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]]
    geom = {"type": "MultiPolygon", "coordinates": [[square, hole]]}
    assert point_in_geojson(lat=2, lon=2, geometry=geom)
    assert not point_in_geojson(lat=5, lon=5, geometry=geom)
    assert not point_in_geojson(lat=20, lon=2, geometry=geom)


def test_haversine_one_degree_lat():
    assert abs(haversine_m(29, -95, 30, -95) - 111_195) < 200


# --- model output parsing ---------------------------------------------------
def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('<think>hmm {"x":2}</think>\n```json\n{"a": "b}"}\n```') == {"a": "b}"}
    assert extract_json('reasoning...</think>{"ok": true}') == {"ok": True}
    with pytest.raises(VLMError):
        extract_json("no json here")
    assert yes_no(True) == "yes" and yes_no("No") == "no" and yes_no("maybe") == "uncertain"


# --- OSM intersections --------------------------------------------------------
def test_extract_intersections_merges_and_skips_same_road():
    osm = {"elements": [
        {"type": "way", "id": 1, "nodes": [1, 2, 3], "tags": {"highway": "residential", "name": "Main St"}},
        {"type": "way", "id": 2, "nodes": [3, 4], "tags": {"highway": "residential", "name": "Main St"}},  # same road split
        {"type": "way", "id": 3, "nodes": [2, 5], "tags": {"highway": "primary", "name": "Clay Rd", "ref": "FM 529"}},
        {"type": "way", "id": 4, "nodes": [6, 5], "tags": {"highway": "primary", "name": "Clay Rd"}},
        {"type": "way", "id": 5, "nodes": [3, 7], "tags": {"highway": "residential"}},  # unnamed
        # dual carriageway: second Main St / Clay Rd node 20 m away -> merged
        {"type": "way", "id": 6, "nodes": [8, 9], "tags": {"highway": "residential", "name": "Main St"}},
        {"type": "way", "id": 7, "nodes": [9, 10], "tags": {"highway": "primary", "name": "Clay Rd", "ref": "FM 529"}},
        {"type": "node", "id": 1, "lat": 29.0, "lon": -95.0}, {"type": "node", "id": 2, "lat": 29.001, "lon": -95.0},
        {"type": "node", "id": 3, "lat": 29.002, "lon": -95.0}, {"type": "node", "id": 4, "lat": 29.003, "lon": -95.0},
        {"type": "node", "id": 5, "lat": 29.001, "lon": -95.001}, {"type": "node", "id": 6, "lat": 29.001, "lon": -95.002},
        {"type": "node", "id": 7, "lat": 29.002, "lon": -95.001}, {"type": "node", "id": 8, "lat": 29.0012, "lon": -94.9},
        {"type": "node", "id": 9, "lat": 29.00118, "lon": -95.0}, {"type": "node", "id": 10, "lat": 29.002, "lon": -95.1},
    ]}
    found = extract_intersections(osm)
    assert len(found) == 1
    assert found[0]["name"] == "Clay Rd (FM 529) & Main St"
    assert sorted(found[0]["osm_node_ids"]) == [2, 9]
    assert road_label({"ref": "SH 6"}) == "SH 6"


# --- dataset / crops ------------------------------------------------------------
@needs_dataset
def test_harvey_pairs_and_pixels():
    pairs = load_pairs(DATASET)
    assert len(pairs) == 575
    p = pairs[0]
    a = p.after
    x, y = a.to_pixel(a.north, a.west)
    assert abs(x) < 1e-6 and abs(y) < 1e-6
    x, y = a.to_pixel(a.south, a.east)
    assert abs(x - a.width_px) < 1e-6 and abs(y - a.height_px) < 1e-6
    assert 270 < a.metres_to_pixels(150) < 310  # ~0.52 m/px
    lat, lon = a.center
    assert best_pair_for(lat, lon, pairs).pair_id == p.pair_id
    assert best_pair_for(0.0, 0.0, pairs) is None


@needs_dataset
def test_crop_clipped_at_edge_still_used():
    from flood_alert.intersection_pass import crop_around

    a = load_pairs(DATASET)[0].after
    full = crop_around(a, *a.center, size_m=150, out_px=448)
    assert abs(full.width - 448) <= 2 and abs(full.height - 448) <= 2
    corner = crop_around(a, a.north, a.west, size_m=150, out_px=448)
    assert 200 < corner.width < 260 and 200 < corner.height < 260  # quarter of the square survives


# --- alerts -------------------------------------------------------------------
@needs_dataset
def test_build_alert_paths(tmp_path):
    from flood_alert.alerts import build_alert
    from flood_alert.store import Store

    settings = Settings(dataset_dir=DATASET, data_dir=tmp_path)
    store = Store(":memory:")
    pair = load_pairs(DATASET)[0]
    lat, lon = pair.after.center
    big = {"type": "Polygon", "coordinates": [[[lon - 1, lat - 1], [lon + 1, lat - 1], [lon + 1, lat + 1], [lon - 1, lat + 1], [lon - 1, lat - 1]]]}
    store.replace_warnings([{"id": "W1", "wfo": "HGX", "issued": "2017-08-27T00:00:00Z", "expires": "2017-08-27T06:00:00Z",
                             "properties": {"locations": "Harris [TX]"}, "geometry": big}])
    store.replace_intersections([
        {"id": "a", "name": "A St & B St", "roads": ["A St", "B St"], "lat": lat, "lon": lon, "pair_id": pair.pair_id, "osm_node_ids": [1]},
        {"id": "b", "name": "C St & D St", "roads": ["C St", "D St"], "lat": lat + 0.0005, "lon": lon, "pair_id": pair.pair_id, "osm_node_ids": [2]},
        {"id": "c", "name": "E St & F St", "roads": ["E St", "F St"], "lat": lat + 0.001, "lon": lon, "pair_id": pair.pair_id, "osm_node_ids": [3]},
    ])
    store.save_observation("a", pair.pair_id, {"roadway_visible": True, "new_water_on_roadway": True, "new_water_or_residue_close_by": True}, "m", "")
    store.save_observation("b", pair.pair_id, {"roadway_visible": False, "new_water_on_roadway": False, "new_water_or_residue_close_by": False}, "m", "")

    result = build_alert(settings, store, lat, lon)
    assert result["status"] == "alert" and result["in_covered_area"]
    names = [i["name"] for i in result["named_intersections"]]
    assert names[0] == "A St & B St"            # High first
    assert "C St & D St" not in names           # Undetermined is map-only
    assert "A St & B St" in result["text"]

    outside = build_alert(settings, store, lat + 0.5, lon + 0.5)
    assert outside["status"] == "alert" and not outside["in_covered_area"]
    assert "no intersection ratings" in outside["text"].lower()

    none = build_alert(settings, store, lat + 5, lon)
    assert none["status"] == "no_warning"

    class FakeClient:
        model = "fake"

        def ask_text(self, prompt, system=None, max_tokens=None):
            assert "A St & B St" in prompt
            return "Flash flood warning. Avoid the area."  # forgets to name the intersection

    composed = build_alert(settings, store, lat, lon, client=FakeClient())
    assert composed["composed_by"] == "model:fake"
    assert "A St & B St" in composed["text"]    # appended because the model dropped it


# --- comparison logic (model sees one image at a time; code compares) -----------
def _seen(surface=None, water=False, mud=False):
    from flood_alert.perception import Seen
    return Seen(description="d", standing_water=water, mud_or_debris=mud, ring_surface=surface)


def test_to_observation_rules():
    from flood_alert.intersection_pass import to_observation

    o = to_observation(_seen("paved_road"), _seen("standing_water"))
    assert o["roadway_visible"] and o["new_water_on_roadway"]
    o = to_observation(_seen("paved_road"), _seen("trees", water=True))
    assert not o["roadway_visible"] and not o["new_water_on_roadway"]          # -> Undetermined
    o = to_observation(_seen("trees"), _seen("paved_road"))
    assert not o["roadway_visible"]
    o = to_observation(_seen("paved_road", water=True), _seen("paved_road", water=True))
    assert o["roadway_visible"] and not o["new_water_or_residue_close_by"]     # water was already there
    o = to_observation(_seen("paved_road"), _seen("paved_road", water=True))
    assert o["new_water_or_residue_close_by"]
    o = to_observation(_seen("paved_road"), _seen("mud"))
    assert o["roadway_visible"] and not o["new_water_on_roadway"] and o["new_water_or_residue_close_by"]


def test_labels_for_pair():
    from flood_alert.image_pass import labels_for_pair

    b, a = labels_for_pair(_seen(water=False), _seen(water=True, mud=True))
    assert (b["flood_visible"], a["flood_visible"], a["flood_aftermath_visible"]) == ("no", "yes", "yes")
    b, a = labels_for_pair(_seen(water=True), _seen(water=True))
    assert a["flood_visible"] == "uncertain"
    b, a = labels_for_pair(_seen(), _seen())
    assert a["flood_visible"] == "no" and a["flood_aftermath_visible"] == "no"
