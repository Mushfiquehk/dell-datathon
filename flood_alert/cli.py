"""Command line: python -m flood_alert <command>

  check                 confirm the model server answers and can see images
  fetch-images          download Harvey images if the dataset clone has only LFS pointers
  label-images          image pass (fills Image Labels for the 1,150 Harvey images)
  export-labels         write data/image_metadata_labelled.csv (project copy)
  fetch-intersections   pull Intersections from OpenStreetMap
  fetch-warnings        pull Harvey Flash Flood Warning polygons (IEM)
  observe               intersection pass (Intersection Observations)
  summary               counts per Susceptibility Rating
  alert LAT LON         print the Alert for one Alert Location
  serve                 map + alerts + review page
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from .config import get_settings
from .store import Store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="flood_alert", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    p = sub.add_parser("fetch-images")
    p.add_argument("--pair", action="append", dest="pair_ids", help="only these Image Pairs (repeatable)")
    p = sub.add_parser("label-images")
    p.add_argument("--limit", type=int)
    p.add_argument("--image", action="append", dest="image_ids")
    p.add_argument("--redo", action="store_true")
    sub.add_parser("export-labels")
    sub.add_parser("fetch-intersections")
    sub.add_parser("fetch-warnings")
    p = sub.add_parser("observe")
    p.add_argument("--limit", type=int)
    p.add_argument("--id", action="append", dest="ids")
    p.add_argument("--pair", action="append", dest="pair_ids")
    p.add_argument("--redo", action="store_true")
    sub.add_parser("summary")
    p = sub.add_parser("alert")
    p.add_argument("lat", type=float)
    p.add_argument("lon", type=float)
    p.add_argument("--warning")
    p.add_argument("--no-model", action="store_true", help="use the template instead of the language model")
    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5050)
    p.add_argument("--no-model", action="store_true")
    args = ap.parse_args(argv)

    settings = get_settings()
    store = Store(settings.db_path)

    if args.cmd == "check":
        from .dataset import load_pairs
        from .vlm import VLMClient

        pairs = load_pairs(settings.dataset_dir)
        print(f"Dataset: {len(pairs)} Harvey Image Pairs in {settings.dataset_dir}")
        client = VLMClient(settings)
        print(f"Model server: {settings.vlm_base_url}  model: {client.model}")
        answer, _ = client.ask_json("What is the dominant ground cover in this aerial image? Reply JSON {\"answer\": string}.",
                                    [pairs[0].after.path])
        print(f"Vision check OK: {answer}")
    elif args.cmd == "fetch-images":
        from .fetch_images import fetch_images

        fetch_images(settings, pair_ids=args.pair_ids)
    elif args.cmd == "label-images":
        from .image_pass import run_image_pass

        run_image_pass(settings, store, limit=args.limit, image_ids=args.image_ids, redo=args.redo)
    elif args.cmd == "export-labels":
        from .image_pass import export_labelled_metadata

        n = export_labelled_metadata(settings, store)
        print(f"Wrote {settings.labelled_metadata_csv} ({n} images labelled)")
    elif args.cmd == "fetch-intersections":
        from .osm import run_fetch_intersections

        run_fetch_intersections(settings, store)
    elif args.cmd == "fetch-warnings":
        from .nws_warnings import fetch_warnings

        fetch_warnings(settings, store)
    elif args.cmd == "observe":
        from .intersection_pass import run_intersection_pass

        run_intersection_pass(settings, store, limit=args.limit, ids=args.ids, pair_ids=args.pair_ids, redo=args.redo)
    elif args.cmd == "summary":
        from .alerts import intersection_ratings

        ratings = intersection_ratings(settings, store)
        labels = store.image_labels()
        print(f"Image Labels: {len(labels)}/1150  flood_visible={dict(Counter(l['flood_visible'] for l in labels.values()))}"
              f"  aftermath={dict(Counter(l['flood_aftermath_visible'] for l in labels.values()))}")
        print(f"Intersections: {len(ratings)}  " + json.dumps(dict(Counter(r['rating'] for r in ratings))))
        print(f"Warnings: {len(store.warnings())}   Reviews: {len(store.reviews())}")
    elif args.cmd == "alert":
        from .alerts import build_alert
        from .vlm import VLMClient

        client = None if args.no_model else VLMClient(settings, model=settings.alert_model or None)
        result = build_alert(settings, store, args.lat, args.lon, warning_id=args.warning, client=client)
        print(result["text"])
        for i in result.get("named_intersections", []):
            print(f"  - {i['name']}: {i['rating']} ({i['distance_km']:.2f} km {i['direction']})")
        print(f"[status={result['status']} covered={result['in_covered_area']} "
              f"warnings={len(result['covering_warnings'])} composed_by={result.get('composed_by')}]")
    elif args.cmd == "serve":
        from .web import create_app

        create_app(settings, store, use_model=not args.no_model).run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
