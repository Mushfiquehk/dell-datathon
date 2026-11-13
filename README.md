# Flood Alert

Turns a broad Flash Flood Warning into a place-by-place picture of which Intersections inside the warned area are most likely to flood. Ratings come from the 575 Hurricane Harvey before/after Image Pairs in the [UTSA flood image dataset](https://huggingface.co/datasets/jennifer-fowler/UTSA-flood-images). Terms are defined in [`CONTEXT.md`](CONTEXT.md) and scope in [`project_guidelines.md`](project_guidelines.md).

## Pipeline

| Step | Command | Output |
|---|---|---|
| Images (only if the clone has Git LFS pointers) | `python -m flood_alert fetch-images` | `data/images/`, sha256-checked |
| Image pass: fill `flood_visible`, `flood_aftermath_visible`, `annotator`, `annotation_notes` for the 1,150 Harvey images | `python -m flood_alert label-images` | sqlite `image_labels` |
| Export the project's own labelled copy of the metadata | `python -m flood_alert export-labels` | `data/image_metadata_labelled.csv` |
| Intersections and road names from OpenStreetMap | `python -m flood_alert fetch-intersections` | sqlite `intersections` |
| Harvey Flash Flood Warning polygons (IEM, HGX + LCH) | `python -m flood_alert fetch-warnings` | sqlite `warnings` |
| Intersection pass: one Intersection Observation each | `python -m flood_alert observe` | sqlite `observations` |
| Counts | `python -m flood_alert summary` | |
| One Alert in the terminal | `python -m flood_alert alert 29.84 -95.64` | |
| Map, on-screen Alerts and spot-check page | `python -m flood_alert serve` → http://127.0.0.1:5050 (review at `/review`) | |

You can stop and restart both model passes at any point: they skip work that is already done. Use `--redo` to run them again, or `--limit`, `--image`, `--pair` and `--id` to run them on a subset.

## Rating rule (`flood_alert/rating.py`)

| Intersection Observation | FEMA `fema_sfha_tf` | Susceptibility Rating |
|---|---|---|
| Roadway not visible in one or both crops | any | Undetermined |
| New Water on the roadway | any | High |
| New Water or Flood Residue close by | any | Moderate |
| No evidence | `T` | Moderate |
| No evidence | `F` | Low |
| No evidence | blank | Moderate (not known to be outside the zone, so it can never be Low) |

An Intersection outside every image footprint is **Unrated**. If an Intersection appears in two overlapping images, it is rated from the one where it sits nearest the centre. Each Intersection is judged from a 150 m crop of both images, with a red ring marking the Intersection. A crop that the image edge cuts off is still used.

## Model settings

Settings are environment variables, and every model call goes through the OpenAI-compatible API, so vLLM, GPUStack and Ollama all work:

```bash
# GB10 (vLLM behind GPUStack)
export VLM_BASE_URL=http://10.246.100.230/v1
export VLM_MODEL=qwen3-vl-8b-instruct
export VLM_API_KEY=...            # do not commit
export VLM_CONCURRENCY=8          # parallel requests; vLLM batches them

# or a local vLLM server: VLM_BASE_URL=http://localhost:8000/v1 (VLM_MODEL blank = first served model)
# or Ollama: VLM_BASE_URL=http://localhost:11434/v1 VLM_MODEL=qwen3-vl:8b VLM_CONCURRENCY=1
```

Other settings: `ALERT_MODEL` (the model that writes Alert text; defaults to `VLM_MODEL`), `VLM_JSON_SCHEMA=0` (turns off guided JSON output if a server rejects it), `NEARBY_KM` (default 1.5), `CROP_SIZE_M` (default 150), `FLOOD_DATASET_DIR` and `FLOOD_DATA_DIR`.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ln -s /path/to/UTSA-flood-images UTSA-flood-images   # or set FLOOD_DATASET_DIR
.venv/bin/python -m pytest -q
```

## Notes

- Nothing is written back to the dataset. Labels live in `data/flood_alert.sqlite` and in the exported CSV copy.
- An Alert names the High and Moderate Intersections within 1.5 km of the Alert Location. Low and Undetermined Intersections appear on the map only. If the language model leaves out an Intersection it should have named, the code appends it to the text. If the model cannot be reached, the Alert falls back to a template.
- Address search sends the typed address to OpenStreetMap Nominatim. Dropping a map pin sends nothing.
- The review page shows about 50 observed Intersections spread across the ratings. It is a spot check and does not give an accuracy figure.
