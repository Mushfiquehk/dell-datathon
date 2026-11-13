Here's the full picture as it stands this morning:
Teams work on **Dell Pro Max GB10** workstations — Grace Blackwell chip, 128 GB unified memory at 273 GB/s, DGX OS. Huge models fit; MoE models run fast, dense models run slow. Same silicon as NVIDIA's DGX Spark, so Spark playbooks apply.
**model shortlist** 
- **Qwen3.6-35B-A3B** — default workhorse, ~97 tok/s, also NVIDIA's own default for these machines.
- **gpt-oss-120b / 20b** — deep reasoning / fast second seat.
- **Nemotron 3.5 Super VL** — vision flagship (Nov 2026 release); Lightning and Nano for speed.
- **Flint** — build copilot; **Qwen3-Embedding-4B** — RAG retrieval.
- Rule noted: don't run all the big ones at once (~113 GB is too tight), and pull models early before venue Wi-Fi saturates.
**Your Project**
- Using the UTSA Flood Image dataset by Jennifer Fowler on HuggingFace
- Only the 575 Hurricane Harvey before/after pairs are used. The 50 unpaired Hill Country images and the 4 Coast Guard video frames are out of scope.
- Goal: give each Intersection inside the imagery's footprint a Low / Moderate / High Susceptibility Rating, so a Flash Flood Warning can be turned into a list of the specific intersections most likely to flood. Terms are defined in `CONTEXT.md`.
- Intersections outside the imagery's footprint are Unrated. No extrapolation, and nothing specific to Bexar County or UTSA.
- Local testing uses Ollama with `qwen3-vl:8b`. The model name is a setting so the same code can point at a different model on the GB10.
- Two passes with the same model. The image pass fills the dataset's blank label fields (`flood_visible`, `flood_aftermath_visible`, `annotator`, `annotation_notes`) for the 1,150 Harvey images only. The intersection pass records an Intersection Observation per Intersection, and the Susceptibility Rating is read directly from it.
- The filled-in labels stay in this project's own copy of the metadata. Nothing is sent back to the public dataset repo.
- An Alert is written for one location a resident supplies. It names the High and Moderate Intersections within 1.5 km, with the text composed by the language model. Low and Undetermined Intersections appear on the map only. Alerts are shown on screen next to the map; nothing is actually sent.
- Each Intersection is judged from one 150 m square crop centred on it, taken from both images of the pair. "On the roadway" means the road surface at the centre of the crop; "close by" means anywhere else in the crop (within about 75 m). A crop clipped by the image edge is still used. An Intersection inside two overlapping images is rated from the one where it sits nearest the centre.
- The FEMA High-Risk Zone answer is the image's own `fema_sfha_tf` value from the dataset, applied to every Intersection in that image. A blank value means "not known to be outside", so those Intersections can be Moderate but never Low.
- Verification is a spot check by eye: about 50 Intersections viewed side by side (before crop, after crop, the model's three answers) and marked right or wrong. It is a spot check, not an accuracy figure, because the reviewer sees the model's answer.
- Intersections and road names come from OpenStreetMap (the `observations.sqlite` described in the dataset README is not in the HuggingFace repo).
- Harvey Flash Flood Warning polygons come from the Iowa Environmental Mesonet archive of NWS storm-based warnings: `https://mesonet.agron.iastate.edu/api/1/vtec/sbw_interval.geojson?begints=2017-08-25T00:00:00Z&endts=2017-09-04T00:00:00Z&wfo=HGX&only_new=true&ph=FF` (use `wfo=LCH` for the Bevil Oaks tiles).
