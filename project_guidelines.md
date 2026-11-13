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
- Initial goal is to label the data in terms of flood risk and annotate with road numbers to locate the roads at the highet risk
