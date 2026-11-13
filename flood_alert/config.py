"""Settings, all overridable with environment variables.

The model is a setting so the same code runs against a local vLLM / Ollama
server or the GB10's vLLM server. Both expose the OpenAI-compatible API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    # Dataset (local clone of the HuggingFace UTSA flood image repo).
    dataset_dir: Path = field(default_factory=lambda: Path(_env("FLOOD_DATASET_DIR", str(REPO_ROOT / "UTSA-flood-images"))))
    # Everything this project produces (sqlite, caches, labelled metadata copy).
    data_dir: Path = field(default_factory=lambda: Path(_env("FLOOD_DATA_DIR", str(REPO_ROOT / "data"))))

    # Vision-language model, OpenAI-compatible endpoint.
    # vLLM default: http://localhost:8000/v1   Ollama: http://localhost:11434/v1
    vlm_base_url: str = field(default_factory=lambda: _env("VLM_BASE_URL", "http://localhost:8000/v1"))
    # Blank means "ask the server which model it serves" (works for vLLM).
    vlm_model: str = field(default_factory=lambda: _env("VLM_MODEL", ""))
    vlm_api_key: str = field(default_factory=lambda: _env("VLM_API_KEY", "EMPTY"))
    vlm_timeout_s: float = field(default_factory=lambda: float(_env("VLM_TIMEOUT_S", "600")))
    vlm_max_tokens: int = field(default_factory=lambda: int(_env("VLM_MAX_TOKENS", "4096")))
    vlm_concurrency: int = field(default_factory=lambda: int(_env("VLM_CONCURRENCY", "4")))
    # Ask the server to constrain output to a JSON schema (vLLM guided decoding).
    vlm_json_schema: bool = field(default_factory=lambda: _env("VLM_JSON_SCHEMA", "1") == "1")
    # Language model that writes Alert text. Blank means the same model as the VLM.
    alert_model: str = field(default_factory=lambda: _env("ALERT_MODEL", ""))

    # Intersection pass geometry (see project_guidelines.md).
    crop_size_m: float = field(default_factory=lambda: float(_env("CROP_SIZE_M", "150")))
    crop_output_px: int = field(default_factory=lambda: int(_env("CROP_OUTPUT_PX", "448")))
    # Alerts.
    nearby_km: float = field(default_factory=lambda: float(_env("NEARBY_KM", "1.5")))

    # External data services.
    overpass_urls: tuple[str, ...] = field(default_factory=lambda: tuple(_env(
        "OVERPASS_URLS",
        "https://overpass-api.de/api/interpreter,"
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter,"
        "https://overpass.kumi.systems/api/interpreter",
    ).split(",")))
    nominatim_url: str = field(default_factory=lambda: _env("NOMINATIM_URL", "https://nominatim.openstreetmap.org/search"))
    user_agent: str = field(default_factory=lambda: _env("HTTP_USER_AGENT", "flood-alert-datathon/0.1"))

    @property
    def metadata_csv(self) -> Path:
        return self.dataset_dir / "image_metadata.csv"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "flood_alert.sqlite"

    @property
    def labelled_metadata_csv(self) -> Path:
        return self.data_dir / "image_metadata_labelled.csv"


def get_settings() -> Settings:
    return Settings()
