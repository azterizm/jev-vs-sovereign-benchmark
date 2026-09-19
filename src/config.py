"""Configuration loader and schema definition."""
import os
import torch
import yaml
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field


class OpenRouterSettings(BaseModel):
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str = "https://openrouter.ai/api/alpha/decisions"
    model: str = "typesafe/jev-1.13"
    timeout_seconds: float = 30.0
    site_url: str = "https://memonsystems.com"
    site_name: str = "Sovereign Legal AI Benchmark"

    @property
    def api_key(self) -> Optional[str]:
        return os.getenv(self.api_key_env)


class BenchmarkSettings(BaseModel):
    warmup_iterations: int = 5
    measured_iterations: int = 20
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"


class ModelsSettings(BaseModel):
    intent_model: str = "distilbert-base-uncased"
    reranker_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    nli_fallback: str = "cross-encoder/nli-deberta-v3-small"


class PricingSettings(BaseModel):
    jev_input_cost_per_m_tokens: float = 0.042
    jev_output_cost_per_m_tokens: float = 0.042
    sovereign_marginal_cost_per_token: float = 0.000000


class AuditSettings(BaseModel):
    standard: str = "EU AI Act Article 15 (Accuracy & Deterministic Robustness)"
    jurisdiction: str = "UK Primary Legislation"
    target_provisions: list[str] = Field(default_factory=list)


class Settings(BaseModel):
    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    benchmark: BenchmarkSettings = Field(default_factory=BenchmarkSettings)
    models: ModelsSettings = Field(default_factory=ModelsSettings)
    pricing: PricingSettings = Field(default_factory=PricingSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)


def load_settings(config_path: Optional[str | Path] = None) -> Settings:
    """Loads settings from YAML file or defaults."""
    if config_path is None:
        # Check current working directory, then repo root
        candidates = [
            Path("config.yaml"),
            Path(__file__).resolve().parent.parent / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.is_file():
                config_path = candidate
                break

    if config_path and Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return Settings(**data)
    return Settings()
