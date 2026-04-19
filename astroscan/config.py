"""Configuration management for AstroScan."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import yaml


DEFAULT_VISION_MODELS = [
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-3-27b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-3-12b-it:free",
    "google/gemma-3-4b-it:free",
]


class Config:
    """AstroScan configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or "config.yaml")
        self._data = {}
        self._load()
    
    def _load(self):
        if self.config_path.exists():
            with open(self.config_path) as f:
                self._data = yaml.safe_load(f) or {}
        else:
            # Try example config
            example = self.config_path.parent / "config.example.yaml"
            if example.exists():
                with open(example) as f:
                    self._data = yaml.safe_load(f) or {}
    
    @property
    def openrouter_api_key(self) -> str:
        return self._data.get("openrouter_api_key", "")
    
    @property
    def input_dir(self) -> Path:
        return Path(self._data.get("input_dir", "./input"))
    
    @property
    def output_dir(self) -> Path:
        return Path(self._data.get("output_dir", "./output"))
    
    @property
    def knowledge_base_dir(self) -> Path:
        return Path(self._data.get("knowledge_base_dir", "./knowledge_base"))
    
    @property
    def vision_models(self) -> list[str]:
        return self._data.get("vision_models", DEFAULT_VISION_MODELS)
    
    @property
    def preprocessing(self) -> dict:
        return self._data.get("preprocessing", {
            "deskew": True,
            "enhance_contrast": True,
            "sharpen": True,
            "denoise": True,
        })
    
    @property
    def rate_limit(self) -> dict:
        return self._data.get("rate_limit", {
            "requests_per_minute": 15,
            "retry_delay_seconds": 10,
            "max_retries": 3,
        })
    
    def ensure_dirs(self):
        """Create all required directories."""
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ["concepts", "definitions", "relationships", "charts", "rules", "full_text"]:
            (self.knowledge_base_dir / subdir).mkdir(exist_ok=True)
