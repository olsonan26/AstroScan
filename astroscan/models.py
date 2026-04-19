"""Data models for AstroScan pipeline."""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class ConceptCategory(str, Enum):
    HOUSE = "house"
    SIGN = "sign"
    PLANET = "planet"
    ASPECT = "aspect"
    RETROGRADE = "retrograde"
    CHART_TYPE = "chart_type"
    RULE = "rule"
    DEFINITION = "definition"
    RELATIONSHIP = "relationship"
    OTHER = "other"


class PageMetadata(BaseModel):
    """Metadata for a processed page."""
    page_number: int
    original_filename: str
    image_width: int = 0
    image_height: int = 0
    file_size_bytes: int = 0
    text_length: int = 0
    num_figures: int = 0
    has_charts: bool = False
    ocr_model: Optional[str] = None
    vision_model: Optional[str] = None
    processing_time_seconds: float = 0.0
    processed_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ChartDescription(BaseModel):
    """Description of a chart/diagram found on a page."""
    page_number: int
    figure_index: int = 0
    chart_type: str = ""          # "natal_wheel", "aspect_grid", "house_diagram", etc.
    description: str = ""          # Full visual description
    astrological_meaning: str = "" # What concept it teaches
    elements_found: list[str] = Field(default_factory=list)  # Signs, planets, houses identified
    raw_analysis: str = ""         # Full vision AI output


class KnowledgeEntry(BaseModel):
    """A single piece of knowledge extracted from the book."""
    id: str = ""
    category: ConceptCategory = ConceptCategory.OTHER
    title: str = ""
    content: str = ""
    page_number: int = 0
    page_range: Optional[str] = None  # e.g. "156-158" for multi-page concepts
    related_entries: list[str] = Field(default_factory=list)  # IDs of related entries
    tags: list[str] = Field(default_factory=list)
    source_text: str = ""         # Original text from the book
    is_definition: bool = False
    is_rule: bool = False


class KnowledgeBase(BaseModel):
    """The complete knowledge base built from the book."""
    source_name: str = "Fixed Astrology Textbook"
    total_pages_processed: int = 0
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())
    entries: list[KnowledgeEntry] = Field(default_factory=list)
    charts: list[ChartDescription] = Field(default_factory=list)
    
    # Quick-access indexes (built from entries)
    houses: dict[str, list[str]] = Field(default_factory=dict)      # house_name -> entry IDs
    signs: dict[str, list[str]] = Field(default_factory=dict)       # sign_name -> entry IDs
    planets: dict[str, list[str]] = Field(default_factory=dict)     # planet_name -> entry IDs
    aspects: dict[str, list[str]] = Field(default_factory=dict)     # aspect_name -> entry IDs
    retrogrades: dict[str, list[str]] = Field(default_factory=dict) # retrograde info -> entry IDs
    definitions: dict[str, str] = Field(default_factory=dict)       # term -> definition
    rules: list[str] = Field(default_factory=list)                  # entry IDs of rules
