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
    chart_type: str = ""
    description: str = ""
    astrological_meaning: str = ""
    elements_found: list[str] = Field(default_factory=list)
    raw_analysis: str = ""


class KnowledgeEntry(BaseModel):
    """A single piece of knowledge extracted from the book."""
    id: str = ""
    category: ConceptCategory = ConceptCategory.OTHER
    title: str = ""
    content: str = ""
    page_number: int = 0
    page_range: Optional[str] = None
    related_entries: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    source_text: str = ""
    is_definition: bool = False
    is_rule: bool = False


class KnowledgeBase(BaseModel):
    """The complete knowledge base built from the book."""
    source_name: str = "Fixed Astrology Textbook"
    total_pages_processed: int = 0
    last_updated: str = Field(default_factory=lambda: datetime.now().isoformat())
    entries: list[KnowledgeEntry] = Field(default_factory=list)
    charts: list[ChartDescription] = Field(default_factory=list)

    # Quick-access indexes
    houses: dict[str, list[str]] = Field(default_factory=dict)
    signs: dict[str, list[str]] = Field(default_factory=dict)
    planets: dict[str, list[str]] = Field(default_factory=dict)
    aspects: dict[str, list[str]] = Field(default_factory=dict)
    retrogrades: dict[str, list[str]] = Field(default_factory=dict)
    definitions: dict[str, str] = Field(default_factory=dict)
    rules: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Graph models (for knowledge graph / GraphRAG)
# ──────────────────────────────────────────────
class GraphEntity(BaseModel):
    """An entity node in the knowledge graph."""
    name: str
    entity_type: str  # planet, sign, house, aspect, element, modality
    mentions: list[str] = Field(default_factory=list)  # entry IDs
    description: str = ""


class GraphRelationship(BaseModel):
    """A relationship edge in the knowledge graph."""
    source: str
    target: str
    relationship_type: str  # rules, belongs_to_element, has_modality, mentioned_with, etc.
    entry_id: str = ""
    weight: float = 1.0


class GraphCommunity(BaseModel):
    """A community/cluster of related concepts."""
    community_id: int
    members: list[str] = Field(default_factory=list)
    label: str = ""
    summary: str = ""
