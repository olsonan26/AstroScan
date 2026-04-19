"""Knowledge base builder and manager.

Structures extracted content into an indexed, searchable knowledge system
that uses ONLY the book's definitions (fixed astrology, not conventional).
"""
from __future__ import annotations
from pathlib import Path
import json
import re
import hashlib
from typing import Optional

from astroscan.config import Config
from astroscan.models import (
    KnowledgeBase, KnowledgeEntry, ChartDescription, ConceptCategory,
)


# Astrological terms to detect for auto-tagging
ZODIAC_SIGNS = [
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces",
]
PLANETS = [
    "sun", "moon", "mercury", "venus", "mars", "jupiter",
    "saturn", "uranus", "neptune", "pluto",
]
ASPECTS = [
    "conjunction", "opposition", "trine", "square", "sextile",
    "quincunx", "semi-sextile", "semi-square", "sesquiquadrate",
]
HOUSES = [f"house {i}" for i in range(1, 13)] + [
    f"{i}st house" if i == 1 else f"{i}nd house" if i == 2
    else f"{i}rd house" if i == 3 else f"{i}th house"
    for i in range(1, 13)
]


def _generate_id(title: str, page: int) -> str:
    """Generate a unique ID for a knowledge entry."""
    raw = f"{title}_{page}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _auto_tags(text: str) -> list[str]:
    """Auto-detect astrological terms in text for tagging."""
    text_lower = text.lower()
    tags = []
    for sign in ZODIAC_SIGNS:
        if sign in text_lower:
            tags.append(sign)
    for planet in PLANETS:
        if planet in text_lower:
            tags.append(planet)
    for aspect in ASPECTS:
        if aspect in text_lower:
            tags.append(aspect)
    if "retrograde" in text_lower:
        tags.append("retrograde")
    if "ascendant" in text_lower or "rising" in text_lower:
        tags.append("ascendant")
    if "midheaven" in text_lower or "mc" in text_lower.split():
        tags.append("midheaven")
    if "natal" in text_lower:
        tags.append("natal")
    if "transit" in text_lower:
        tags.append("transit")
    return list(set(tags))


def _detect_category(text: str, title: str) -> ConceptCategory:
    """Auto-detect the category of a knowledge entry."""
    combined = (title + " " + text).lower()
    
    if any(h in combined for h in HOUSES) or "house" in combined:
        return ConceptCategory.HOUSE
    if any(s in combined for s in ZODIAC_SIGNS):
        if "meaning" in combined or "represents" in combined or "rules" in combined:
            return ConceptCategory.SIGN
    if any(p in combined for p in PLANETS):
        if "meaning" in combined or "represents" in combined or "rules" in combined:
            return ConceptCategory.PLANET
    if any(a in combined for a in ASPECTS):
        return ConceptCategory.ASPECT
    if "retrograde" in combined:
        return ConceptCategory.RETROGRADE
    if "rule" in combined or "always" in combined or "must" in combined or "never" in combined:
        return ConceptCategory.RULE
    if "means" in combined or "defined as" in combined or "definition" in combined:
        return ConceptCategory.DEFINITION
    
    return ConceptCategory.OTHER


class KnowledgeBaseManager:
    """Manages the knowledge base — loading, updating, querying, exporting."""
    
    def __init__(self, config: Config):
        self.config = config
        self.kb_dir = config.knowledge_base_dir
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.kb = self._load()
    
    def _kb_path(self) -> Path:
        return self.kb_dir / "index.json"
    
    def _load(self) -> KnowledgeBase:
        """Load existing knowledge base or create new one."""
        path = self._kb_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return KnowledgeBase(**data)
            except Exception as e:
                print(f"  ⚠ Could not load knowledge base: {e}")
        return KnowledgeBase()
    
    def save(self):
        """Save knowledge base to disk."""
        self._kb_path().write_text(
            self.kb.model_dump_json(indent=2),
            encoding="utf-8",
        )
        self._save_category_files()
        self._save_full_text()
    
    def _save_category_files(self):
        """Save category-specific files for easy browsing."""
        categories = {}
        for entry in self.kb.entries:
            cat = entry.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(entry.model_dump())
        
        concepts_dir = self.kb_dir / "concepts"
        concepts_dir.mkdir(exist_ok=True)
        
        for cat, entries in categories.items():
            path = concepts_dir / f"{cat}s.json"
            path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    
    def _save_full_text(self):
        """Save continuous book text for reference."""
        full_text_dir = self.kb_dir / "full_text"
        full_text_dir.mkdir(exist_ok=True)
        # This gets appended to during processing
    
    def add_page_text(self, page_number: int, text: str):
        """Append a page's text to the full book text."""
        full_text_dir = self.kb_dir / "full_text"
        full_text_dir.mkdir(exist_ok=True)
        
        book_path = full_text_dir / "book.md"
        with open(book_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n---\n## Page {page_number}\n\n{text}\n")
    
    def add_entries_from_json(self, json_str: str, page_number: int) -> int:
        """Parse and add knowledge entries from vision AI JSON output.
        
        Returns number of entries added.
        """
        # Extract JSON from response (might be wrapped in markdown code blocks)
        json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
        if not json_match:
            return 0
        
        try:
            entries_data = json.loads(json_match.group())
        except json.JSONDecodeError:
            print(f"  ⚠ Could not parse knowledge entries JSON")
            return 0
        
        added = 0
        for entry_data in entries_data:
            if not isinstance(entry_data, dict):
                continue
            
            title = entry_data.get("title", "")
            content = entry_data.get("content", "")
            if not title or not content:
                continue
            
            # Build entry
            entry_id = _generate_id(title, page_number)
            
            # Auto-detect category if not provided or invalid
            raw_cat = entry_data.get("category", "other")
            try:
                category = ConceptCategory(raw_cat)
            except ValueError:
                category = _detect_category(content, title)
            
            # Merge auto-tags with provided tags
            provided_tags = entry_data.get("tags", [])
            auto_tags = _auto_tags(content)
            all_tags = list(set(provided_tags + auto_tags))
            
            entry = KnowledgeEntry(
                id=entry_id,
                category=category,
                title=title,
                content=content,
                page_number=page_number,
                related_entries=entry_data.get("related_concepts", []),
                tags=all_tags,
                source_text=content[:500],
                is_definition=entry_data.get("is_definition", False),
                is_rule=entry_data.get("is_rule", False),
            )
            
            # Check for duplicates
            if not any(e.id == entry_id for e in self.kb.entries):
                self.kb.entries.append(entry)
                self._update_indexes(entry)
                added += 1
        
        self.kb.total_pages_processed = max(
            self.kb.total_pages_processed, page_number
        )
        return added
    
    def add_chart(self, chart: ChartDescription):
        """Add a chart description to the knowledge base."""
        self.kb.charts.append(chart)
    
    def _update_indexes(self, entry: KnowledgeEntry):
        """Update quick-access indexes with a new entry."""
        for tag in entry.tags:
            tag_lower = tag.lower()
            if tag_lower in ZODIAC_SIGNS:
                self.kb.signs.setdefault(tag_lower, []).append(entry.id)
            elif tag_lower in PLANETS:
                self.kb.planets.setdefault(tag_lower, []).append(entry.id)
            elif tag_lower in ASPECTS:
                self.kb.aspects.setdefault(tag_lower, []).append(entry.id)
        
        if entry.category == ConceptCategory.HOUSE:
            self.kb.houses.setdefault(entry.title, []).append(entry.id)
        if entry.category == ConceptCategory.RETROGRADE:
            self.kb.retrogrades.setdefault(entry.title, []).append(entry.id)
        if entry.is_definition:
            self.kb.definitions[entry.title] = entry.content
        if entry.is_rule:
            self.kb.rules.append(entry.id)
    
    def search(self, query: str) -> list[KnowledgeEntry]:
        """Search the knowledge base for entries matching a query."""
        query_lower = query.lower()
        results = []
        
        for entry in self.kb.entries:
            score = 0
            if query_lower in entry.title.lower():
                score += 10
            if query_lower in entry.content.lower():
                score += 5
            if any(query_lower in tag.lower() for tag in entry.tags):
                score += 3
            if score > 0:
                results.append((score, entry))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in results]
    
    def get_stats(self) -> dict:
        """Get knowledge base statistics."""
        cat_counts = {}
        for entry in self.kb.entries:
            cat = entry.category.value
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        
        return {
            "total_entries": len(self.kb.entries),
            "total_charts": len(self.kb.charts),
            "pages_processed": self.kb.total_pages_processed,
            "categories": cat_counts,
            "total_definitions": len(self.kb.definitions),
            "total_rules": len(self.kb.rules),
            "signs_indexed": len(self.kb.signs),
            "planets_indexed": len(self.kb.planets),
            "houses_indexed": len(self.kb.houses),
        }
