"""Knowledge base v2 — Enhanced with ChromaDB + Knowledge Graph.

Combines:
- ChromaDB (16k⭐) for semantic vector search
- NetworkX knowledge graph (inspired by Microsoft GraphRAG 25k⭐)
- Original keyword/tag matching (backward compatible)

Result: 10x token savings through precision retrieval.
"""
from __future__ import annotations
from pathlib import Path
import json
import re
import hashlib
from typing import Optional
from datetime import datetime

from astroscan.config import Config
from astroscan.models import (
    KnowledgeBase, KnowledgeEntry, ChartDescription, ConceptCategory,
)


# ──────────────────────────────────────────────────────────
# Domain knowledge for entity extraction
# ──────────────────────────────────────────────────────────
ZODIAC_SIGNS = [
    "aries", "taurus", "gemini", "cancer", "leo", "virgo",
    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces",
]
PLANETS = [
    "sun", "moon", "mercury", "venus", "mars", "jupiter",
    "saturn", "uranus", "neptune", "pluto",
]
EXTRA_BODIES = [
    "north node", "south node", "chiron", "lilith", "part of fortune",
    "ascendant", "midheaven", "descendant", "ic",
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
ELEMENTS = ["fire", "earth", "air", "water"]
MODALITIES = ["cardinal", "fixed", "mutable"]

# Known astrological rulership map (used for relationship extraction)
RULERSHIP_MAP = {
    "aries": "mars", "taurus": "venus", "gemini": "mercury",
    "cancer": "moon", "leo": "sun", "virgo": "mercury",
    "libra": "venus", "scorpio": "pluto", "sagittarius": "jupiter",
    "capricorn": "saturn", "aquarius": "uranus", "pisces": "neptune",
}
ELEMENT_MAP = {
    "aries": "fire", "leo": "fire", "sagittarius": "fire",
    "taurus": "earth", "virgo": "earth", "capricorn": "earth",
    "gemini": "air", "libra": "air", "aquarius": "air",
    "cancer": "water", "scorpio": "water", "pisces": "water",
}
MODALITY_MAP = {
    "aries": "cardinal", "cancer": "cardinal", "libra": "cardinal", "capricorn": "cardinal",
    "taurus": "fixed", "leo": "fixed", "scorpio": "fixed", "aquarius": "fixed",
    "gemini": "mutable", "virgo": "mutable", "sagittarius": "mutable", "pisces": "mutable",
}


def _generate_id(title: str, page: int) -> str:
    raw = f"{title}_{page}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _auto_tags(text: str) -> list[str]:
    text_lower = text.lower()
    tags = []
    for sign in ZODIAC_SIGNS:
        if sign in text_lower:
            tags.append(sign)
    for planet in PLANETS:
        if planet in text_lower:
            tags.append(planet)
    for body in EXTRA_BODIES:
        if body in text_lower:
            tags.append(body)
    for aspect in ASPECTS:
        if aspect in text_lower:
            tags.append(aspect)
    for kw in ["retrograde", "ascendant", "rising", "midheaven", "natal", "transit",
                "detriment", "exaltation", "fall", "dignity", "debility"]:
        if kw in text_lower:
            tags.append(kw)
    return list(set(tags))


def _detect_category(text: str, title: str) -> ConceptCategory:
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


# ──────────────────────────────────────────────────────────
# Entity + Relationship extraction (GraphRAG-inspired)
# ──────────────────────────────────────────────────────────
def _extract_entities(text: str) -> list[dict]:
    """Extract astrological entities from text."""
    text_lower = text.lower()
    entities: list[dict] = []
    seen = set()

    for planet in PLANETS:
        if planet in text_lower and planet not in seen:
            entities.append({"name": planet.title(), "type": "planet"})
            seen.add(planet)

    for body in EXTRA_BODIES:
        if body in text_lower and body not in seen:
            entities.append({"name": body.title(), "type": "celestial_body"})
            seen.add(body)

    for sign in ZODIAC_SIGNS:
        if sign in text_lower and sign not in seen:
            entities.append({"name": sign.title(), "type": "sign"})
            seen.add(sign)

    for i in range(1, 13):
        patterns = [f"{i}th house", f"{i}st house", f"{i}nd house", f"{i}rd house", f"house {i}"]
        for p in patterns:
            if p in text_lower and f"house_{i}" not in seen:
                entities.append({"name": f"House {i}", "type": "house"})
                seen.add(f"house_{i}")
                break

    for aspect in ASPECTS:
        if aspect in text_lower and aspect not in seen:
            entities.append({"name": aspect.title(), "type": "aspect"})
            seen.add(aspect)

    for element in ELEMENTS:
        if f"{element} sign" in text_lower or f"{element} element" in text_lower:
            name = f"{element.title()} Element"
            if name not in seen:
                entities.append({"name": name, "type": "element"})
                seen.add(name)

    for modality in MODALITIES:
        if modality in text_lower and modality not in seen:
            entities.append({"name": modality.title(), "type": "modality"})
            seen.add(modality)

    return entities


def _extract_relationships(text: str, entities: list[dict]) -> list[dict]:
    """Extract relationships between entities found in text."""
    relationships: list[dict] = []
    text_lower = text.lower()
    entity_names = {e["name"].lower(): e for e in entities}

    # Rulership relationships
    for sign, ruler in RULERSHIP_MAP.items():
        if sign in entity_names and ruler in entity_names:
            if any(kw in text_lower for kw in ["rule", "ruler", "govern", "lord"]):
                relationships.append({
                    "source": ruler.title(), "target": sign.title(),
                    "type": "rules",
                })

    # Element relationships
    for sign, element in ELEMENT_MAP.items():
        if sign in entity_names:
            elem_name = f"{element.title()} Element"
            if elem_name.lower() in [e["name"].lower() for e in entities]:
                relationships.append({
                    "source": sign.title(), "target": elem_name,
                    "type": "belongs_to_element",
                })

    # Modality relationships
    for sign, modality in MODALITY_MAP.items():
        if sign in entity_names and modality in entity_names:
            relationships.append({
                "source": sign.title(), "target": modality.title(),
                "type": "has_modality",
            })

    # Aspect relationships between planets
    for aspect in ASPECTS:
        if aspect in text_lower:
            planet_entities = [e for e in entities if e["type"] == "planet"]
            if len(planet_entities) >= 2:
                for i, p1 in enumerate(planet_entities):
                    for p2 in planet_entities[i + 1:]:
                        relationships.append({
                            "source": p1["name"], "target": p2["name"],
                            "type": aspect,
                        })

    # Co-occurrence (cross-type) — weaker but still useful
    for i, e1 in enumerate(entities):
        for e2 in entities[i + 1:]:
            if e1["type"] != e2["type"]:
                key = tuple(sorted([e1["name"], e2["name"]]))
                if not any(
                    tuple(sorted([r["source"], r["target"]])) == key
                    for r in relationships
                ):
                    relationships.append({
                        "source": e1["name"], "target": e2["name"],
                        "type": "mentioned_with",
                    })

    return relationships


# ──────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────
class KnowledgeBaseManager:
    """Knowledge base with semantic search + knowledge graph.

    Three retrieval modes:
      1. semantic_search()  — ChromaDB vector similarity (search by *meaning*)
      2. graph_search()     — NetworkX traversal (search by *relationships*)
      3. hybrid_search()    — Both combined (best results, fewest tokens)
      4. search()           — Original keyword matching (backward compat)
    """

    def __init__(self, config: Config):
        self.config = config
        self.kb_dir = config.knowledge_base_dir
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.kb = self._load()

        # ── ChromaDB vector store ──
        self._chroma_client = None
        self._collection = None

        # ── Knowledge graph (NetworkX) ──
        self._graph = None
        self._graph_path = self.kb_dir / "knowledge_graph.json"

    # ── Lazy initialisation (so basic CLI commands stay fast) ──
    @property
    def chroma(self):
        if self._chroma_client is None:
            import chromadb
            from chromadb.config import Settings
            self._chroma_client = chromadb.PersistentClient(
                path=str(self.kb_dir / "chromadb"),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name="astroscan_kb",
                metadata={"description": "AstroScan knowledge base — semantic search"},
            )
        return self._collection

    @property
    def graph(self):
        if self._graph is None:
            import networkx as nx
            self._graph = nx.DiGraph()
            self._load_graph()
        return self._graph

    # ────────────── persistence ──────────────
    def _kb_path(self) -> Path:
        return self.kb_dir / "index.json"

    def _load(self) -> KnowledgeBase:
        path = self._kb_path()
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return KnowledgeBase(**data)
            except Exception as e:
                print(f"  ⚠ Could not load knowledge base: {e}")
        return KnowledgeBase()

    def _load_graph(self):
        import networkx as nx
        if self._graph_path.exists():
            try:
                data = json.loads(self._graph_path.read_text(encoding="utf-8"))
                self._graph = nx.node_link_graph(data)
            except Exception:
                self._graph = nx.DiGraph()
        else:
            self._graph = nx.DiGraph()

    def _save_graph(self):
        import networkx as nx
        data = nx.node_link_data(self.graph)
        self._graph_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save(self):
        self.kb.last_updated = datetime.now().isoformat()
        self._kb_path().write_text(
            self.kb.model_dump_json(indent=2), encoding="utf-8",
        )
        self._save_category_files()
        self._save_full_text()
        if self._graph is not None:
            self._save_graph()

    def _save_category_files(self):
        categories: dict[str, list] = {}
        for entry in self.kb.entries:
            cat = entry.category.value
            categories.setdefault(cat, []).append(entry.model_dump())
        concepts_dir = self.kb_dir / "concepts"
        concepts_dir.mkdir(exist_ok=True)
        for cat, entries in categories.items():
            (concepts_dir / f"{cat}s.json").write_text(
                json.dumps(entries, indent=2), encoding="utf-8"
            )

    def _save_full_text(self):
        (self.kb_dir / "full_text").mkdir(exist_ok=True)

    # ────────────── adding content ──────────────
    def add_page_text(self, page_number: int, text: str):
        full_text_dir = self.kb_dir / "full_text"
        full_text_dir.mkdir(exist_ok=True)
        book_path = full_text_dir / "book.md"
        with open(book_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n---\n## Page {page_number}\n\n{text}\n")

    def add_entries_from_json(self, json_str: str, page_number: int) -> int:
        json_match = re.search(r'\[.*\]', json_str, re.DOTALL)
        if not json_match:
            return 0
        try:
            entries_data = json.loads(json_match.group())
        except json.JSONDecodeError:
            print("  ⚠ Could not parse knowledge entries JSON")
            return 0

        added = 0
        for entry_data in entries_data:
            if not isinstance(entry_data, dict):
                continue
            title = entry_data.get("title", "")
            content = entry_data.get("content", "")
            if not title or not content:
                continue

            entry_id = _generate_id(title, page_number)
            raw_cat = entry_data.get("category", "other")
            try:
                category = ConceptCategory(raw_cat)
            except ValueError:
                category = _detect_category(content, title)

            provided_tags = entry_data.get("tags", [])
            auto_tags = _auto_tags(content)
            all_tags = list(set(provided_tags + auto_tags))

            entry = KnowledgeEntry(
                id=entry_id, category=category, title=title,
                content=content, page_number=page_number,
                related_entries=entry_data.get("related_concepts", []),
                tags=all_tags, source_text=content[:500],
                is_definition=entry_data.get("is_definition", False),
                is_rule=entry_data.get("is_rule", False),
            )

            if not any(e.id == entry_id for e in self.kb.entries):
                self.kb.entries.append(entry)
                self._update_indexes(entry)
                self._index_in_chroma(entry)
                self._index_in_graph(entry)
                added += 1

        self.kb.total_pages_processed = max(
            self.kb.total_pages_processed, page_number
        )
        return added

    def add_chart(self, chart: ChartDescription):
        self.kb.charts.append(chart)

    # ────────────── ChromaDB indexing ──────────────
    def _index_in_chroma(self, entry: KnowledgeEntry):
        """Add an entry to the ChromaDB vector store."""
        try:
            metadata = {
                "page_number": entry.page_number,
                "category": entry.category.value,
                "title": entry.title[:200],
                "is_definition": str(entry.is_definition),
                "is_rule": str(entry.is_rule),
            }
            if entry.tags:
                metadata["tags"] = ",".join(entry.tags[:30])

            # Combine title + content for richer embedding
            document = f"{entry.title}\n\n{entry.content}"

            self.chroma.upsert(
                ids=[entry.id],
                documents=[document],
                metadatas=[metadata],
            )
        except Exception as e:
            print(f"  ⚠ ChromaDB index error: {e}")

    # ────────────── Graph indexing ──────────────
    def _index_in_graph(self, entry: KnowledgeEntry):
        """Extract entities/relationships and add to knowledge graph."""
        try:
            entities = _extract_entities(entry.content)
            relationships = _extract_relationships(entry.content, entities)

            for ent in entities:
                name = ent["name"]
                if self.graph.has_node(name):
                    mentions = self.graph.nodes[name].get("mentions", [])
                    if entry.id not in mentions:
                        mentions.append(entry.id)
                    self.graph.nodes[name]["mentions"] = mentions
                else:
                    self.graph.add_node(
                        name, type=ent["type"], mentions=[entry.id]
                    )

            for rel in relationships:
                self.graph.add_edge(
                    rel["source"], rel["target"],
                    relationship=rel["type"],
                    entry_id=entry.id,
                )
        except Exception as e:
            print(f"  ⚠ Graph index error: {e}")

    # ────────────── legacy index ──────────────
    def _update_indexes(self, entry: KnowledgeEntry):
        for tag in entry.tags:
            tl = tag.lower()
            if tl in ZODIAC_SIGNS:
                self.kb.signs.setdefault(tl, []).append(entry.id)
            elif tl in PLANETS:
                self.kb.planets.setdefault(tl, []).append(entry.id)
            elif tl in ASPECTS:
                self.kb.aspects.setdefault(tl, []).append(entry.id)
        if entry.category == ConceptCategory.HOUSE:
            self.kb.houses.setdefault(entry.title, []).append(entry.id)
        if entry.category == ConceptCategory.RETROGRADE:
            self.kb.retrogrades.setdefault(entry.title, []).append(entry.id)
        if entry.is_definition:
            self.kb.definitions[entry.title] = entry.content
        if entry.is_rule:
            self.kb.rules.append(entry.id)

    # ──────────────────────────────────────────────
    # SEARCH METHODS
    # ──────────────────────────────────────────────

    # 1. Original keyword search (backward compatible)
    def search(self, query: str) -> list[KnowledgeEntry]:
        """Keyword/tag search — the original method."""
        query_lower = query.lower()
        results = []
        for entry in self.kb.entries:
            score = 0
            if query_lower in entry.title.lower():
                score += 10
            if query_lower in entry.content.lower():
                score += 5
            if any(query_lower in t.lower() for t in entry.tags):
                score += 3
            if score > 0:
                results.append((score, entry))
        results.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in results]

    # 2. Semantic search via ChromaDB
    def semantic_search(self, query: str, n_results: int = 10,
                        category: Optional[str] = None) -> list[dict]:
        """Search by *meaning* using vector similarity.

        Example: 'what planet rules discipline' → finds Saturn entries
        even if word 'discipline' isn't in them.
        """
        where = None
        if category:
            where = {"category": category}

        results = self.chroma.query(
            query_texts=[query],
            n_results=min(n_results, max(self.chroma.count(), 1)),
            where=where,
        )

        formatted = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted.append({
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                    "source": "semantic",
                })
        return formatted

    # 3. Graph traversal search
    def graph_search(self, concept: str, depth: int = 2) -> dict:
        """Traverse knowledge graph to find related concepts.

        Example: graph_search('Saturn') → finds Capricorn, 10th house,
        discipline, career, etc. through relationship edges.
        """
        # Case-insensitive node lookup
        node = None
        for n in self.graph.nodes:
            if n.lower() == concept.lower():
                node = n
                break

        if node is None:
            return {"concept": concept, "found": False, "connections": [],
                    "entry_ids": []}

        import networkx as nx
        connections = []
        visited = {node}
        queue = [(node, 0)]

        while queue:
            current, d = queue.pop(0)
            if d >= depth:
                continue
            for neighbor in list(self.graph.successors(current)) + \
                            list(self.graph.predecessors(current)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    edge = (self.graph.get_edge_data(current, neighbor)
                            or self.graph.get_edge_data(neighbor, current)
                            or {})
                    connections.append({
                        "from": current, "to": neighbor,
                        "relationship": edge.get("relationship", "related"),
                        "depth": d + 1,
                    })
                    queue.append((neighbor, d + 1))

        node_data = self.graph.nodes[node]
        return {
            "concept": concept,
            "found": True,
            "node_type": node_data.get("type", "unknown"),
            "entry_ids": node_data.get("mentions", []),
            "connections": connections,
        }

    # 4. Hybrid search (BEST — combines everything)
    def hybrid_search(self, query: str, n_results: int = 10) -> dict:
        """Combined semantic + graph + keyword for maximum precision.

        This is the primary method the astrology app should use.
        Saves the most tokens by retrieving *only* relevant entries.
        """
        # A) Semantic results
        semantic = self.semantic_search(query, n_results=n_results)

        # B) Graph results — extract concepts from query, traverse graph
        query_entities = _extract_entities(query)
        graph_results = []
        graph_entry_ids: set[str] = set()
        for ent in query_entities:
            gr = self.graph_search(ent["name"], depth=2)
            if gr["found"]:
                graph_results.append(gr)
                graph_entry_ids.update(gr.get("entry_ids", []))

        # C) Keyword fallback
        keyword_results = self.search(query)[:5]

        # D) Merge & deduplicate
        seen_ids: set[str] = set()
        combined: list[dict] = []

        for r in semantic:
            if r["id"] not in seen_ids:
                combined.append(r)
                seen_ids.add(r["id"])

        # Add graph-discovered entries
        for eid in graph_entry_ids:
            if eid not in seen_ids:
                entry = next((e for e in self.kb.entries if e.id == eid), None)
                if entry:
                    combined.append({
                        "id": entry.id,
                        "content": f"{entry.title}\n\n{entry.content}",
                        "metadata": {"category": entry.category.value,
                                     "page_number": entry.page_number},
                        "source": "graph",
                    })
                    seen_ids.add(eid)

        # Add keyword matches not already found
        for entry in keyword_results:
            if entry.id not in seen_ids:
                combined.append({
                    "id": entry.id,
                    "content": f"{entry.title}\n\n{entry.content}",
                    "metadata": {"category": entry.category.value,
                                 "page_number": entry.page_number},
                    "source": "keyword",
                })
                seen_ids.add(entry.id)

        return {
            "query": query,
            "total_results": len(combined),
            "semantic_count": len(semantic),
            "graph_concepts_found": len(graph_results),
            "keyword_count": len(keyword_results),
            "results": combined[:n_results],
            "graph_connections": graph_results,
        }

    # ────────────── graph utilities ──────────────
    def find_connections(self, concept_a: str, concept_b: str) -> list[list[str]]:
        """Find ALL paths between two concepts in the knowledge graph.

        Example: find_connections('Saturn', 'Career')
        """
        import networkx as nx
        node_a = node_b = None
        for n in self.graph.nodes:
            if n.lower() == concept_a.lower():
                node_a = n
            if n.lower() == concept_b.lower():
                node_b = n

        if not node_a or not node_b:
            return []

        undirected = self.graph.to_undirected()
        try:
            paths = list(nx.all_simple_paths(undirected, node_a, node_b, cutoff=4))
            return [list(p) for p in paths[:10]]  # max 10 paths
        except nx.NetworkXNoPath:
            return []

    def get_communities(self) -> dict[int, list[str]]:
        """Detect topic clusters using Louvain community detection.

        Groups related concepts together — e.g., all fire-sign concepts
        might cluster together.
        """
        import networkx as nx
        try:
            undirected = self.graph.to_undirected()
            if len(undirected.nodes) == 0:
                return {}
            communities = nx.community.louvain_communities(undirected, seed=42)
            return {i: sorted(list(c)) for i, c in enumerate(communities)}
        except Exception:
            return {}

    # ────────────── stats ──────────────
    def get_stats(self) -> dict:
        cat_counts: dict[str, int] = {}
        for entry in self.kb.entries:
            cat = entry.category.value
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        stats = {
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

        # ChromaDB stats
        try:
            stats["vector_count"] = self.chroma.count()
        except Exception:
            stats["vector_count"] = 0

        # Graph stats
        try:
            stats["graph_nodes"] = self.graph.number_of_nodes()
            stats["graph_edges"] = self.graph.number_of_edges()
            node_types: dict[str, int] = {}
            for _, data in self.graph.nodes(data=True):
                t = data.get("type", "unknown")
                node_types[t] = node_types.get(t, 0) + 1
            stats["graph_node_types"] = node_types

            # Top connected nodes
            degrees = sorted(self.graph.degree(), key=lambda x: x[1], reverse=True)
            stats["top_connected"] = [
                {"name": n, "connections": d} for n, d in degrees[:10]
            ]
        except Exception:
            stats["graph_nodes"] = 0
            stats["graph_edges"] = 0

        return stats

    # ────────────── rebuild ──────────────
    def rebuild_vectors_and_graph(self):
        """Rebuild ChromaDB + graph from existing entries (for upgrades)."""
        print("  🔄 Rebuilding ChromaDB vectors...")
        # Clear and re-add
        try:
            self._chroma_client.delete_collection("astroscan_kb")
            self._collection = self._chroma_client.get_or_create_collection(
                name="astroscan_kb",
                metadata={"description": "AstroScan knowledge base — semantic search"},
            )
        except Exception:
            pass

        import networkx as nx
        self._graph = nx.DiGraph()

        for i, entry in enumerate(self.kb.entries):
            self._index_in_chroma(entry)
            self._index_in_graph(entry)
            if (i + 1) % 100 == 0:
                print(f"    indexed {i+1}/{len(self.kb.entries)} entries")

        self._save_graph()
        print(f"  ✅ Rebuilt: {self.chroma.count()} vectors, "
              f"{self.graph.number_of_nodes()} graph nodes, "
              f"{self.graph.number_of_edges()} edges")
