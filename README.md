# 🔭 AstroScan v3.0 — Book-to-Knowledge-Base Pipeline

> Photograph book pages → AI extracts all knowledge → Searchable, indexed knowledge base
> Built specifically for a 1000+ page astrology textbook with charts, diagrams, and precise definitions.

---

## 🏗️ Architecture

```
📷 Book Photos
    │
    ▼
┌─────────────────────────────┐
│ 📐 DocScanner Dewarping     │  Fix curved/warped page photos
│   (Geometric + Deep Learning)│  (auto-detects if needed)
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│ 🔧 OpenCV Preprocessing     │  Deskew → Denoise → Contrast → Sharpen
│   (CLAHE + Unsharp Mask)    │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────────────────────────────┐
│ 📖 Multi-Engine OCR (best accuracy via cross-check) │
│                                                      │
│  🏆 MinerU (33k⭐)          Primary — beats GPT-4o  │
│  📄 Marker + Surya (37k⭐)  Secondary — structured   │
│  👁️ Vision AI (free models)  Fallback — always works │
│                                                      │
│  → Results merged with confidence scoring            │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│ 👁️ Vision AI Chart Analysis                 │
│  Analyzes natal wheels, aspect tables,      │
│  diagrams, symbols — anything visual        │
│  (Free via OpenRouter: Gemma, Nemotron)     │
└─────────────┬───────────────────────────────┘
              ▼
┌─────────────────────────────────────────────────────────┐
│ 🧠 3-Layer Knowledge Base                               │
│                                                          │
│  Layer 1: ChromaDB Vectors (16k⭐)                      │
│    → Semantic search: "what rules career?" finds         │
│      Saturn/10th house/Capricorn even without keywords   │
│                                                          │
│  Layer 2: Knowledge Graph (GraphRAG-inspired, 25k⭐)    │
│    → Relationship mapping: Saturn → rules → Capricorn    │
│    → Path finding: "How is Mars connected to Aries?"     │
│    → Community detection: auto-clusters related concepts │
│                                                          │
│  Layer 3: Structured Index (category/tag lookups)        │
│    → Quick access by: house, sign, planet, aspect, rule  │
│    → Full-text search across all extracted content        │
└─────────────────────────────────────────────────────────┘
```

---

## ⚡ What's New in v3.0

| Feature | v1.0 | v2.0 | v3.0 |
|---------|------|------|------|
| OCR Engine | Marker only | Marker only | **MinerU + Marker (merged)** |
| Page Dewarping | ❌ | ❌ | **✅ Auto-detect + fix curved pages** |
| Search | Keyword only | + Semantic + Graph | + Multi-engine confidence |
| Knowledge Base | JSON index | + ChromaDB + NetworkX | Same (already god-tier) |
| CLI Commands | 5 | 10 | **13** |

### New CLI Commands in v3.0
- `astroscan engines` — Show which OCR engines are installed
- `astroscan check-page <image>` — Analyze curvature/quality before processing
- `astroscan dewarp <image>` — Dewarp a single curved page

---

## 🚀 Quick Start (Windows)

### 1. Install
```cmd
git clone https://github.com/olsonan26/AstroScan.git
cd AstroScan
pip install -e .

REM Optional: Install MinerU for best accuracy (GPU recommended)
pip install magic-pdf[full]
```

### 2. Configure
```cmd
copy config.example.yaml config.yaml
REM Edit config.yaml: add your OpenRouter API key
```

### 3. Drop photos & process
```cmd
REM Put book page photos in ./input/
astroscan process
```

### 4. Search your knowledge base
```cmd
astroscan hybrid-search "what sign does Saturn rule"
astroscan graph-search "Saturn"
astroscan find-path "Mars" "Aries"
astroscan communities
astroscan stats
```

---

## 🔧 All CLI Commands

```
📖 Processing
  astroscan process           Full pipeline (dewarp → OCR → vision → KB)
  astroscan process -f img    Process a single image
  astroscan process -p 1-50   Process specific pages

🔍 Search (3 modes + hybrid)
  astroscan search <query>          Keyword search (fast, exact)
  astroscan semantic-search <query> AI-powered meaning search (ChromaDB)
  astroscan graph-search <concept>  Knowledge graph traversal
  astroscan hybrid-search <query>   🔥 All 3 combined (BEST)

🕸️ Knowledge Graph
  astroscan find-path <A> <B>    Find connections between concepts
  astroscan communities          Show concept clusters

📊 Utilities
  astroscan stats              Knowledge base statistics
  astroscan engines            Show installed OCR engines
  astroscan check-page <img>   Analyze page quality + curvature
  astroscan dewarp <img>       Dewarp a single curved page
  astroscan rebuild-kb         Rebuild KB from output files
  astroscan export             Export KB to JSON or Markdown
```

---

## 📦 Open Source Stack

| Component | Stars | Role |
|-----------|-------|------|
| **MinerU** (OpenDataLab) | ⭐33.1k | Primary OCR — 1.2B model beats GPT-4o |
| **Marker** | ⭐23k | Secondary OCR — structured markdown |
| **Surya** (in Marker) | ⭐14k | Layout detection, reading order |
| **ChromaDB** | ⭐16k | Semantic vector search |
| **NetworkX** + GraphRAG concepts | ⭐25k | Knowledge graph, communities |
| **OpenCV** | ⭐82k | Image preprocessing + geometric dewarping |
| **DocScanner** (IJCV 2025) | ⭐205 | Deep learning page rectification |
| **OpenRouter** | — | Free vision models for chart analysis |

---

## ⚙️ Configuration

```yaml
# config.yaml
openrouter_api_key: "sk-or-v1-your-key-here"

input_dir: "./input"
output_dir: "./output"
knowledge_base_dir: "./knowledge_base"

# v3.0: Dewarping
dewarping:
  enabled: true
  method: "auto"  # auto, docscanner, geometric, off
  auto_detect_threshold: 0.3

# v3.0: OCR engines
ocr_engines:
  mineru: true      # Best accuracy (needs pip install magic-pdf[full])
  marker: true      # Structured markdown
  vision_ai: true   # Free fallback
  merge_strategy: "best"  # best, combine, mineru_only, marker_only

# Preprocessing
preprocessing:
  deskew: true
  enhance_contrast: true
  sharpen: true
  denoise: true

# Vision models (free, rotated for rate limits)
vision_models:
  - "google/gemma-4-26b-a4b-it:free"
  - "google/gemma-3-27b-it:free"
  - "nvidia/nemotron-nano-12b-v2-vl:free"
  - "google/gemma-3-12b-it:free"
  - "google/gemma-3-4b-it:free"

rate_limit:
  requests_per_minute: 15
  retry_delay_seconds: 10
  max_retries: 3
```

---

## 📁 Project Structure

```
AstroScan/
├── astroscan/
│   ├── __init__.py
│   ├── __main__.py         Entry point
│   ├── cli.py              13 CLI commands
│   ├── config.py           Configuration management
│   ├── models.py           Pydantic data models
│   ├── dewarper.py         ✨ NEW: DocScanner + geometric dewarping
│   ├── mineru_ocr.py       ✨ NEW: MinerU integration + multi-engine merge
│   ├── preprocess.py       OpenCV image preprocessing
│   ├── ocr.py              Marker + Surya OCR
│   ├── vision.py           OpenRouter Vision AI
│   ├── knowledge_base.py   ChromaDB + Knowledge Graph + structured index
│   └── pipeline.py         Full processing orchestrator
├── config.example.yaml
├── requirements.txt
├── pyproject.toml
├── install_windows.bat
└── README.md
```

---

## 🔒 Important: Book-Only Knowledge

This system is built to capture knowledge from ONE specific book only.
The knowledge base is the **single source of truth** — no conventional
astrology, no pop astrology, no external interpretations are mixed in.
Only what's in the book goes into the knowledge base.

---

*Built with ❤️ by AngoraBuilds — powered by the best open source on GitHub*
