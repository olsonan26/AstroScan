# 🔭 AstroScan

**The ultimate book-to-knowledge-base pipeline for astrology textbooks.**

AstroScan combines the most powerful open-source OCR, layout detection, and vision AI tools into one godlike pipeline that:

1. **Preprocesses** photographed book pages (deskew, contrast, sharpening)
2. **Extracts text** with precise layout detection using [Marker](https://github.com/datalab-to/marker) (⭐23k) + [Surya](https://github.com/datalab-to/surya) (⭐14k)
3. **Extracts & analyzes charts/diagrams** using free vision AI models via OpenRouter
4. **Builds a structured knowledge base** — every concept, definition, house, sign, planet, retrograde, and rule indexed and searchable

Built specifically for scanning astrology textbooks where **every symbol, chart, and definition matters**.

---

## 🚀 Quick Start (Windows)

### Prerequisites
- Python 3.10 or higher ([Download](https://www.python.org/downloads/))
- During install, check **"Add Python to PATH"**

### Installation

```bash
# Clone the repo
git clone https://github.com/olsonan26/AstroScan.git
cd AstroScan

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy config and add your OpenRouter API key
copy config.example.yaml config.yaml
# Edit config.yaml with your settings
```

Or use the one-click installer:
```bash
install_windows.bat
```

### Configuration

Edit `config.yaml`:
```yaml
openrouter_api_key: "sk-or-v1-your-key-here"
input_dir: "./input"          # Drop your book page photos here
output_dir: "./output"        # Processed pages go here
knowledge_base_dir: "./knowledge_base"  # The brain
```

### Usage

```bash
# Process all photos in the input directory
python -m astroscan process

# Process a specific image
python -m astroscan process --file "Page_001.jpg"

# Process a range of pages
python -m astroscan process --pages 1-50

# View knowledge base stats
python -m astroscan stats

# Search the knowledge base
python -m astroscan search "Saturn retrograde"

# Export knowledge base
python -m astroscan export --format json
python -m astroscan export --format markdown
```

---

## 📦 What Powers AstroScan

| Component | Tool | Stars | Role |
|-----------|------|-------|------|
| OCR Engine | [Marker](https://github.com/datalab-to/marker) | ⭐23k | Converts page images to structured markdown with layout preservation |
| Layout Detection | [Surya](https://github.com/datalab-to/surya) | ⭐14k | Detects text regions, tables, figures, reading order (built into Marker) |
| Image Preprocessing | OpenCV + scikit-image | — | Deskew, contrast enhancement, sharpening, noise removal |
| Chart Analysis | OpenRouter Vision AI | — | Free vision models analyze natal charts, aspect grids, diagrams |
| Knowledge Base | Custom | — | Structures extracted content into indexed, searchable knowledge |

### Free Vision Models Used (via OpenRouter)
- `google/gemma-4-26b-a4b-it:free` — Best quality, 262K context
- `google/gemma-3-27b-it:free` — 27B params, 131K context
- `nvidia/nemotron-nano-12b-v2-vl:free` — Fast, reliable
- `google/gemma-3-12b-it:free` — Good balance of speed/quality
- Automatic model rotation to avoid rate limits

---

## 🧠 Knowledge Base

The knowledge base is the **brain** of this system. It's not just raw text — it's structured, categorized, and indexed:

### Structure
```
knowledge_base/
├── index.json              # Master index of all content
├── concepts/               # Core astrological concepts
│   ├── houses.json
│   ├── signs.json
│   ├── planets.json
│   ├── aspects.json
│   └── retrogrades.json
├── definitions/            # Term definitions (book-specific only)
│   └── glossary.json
├── relationships/          # How concepts connect
│   └── mappings.json
├── charts/                 # All chart/diagram descriptions
│   └── diagrams.json
├── rules/                  # Astrological rules and methods
│   └── methods.json
└── full_text/              # Complete page-by-page text
    └── book.md
```

### Key Design: Fixed Astrology Only

This knowledge base stores **only what the book teaches**. It does NOT use conventional/mainstream astrology definitions. Every concept, meaning, and rule comes exclusively from the source material.

This ensures the downstream app built from this knowledge base uses the correct "fixed astrology" system.

---

## 📁 Output Structure

Each processed page produces:
```
output/
├── page_0001/
│   ├── original.jpg          # Original photo
│   ├── preprocessed.jpg      # After deskew/contrast/sharpening
│   ├── text.md               # Extracted text (markdown)
│   ├── figures/              # Extracted images/charts
│   │   ├── figure_001.png
│   │   └── figure_002.png
│   ├── chart_analysis.md     # Vision AI analysis of diagrams
│   ├── knowledge_entries.json # Structured knowledge from this page
│   └── metadata.json         # Processing metadata
├── page_0002/
│   └── ...
└── processing_log.json       # Overall processing stats
```

---

## ⚡ Performance

- **Processing speed**: ~30-60 seconds per page (depends on content complexity)
- **Vision AI cost**: $0.00 (uses free OpenRouter models)
- **Rate limits**: Automatic model rotation handles free tier limits (~20 req/min)
- **For 1000+ pages**: Pipeline supports batch processing with automatic resume

---

## 🔧 Advanced Usage

### Resume interrupted processing
```bash
python -m astroscan process --resume
```

### Reprocess specific pages
```bash
python -m astroscan process --pages 42,56,78 --force
```

### Update knowledge base from existing output
```bash
python -m astroscan rebuild-kb
```

### Run with verbose logging
```bash
python -m astroscan process --verbose
```

---

## 📋 Requirements

- Python 3.10+
- 4GB+ RAM (for Marker/Surya models)
- ~2GB disk space (for model downloads on first run)
- Internet connection (for OpenRouter vision AI calls)
- GPU optional but recommended for faster OCR

---

## License

Private — built for AngoraBuilds.
