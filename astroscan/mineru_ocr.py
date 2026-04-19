"""MinerU integration for document parsing.

MinerU (⭐33.1k, OpenDataLab) is the #1 open-source document parser.
Their 1.2B model beats GPT-4o, Gemini 2.5 Pro, and Qwen2.5-VL-72B
on OmniDocBench. Handles tables, formulas, layout, images — everything.

This module wraps MinerU as an alternative/enhancement to Marker+Surya.
When MinerU is available, it provides:
- Superior layout analysis (columns, headers, footnotes)
- Better table extraction
- Formula recognition
- Image/figure extraction with captions
- Reading order detection

Falls back gracefully to Marker or Vision AI when not installed.

Install: pip install magic-pdf[full]
Docs: https://github.com/opendatalab/MinerU
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import shutil
import tempfile


def is_mineru_available() -> bool:
    """Check if MinerU is installed and ready."""
    try:
        from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
        from magic_pdf.data.dataset import PymuDocDataset
        return True
    except ImportError:
        return False


def extract_with_mineru(
    image_path: str | Path,
    output_dir: str | Path,
    page_number: int = 0,
) -> dict:
    """Extract text, tables, figures from a page image using MinerU.
    
    MinerU excels at:
    - Layout analysis (detecting text regions, tables, figures, captions)
    - Table structure recognition (rows/cols/headers)
    - Formula detection and LaTeX conversion
    - Figure extraction with associated captions
    - Multi-column reading order

    Args:
        image_path: Path to page image (jpg/png)
        output_dir: Directory for extracted content
        page_number: Page number for labeling

    Returns:
        Dict with 'text', 'tables', 'figures', 'layout_info', 'metadata'
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not is_mineru_available():
        return _not_available_result()

    try:
        return _run_mineru_pipeline(image_path, output_dir, page_number)
    except Exception as e:
        print(f"  ⚠ MinerU processing error: {e}")
        return _error_result(str(e))


def _run_mineru_pipeline(
    image_path: Path,
    output_dir: Path,
    page_number: int,
) -> dict:
    """Run the MinerU OCR pipeline on a single page image.

    Uses MinerU's auto-detect mode which handles both
    scanned images and digital documents.
    """
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze

    # MinerU works with temp directories
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        # Copy image to temp
        shutil.copy2(image_path, tmp_path / image_path.name)

        # Setup MinerU writer
        writer = FileBasedDataWriter(str(output_dir))
        image_writer = FileBasedDataWriter(str(output_dir / "figures"))

        # Read the image
        reader = FileBasedDataReader("")
        image_bytes = reader.read(str(image_path))

        # Analyze document layout
        model_json = doc_analyze(
            image_bytes,
            ocr=True,  # Force OCR mode for scanned book pages
        )

        # Process with MinerU's pipeline
        from magic_pdf.data.dataset import PymuDocDataset

        dataset = PymuDocDataset(image_bytes)
        
        # Use OCR mode (best for photographed book pages)
        pipe_result = dataset.apply(
            model_json,
            ocr=True,
            image_writer=image_writer,
        )

        # Extract markdown
        markdown_text = pipe_result.get_markdown(image_writer)

        # Extract structured content
        content_list = pipe_result.get_content_list(image_writer)

        # Parse content types
        text_blocks = []
        tables = []
        figures = []

        for block in content_list:
            block_type = block.get("type", "text")
            if block_type == "table":
                tables.append({
                    "html": block.get("html", ""),
                    "markdown": block.get("text", ""),
                    "bbox": block.get("bbox", []),
                })
            elif block_type == "image":
                fig_path = block.get("img_path", "")
                figures.append({
                    "path": fig_path,
                    "caption": block.get("caption", ""),
                    "bbox": block.get("bbox", []),
                })
            else:
                text_blocks.append(block.get("text", ""))

        # Save markdown
        md_path = output_dir / "text_mineru.md"
        md_path.write_text(markdown_text, encoding="utf-8")

        # Save structured content
        content_path = output_dir / "content_mineru.json"
        content_path.write_text(json.dumps(content_list, indent=2, ensure_ascii=False))

        return {
            "text": markdown_text,
            "text_blocks": text_blocks,
            "tables": tables,
            "figures": figures,
            "text_length": len(markdown_text),
            "num_tables": len(tables),
            "num_figures": len(figures),
            "content_blocks": len(content_list),
            "engine": "mineru",
            "model": "MinerU-2.5 (1.2B)",
        }


def extract_with_mineru_cli(
    image_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Alternative: use MinerU via CLI for simpler integration.

    Useful when the Python API has version conflicts.
    Requires: pip install magic-pdf[full]
    """
    import subprocess

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "magic-pdf",
                "-p", str(image_path),
                "-o", str(output_dir),
                "-m", "ocr",  # Force OCR mode
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            return _error_result(f"CLI error: {result.stderr[:500]}")

        # Find generated markdown
        md_files = list(output_dir.rglob("*.md"))
        if md_files:
            text = md_files[0].read_text(encoding="utf-8")
        else:
            text = ""

        # Find extracted images
        figure_paths = list(output_dir.rglob("*.png")) + list(output_dir.rglob("*.jpg"))

        return {
            "text": text,
            "text_length": len(text),
            "figures": [{"path": str(p)} for p in figure_paths],
            "num_figures": len(figure_paths),
            "engine": "mineru-cli",
        }

    except subprocess.TimeoutExpired:
        return _error_result("MinerU CLI timed out (120s)")
    except FileNotFoundError:
        return _error_result("magic-pdf CLI not found. Install: pip install magic-pdf[full]")


def merge_ocr_results(
    marker_result: dict,
    mineru_result: dict,
    vision_result: Optional[dict] = None,
) -> dict:
    """Merge results from multiple OCR engines for best accuracy.

    Strategy:
    1. Use MinerU text if available (best accuracy for complex layouts)
    2. Cross-reference with Marker for validation
    3. Use Vision AI for charts/diagrams that OCR misses
    4. Combine all figures (deduplicated)

    Args:
        marker_result: Result from Marker+Surya
        mineru_result: Result from MinerU
        vision_result: Optional result from Vision AI

    Returns:
        Merged result dict
    """
    # Pick primary text source
    mineru_text = mineru_result.get("text", "")
    marker_text = marker_result.get("text", "")
    vision_text = (vision_result or {}).get("text", "")

    # Prefer MinerU (most accurate for complex docs), then Marker, then Vision
    if mineru_text and len(mineru_text) > len(marker_text) * 0.5:
        primary_text = mineru_text
        primary_engine = "mineru"
    elif marker_text:
        primary_text = marker_text
        primary_engine = "marker"
    elif vision_text:
        primary_text = vision_text
        primary_engine = "vision"
    else:
        primary_text = ""
        primary_engine = "none"

    # Merge figures from all sources (unique by path)
    all_figures = {}
    for src in [marker_result, mineru_result, vision_result or {}]:
        for fig in src.get("figures", []):
            path = fig.get("path", "") if isinstance(fig, dict) else fig
            if path:
                all_figures[path] = fig

    # Merge tables (MinerU is better at tables)
    tables = mineru_result.get("tables", []) or marker_result.get("tables", [])

    return {
        "text": primary_text,
        "text_length": len(primary_text),
        "primary_engine": primary_engine,
        "engines_used": [e for e in ["mineru", "marker", "vision"]
                         if (e == "mineru" and mineru_text)
                         or (e == "marker" and marker_text)
                         or (e == "vision" and vision_text)],
        "figures": list(all_figures.values()),
        "num_figures": len(all_figures),
        "tables": tables,
        "num_tables": len(tables),
        "confidence": _confidence_score(mineru_text, marker_text, vision_text),
    }


def _confidence_score(
    mineru_text: str, marker_text: str, vision_text: str
) -> float:
    """Estimate OCR confidence based on cross-engine agreement.

    Higher score when multiple engines produce similar text.
    """
    texts = [t for t in [mineru_text, marker_text, vision_text] if t]

    if len(texts) <= 1:
        return 0.5  # Single engine — moderate confidence

    # Compare lengths as rough agreement metric
    lengths = [len(t) for t in texts]
    avg_len = sum(lengths) / len(lengths)

    if avg_len == 0:
        return 0.0

    # Variance in lengths — lower = more agreement
    variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)
    normalized_variance = variance / (avg_len ** 2)

    # Convert to 0-1 score (lower variance = higher confidence)
    return max(0.0, min(1.0, 1.0 - normalized_variance))


def _not_available_result() -> dict:
    return {
        "text": "",
        "text_length": 0,
        "tables": [],
        "figures": [],
        "engine": "mineru",
        "available": False,
        "error": "MinerU not installed. Install: pip install magic-pdf[full]",
    }


def _error_result(error: str) -> dict:
    return {
        "text": "",
        "text_length": 0,
        "tables": [],
        "figures": [],
        "engine": "mineru",
        "available": True,
        "error": error,
    }
