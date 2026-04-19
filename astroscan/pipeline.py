"""Full AstroScan processing pipeline.

Orchestrates: Dewarp → Preprocess → OCR (MinerU + Marker) → Vision Analysis → Knowledge Base

v3.0 Pipeline:
1. Dewarp (DocScanner/geometric correction for curved book pages)
2. Preprocess (deskew, contrast, sharpen, denoise)
3. OCR Pass A: MinerU (33k⭐, beats GPT-4o at document understanding)
4. OCR Pass B: Marker + Surya (23k⭐ + 14k⭐, structured markdown)
5. Merge OCR results (cross-engine validation for best accuracy)
6. Vision AI chart/diagram analysis
7. Knowledge extraction → ChromaDB + Knowledge Graph
"""
from __future__ import annotations
from pathlib import Path
import asyncio
import json
import shutil
import time
import re
from typing import Optional

from astroscan.config import Config
from astroscan.preprocess import preprocess_page
from astroscan.ocr import extract_text_and_figures
from astroscan.dewarper import dewarp_page, estimate_curvature
from astroscan.mineru_ocr import extract_with_mineru, merge_ocr_results, is_mineru_available
from astroscan.vision import VisionAI
from astroscan.knowledge_base import KnowledgeBaseManager
from astroscan.models import PageMetadata, ChartDescription


# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _extract_page_number(filename: str) -> int:
    """Extract page number from filename like 'Page 042' or 'page_042.jpg'."""
    # Try common patterns
    patterns = [
        r'[Pp]age\s*[_-]?\s*(\d+)',   # Page 042, page_042, Page-42
        r'^(\d+)',                       # 042.jpg
        r'[_-](\d+)\.',                 # something_042.jpg
    ]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    return 0


def list_input_images(input_dir: Path) -> list[tuple[Path, int]]:
    """List all images in input directory, sorted by page number.
    
    Returns:
        List of (image_path, page_number) tuples, sorted by page number.
    """
    images = []
    for f in sorted(input_dir.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
            page_num = _extract_page_number(f.stem)
            images.append((f, page_num))
    
    # Sort by page number (0 = unknown goes to end)
    images.sort(key=lambda x: (x[1] == 0, x[1]))
    return images


def _get_processed_pages(output_dir: Path) -> set[int]:
    """Get set of already-processed page numbers."""
    processed = set()
    if not output_dir.exists():
        return processed
    
    for d in output_dir.iterdir():
        if d.is_dir() and d.name.startswith("page_"):
            meta_path = d / "metadata.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    processed.add(meta.get("page_number", 0))
                except Exception:
                    pass
    return processed


async def process_single_page(
    image_path: Path,
    page_number: int,
    config: Config,
    vision_ai: VisionAI,
    kb_manager: KnowledgeBaseManager,
    force: bool = False,
) -> PageMetadata:
    """Process a single book page through the full pipeline.
    
    Pipeline:
    1. Preprocess (deskew, contrast, sharpen, denoise)
    2. OCR via Marker (text + figure extraction)
    3. If Marker unavailable → OCR via Vision AI
    4. Vision AI chart/diagram analysis
    5. Knowledge extraction and indexing
    
    Args:
        image_path: Path to the page image
        page_number: Page number in the book
        config: Configuration
        vision_ai: VisionAI instance
        kb_manager: Knowledge base manager
        force: Reprocess even if already done
    
    Returns:
        PageMetadata with processing results
    """
    start_time = time.time()
    page_dir = config.output_dir / f"page_{page_number:04d}"
    
    # Check if already processed
    meta_path = page_dir / "metadata.json"
    if meta_path.exists() and not force:
        print(f"  ⏭ Page {page_number} already processed (use --force to reprocess)")
        return PageMetadata(**json.loads(meta_path.read_text()))
    
    page_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"📄 Processing Page {page_number}: {image_path.name}")
    print(f"   v3.0 Pipeline: Dewarp → Preprocess → MinerU + Marker → Vision → KB")
    print(f"{'='*60}")
    
    # Copy original
    original_dest = page_dir / f"original{image_path.suffix}"
    if not original_dest.exists():
        shutil.copy2(image_path, original_dest)
    
    # Get image dimensions
    from PIL import Image
    img = Image.open(image_path)
    width, height = img.size
    file_size = image_path.stat().st_size
    
    # ── Step 0: Curvature Detection + Dewarping ─────────────────
    print("\n📐 Step 0: Curvature detection...")
    curvature = estimate_curvature(image_path)
    dewarped_path = page_dir / "dewarped.jpg"
    
    if curvature["needs_dewarping"]:
        print(f"  📐 Curvature detected (score: {curvature['curvature_score']:.2f})")
        try:
            dewarp_page(image_path, dewarped_path, method=curvature["estimated_method"])
            print(f"  ✅ Dewarped using {curvature['estimated_method']} method")
            preprocess_input = dewarped_path
        except Exception as e:
            print(f"  ⚠ Dewarping failed ({e}), using original")
            preprocess_input = image_path
    else:
        print(f"  ✅ Page is flat (score: {curvature['curvature_score']:.2f}), skipping dewarp")
        preprocess_input = image_path
    
    # ── Step 1: Preprocess ──────────────────────────────────────
    print("\n🔧 Step 1: Preprocessing...")
    preprocessed_path = page_dir / "preprocessed.jpg"
    try:
        pp = config.preprocessing
        preprocess_page(
            preprocess_input, preprocessed_path,
            do_deskew=pp.get("deskew", True),
            do_contrast=pp.get("enhance_contrast", True),
            do_sharpen=pp.get("sharpen", True),
            do_denoise=pp.get("denoise", True),
        )
        print("  ✅ Preprocessed (deskew + contrast + sharpen + denoise)")
        ocr_input = preprocessed_path
    except Exception as e:
        print(f"  ⚠ Preprocessing failed ({e}), using original")
        ocr_input = preprocess_input
    
    # ── Step 2a: OCR via MinerU (primary — best accuracy) ───────
    mineru_result = {}
    if is_mineru_available():
        print("\n🏆 Step 2a: MinerU OCR (beats GPT-4o at doc understanding)...")
        try:
            mineru_result = extract_with_mineru(ocr_input, page_dir, page_number)
            if mineru_result.get("text"):
                print(f"  ✅ MinerU: {mineru_result['text_length']} chars, "
                      f"{mineru_result.get('num_tables', 0)} tables, "
                      f"{mineru_result.get('num_figures', 0)} figures")
            else:
                print(f"  ⚠ MinerU returned no text: {mineru_result.get('error', '')}")
        except Exception as e:
            print(f"  ⚠ MinerU error: {e}")
    else:
        print("\n  ℹ MinerU not installed (install: pip install magic-pdf[full])")
    
    # ── Step 2b: OCR via Marker + Surya (secondary) ─────────────
    print("\n📖 Step 2b: Marker + Surya OCR...")
    marker_result = extract_text_and_figures(ocr_input, page_dir, page_number)
    marker_text = marker_result.get("text", "")
    figures = marker_result.get("figures", [])
    
    if marker_text:
        print(f"  ✅ Marker: {len(marker_text)} chars, {len(figures)} figures")
    
    # ── Step 2c: Merge OCR results ──────────────────────────────
    print("\n🔀 Step 2c: Merging OCR engines...")
    if mineru_result.get("text") or marker_text:
        merged = merge_ocr_results(marker_result, mineru_result)
        text = merged["text"]
        figures = merged.get("figures", figures)
        ocr_model = f"merged ({merged['primary_engine']}, confidence: {merged.get('confidence', 0):.2f})"
        print(f"  ✅ Primary engine: {merged['primary_engine']} | "
              f"Confidence: {merged.get('confidence', 0):.2f} | "
              f"Engines: {', '.join(merged.get('engines_used', []))}")
    else:
        text = ""
        ocr_model = "none"
    
    # ── Step 3: Fallback OCR via Vision AI ──────────────────────
    if not text:
        print("\n📝 Step 3: Vision AI OCR (fallback)...")
        text, ocr_model = await vision_ai.extract_text(ocr_input)
        if text:
            (page_dir / "text.md").write_text(text, encoding="utf-8")
            # Also try merging with any partial results
            if mineru_result.get("text") or marker_text:
                merged = merge_ocr_results(
                    marker_result, mineru_result,
                    {"text": text, "figures": []},
                )
                text = merged["text"]
                ocr_model = f"merged+vision ({merged['primary_engine']})"
    else:
        print(f"\n  ✅ Combined OCR: {len(text)} chars")
    
    # ── Step 4: Chart/Diagram Analysis ──────────────────────────
    print("\n🔭 Step 4: Chart/diagram analysis...")
    chart_analysis, vision_model = await vision_ai.analyze_charts(ocr_input)
    
    has_charts = bool(chart_analysis)
    if chart_analysis:
        (page_dir / "chart_analysis.md").write_text(chart_analysis, encoding="utf-8")
        
        # Add to knowledge base as chart description
        chart = ChartDescription(
            page_number=page_number,
            description=chart_analysis,
            raw_analysis=chart_analysis,
        )
        kb_manager.add_chart(chart)
    
    # ── Step 5: Knowledge Extraction ────────────────────────────
    print("\n🧠 Step 5: Knowledge extraction...")
    if text:
        kb_json, _ = await vision_ai.extract_knowledge(text, page_number)
        entries_added = kb_manager.add_entries_from_json(kb_json, page_number)
        print(f"  ✅ {entries_added} knowledge entries added")
        
        # Add full page text to book
        kb_manager.add_page_text(page_number, text)
        
        # Save knowledge entries for this page
        (page_dir / "knowledge_entries.json").write_text(
            kb_json, encoding="utf-8"
        )
    
    # ── Save metadata ───────────────────────────────────────────
    elapsed = time.time() - start_time
    metadata = PageMetadata(
        page_number=page_number,
        original_filename=image_path.name,
        image_width=width,
        image_height=height,
        file_size_bytes=file_size,
        text_length=len(text),
        num_figures=len(figures),
        has_charts=has_charts,
        ocr_model=ocr_model,
        vision_model=vision_model,
        processing_time_seconds=round(elapsed, 1),
    )
    meta_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
    
    # Save knowledge base after each page
    kb_manager.save()
    
    print(f"\n{'─'*50}")
    print(f"✅ Page {page_number} complete in {elapsed:.1f}s")
    print(f"   Text: {len(text)} chars | Figures: {len(figures)} | Charts: {'Yes' if has_charts else 'No'}")
    print(f"{'─'*50}")
    
    return metadata


async def process_batch(
    config: Config,
    pages: Optional[list[int]] = None,
    force: bool = False,
    resume: bool = True,
) -> dict:
    """Process all images in the input directory.
    
    Args:
        config: Configuration
        pages: Specific page numbers to process (None = all)
        force: Reprocess already-completed pages
        resume: Skip already-processed pages
    
    Returns:
        Processing summary dict
    """
    config.ensure_dirs()
    
    # Find all images
    all_images = list_input_images(config.input_dir)
    if not all_images:
        print("❌ No images found in input directory!")
        print(f"   Drop book page photos in: {config.input_dir.absolute()}")
        return {"error": "No images found"}
    
    # Filter to requested pages
    if pages:
        all_images = [(p, n) for p, n in all_images if n in pages]
    
    # Skip already processed (unless --force)
    if resume and not force:
        processed = _get_processed_pages(config.output_dir)
        all_images = [(p, n) for p, n in all_images if n not in processed or n == 0]
    
    total = len(all_images)
    if total == 0:
        print("✅ All pages already processed!")
        return {"processed": 0, "skipped": "all"}
    
    print(f"\n🚀 Processing {total} pages...")
    print(f"   Input:  {config.input_dir.absolute()}")
    print(f"   Output: {config.output_dir.absolute()}")
    print(f"   KB:     {config.knowledge_base_dir.absolute()}")
    
    # Initialize components
    vision_ai = VisionAI(config)
    kb_manager = KnowledgeBaseManager(config)
    
    # Process each page
    results = []
    start = time.time()
    
    for i, (image_path, page_number) in enumerate(all_images):
        # Auto-assign page numbers if not detected
        if page_number == 0:
            page_number = i + 1
        
        try:
            meta = await process_single_page(
                image_path, page_number, config, vision_ai, kb_manager, force
            )
            results.append(meta)
        except Exception as e:
            print(f"\n❌ Error processing {image_path.name}: {e}")
            results.append(None)
        
        # Progress
        pct = (i + 1) / total * 100
        elapsed = time.time() - start
        avg_per_page = elapsed / (i + 1)
        remaining = avg_per_page * (total - i - 1)
        print(f"\n📊 Progress: {i+1}/{total} ({pct:.0f}%) — ~{remaining/60:.0f}m remaining")
    
    # Final save
    kb_manager.save()
    
    # Summary
    total_time = time.time() - start
    successful = sum(1 for r in results if r is not None)
    total_text = sum(r.text_length for r in results if r)
    total_charts = sum(1 for r in results if r and r.has_charts)
    stats = kb_manager.get_stats()
    
    summary = {
        "pages_processed": successful,
        "pages_failed": total - successful,
        "total_text_chars": total_text,
        "pages_with_charts": total_charts,
        "knowledge_entries": stats["total_entries"],
        "total_time_seconds": round(total_time, 1),
        "avg_time_per_page": round(total_time / max(successful, 1), 1),
    }
    
    # Save processing log
    log_path = config.output_dir / "processing_log.json"
    log_path.write_text(json.dumps(summary, indent=2))
    
    print(f"\n{'='*60}")
    print(f"🎉 BATCH PROCESSING COMPLETE!")
    print(f"   Pages processed: {successful}/{total}")
    print(f"   Total text: {total_text:,} characters")
    print(f"   Charts found: {total_charts}")
    print(f"   Knowledge entries: {stats['total_entries']}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    return summary
