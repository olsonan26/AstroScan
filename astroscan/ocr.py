"""OCR text extraction using Marker (⭐23k) + Surya (⭐14k).

Marker converts document images to structured markdown with:
- Layout detection (columns, headers, paragraphs)
- Table recognition
- Figure/image extraction
- Reading order detection
All powered by Surya under the hood.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import json
import shutil


def extract_text_and_figures(
    image_path: str | Path,
    output_dir: str | Path,
    page_number: int = 0,
) -> dict:
    """Extract text and figures from a book page image using Marker.
    
    Args:
        image_path: Path to the (preprocessed) page image
        output_dir: Directory to save extracted content
        page_number: Page number for labeling
    
    Returns:
        Dict with 'text' (markdown string), 'figures' (list of figure paths),
        and 'metadata' from Marker.
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser
    except ImportError:
        # Fallback: if marker not installed, return empty
        # The vision.py module can handle OCR as fallback
        return _fallback_ocr(image_path, output_dir)
    
    try:
        # Create converter for single image
        config_parser = ConfigParser({
            "output_format": "markdown",
            "paginate_output": False,
        })
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=create_model_dict(),
        )
        
        # Convert image
        rendered = converter(str(image_path))
        
        # Extract text
        text = rendered.markdown if hasattr(rendered, 'markdown') else str(rendered)
        
        # Save text
        text_path = output_dir / "text.md"
        text_path.write_text(text, encoding="utf-8")
        
        # Extract figures/images
        figure_paths = []
        if hasattr(rendered, 'images') and rendered.images:
            for i, (name, img) in enumerate(rendered.images.items()):
                fig_path = figures_dir / f"figure_{i+1:03d}.png"
                img.save(str(fig_path))
                figure_paths.append(str(fig_path))
        
        return {
            "text": text,
            "figures": figure_paths,
            "text_length": len(text),
            "num_figures": len(figure_paths),
        }
        
    except Exception as e:
        print(f"  ⚠ Marker processing failed: {e}")
        return _fallback_ocr(image_path, output_dir)


def _fallback_ocr(image_path: Path, output_dir: Path) -> dict:
    """Fallback when Marker is not available.
    
    Returns minimal result — vision.py will handle OCR via OpenRouter.
    """
    # Copy original to output
    dest = output_dir / "original_for_vision_ocr.jpg"
    if not dest.exists():
        shutil.copy2(image_path, dest)
    
    return {
        "text": "",
        "figures": [],
        "text_length": 0,
        "num_figures": 0,
        "fallback": True,
        "note": "Marker not installed — using vision AI for OCR instead",
    }


def extract_text_surya_only(image_path: str | Path) -> str:
    """Direct Surya OCR as alternative to full Marker pipeline.
    
    Useful for quick text extraction without layout analysis.
    """
    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor
        from PIL import Image
        
        det_predictor = DetectionPredictor()
        rec_predictor = RecognitionPredictor()
        
        image = Image.open(image_path)
        predictions = rec_predictor([image], det_predictor=det_predictor)
        
        if predictions and predictions[0].text_lines:
            return "\n".join(line.text for line in predictions[0].text_lines)
        return ""
        
    except ImportError:
        return ""
    except Exception as e:
        print(f"  ⚠ Surya OCR failed: {e}")
        return ""
