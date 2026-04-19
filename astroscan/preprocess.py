"""Image preprocessing for book page photos.

Combines techniques from page_dewarp (⭐2.8k) and standard CV preprocessing
to prepare photographed book pages for optimal OCR quality.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def load_image(path: str | Path) -> np.ndarray:
    """Load an image file as BGR numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    return img


def deskew(image: np.ndarray) -> np.ndarray:
    """Correct page rotation/tilt using Hough line detection.
    
    Detects dominant text line angles and rotates to straighten.
    Based on techniques from page_dewarp by mzucker.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                             minLineLength=100, maxLineGap=10)
    
    if lines is None or len(lines) == 0:
        return image
    
    # Calculate angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Only consider near-horizontal lines (text lines)
        if abs(angle) < 15:
            angles.append(angle)
    
    if not angles:
        return image
    
    # Median angle for robustness
    median_angle = np.median(angles)
    
    # Only correct if tilt is significant
    if abs(median_angle) < 0.3:
        return image
    
    # Rotate to correct
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Improves readability of text on photographed pages, especially
    with uneven lighting.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    
    # Merge and convert back
    enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def sharpen(image: np.ndarray) -> np.ndarray:
    """Sharpen text edges using unsharp masking.
    
    Makes text crisper for better OCR accuracy.
    """
    # Gaussian blur
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    # Unsharp mask: original + (original - blurred) * amount
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove camera noise while preserving text edges.
    
    Uses non-local means denoising which is excellent at
    preserving sharp text while removing photographic noise.
    """
    return cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)


def auto_crop(image: np.ndarray, margin: int = 20) -> np.ndarray:
    """Auto-crop to page content, removing dark borders/background.
    
    Useful when photos include the desk/table around the book.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold to find page
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Largest contour is likely the page
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Only crop if the detected region is large enough (>50% of image)
    img_area = image.shape[0] * image.shape[1]
    if (w * h) < (img_area * 0.5):
        return image
    
    # Add margin
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)
    
    return image[y:y+h, x:x+w]


def preprocess_page(image_path: str | Path, output_path: str | Path,
                    do_deskew: bool = True,
                    do_contrast: bool = True,
                    do_sharpen: bool = True,
                    do_denoise: bool = True,
                    do_crop: bool = False) -> Path:
    """Full preprocessing pipeline for a book page photo.
    
    Args:
        image_path: Path to input image
        output_path: Path to save preprocessed image
        do_deskew: Correct page tilt
        do_contrast: Enhance contrast
        do_sharpen: Sharpen text
        do_denoise: Remove noise
        do_crop: Auto-crop to page (disabled by default for book photos)
    
    Returns:
        Path to preprocessed image
    """
    output_path = Path(output_path)
    image = load_image(image_path)
    
    if do_crop:
        image = auto_crop(image)
    
    if do_deskew:
        image = deskew(image)
    
    if do_denoise:
        image = denoise(image)
    
    if do_contrast:
        image = enhance_contrast(image)
    
    if do_sharpen:
        image = sharpen(image)
    
    cv2.imwrite(str(output_path), image)
    return output_path
