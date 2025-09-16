import json
import cv2
import os
from pathlib import Path
from typing import Dict, Any
from paddleocr import PaddleOCR

def preprocess_roi(roi):
    """Preprocess ROI for better OCR results"""
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Binarization
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    return denoised

def extract_data_from_image(image_path: str, template_path: str, lang: str = "devanagari", confidence_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Extract structured data from an image based on a JSON template.
    
    Args:
        image_path (str): Path to form image.
        template_path (str): Path to JSON template with field definitions.
        lang (str): OCR language ("devanagari" for Nepali).
        confidence_threshold (float): Minimum confidence to accept OCR result.
    
    Returns:
        Dict[str, Any]: Mapping of field_id -> extracted text with metadata.
    """
    # Load template
    with open(template_path, "r", encoding="utf-8") as f:
        template = json.load(f)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Initialize OCR
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=False)

    extracted_data = {}
    dpi = template.get("metadata", {}).get("dpi", 300)
    pixel_to_mm = 25.4 / dpi

    for field in template.get("fields", []):
        field_id = field.get("id")
        field_name = field.get("name")
        bbox = field.get("bbox")
        
        if not bbox:
            print(f"⚠️ Skipping {field_name}: No bounding box provided")
            continue

        # Convert mm to pixels if needed
        if "mm" in bbox:
            # Convert mm to px using dpi
            px_bbox = [
                round(bbox["mm"][0] * pixel_to_mm),
                round(bbox["mm"][1] * pixel_to_mm),
                round(bbox["mm"][2] * pixel_to_mm),
                round(bbox["mm"][3] * pixel_to_mm)
            ]
        else:
            px_bbox = bbox["px"]
        
        x, y, w, h = px_bbox
        # Ensure coordinates are valid
        x, y = max(0, x), max(0, y)
        w, h = min(w, image.shape[1]-x), min(h, image.shape[0]-y)
        
        # Crop ROI
        roi = image[y:y+h, x:x+w]
        
        # Preprocess for better OCR results
        roi = preprocess_roi(roi)
        
        # Run OCR on ROI
        results = ocr.ocr(roi, cls=True)
        
        text = ""
        conf = 0.0
        if results and results[0]:
            # Take best line
            _, (txt, score) = results[0][0]
            text, conf = txt.strip(), float(score)
        
        # Skip low-confidence results
        if conf < confidence_threshold:
            print(f"⚠️ Low confidence for {field_name}: {conf:.2f} < {confidence_threshold}")
            continue
            
        # Field-specific processing
        if field.get("type") == "box_grid":
            # For box grids, split into sub-boxes and run OCR on each
            # This is a simplified version - for real implementation, you'd need to detect individual boxes
            pass
            
        extracted_data[field_id] = {
            "text": text,
            "confidence": conf,
            "original_bbox": bbox,
            "processed_bbox": [x, y, w, h],
            "field_type": field.get("type", "text_line")
        }

    return extracted_data


if __name__ == "__main__":
    # Example usage (replace with real paths)
    img = "./templates/business_front.jpg"
    template = "./templates/business_front.json"

    data = extract_data_from_image(img, template, lang="devanagari", confidence_threshold=0.7)
    print("✅ Extracted Data:")
    print(json.dumps(data, ensure_ascii=False, indent=2))