import json
from paddleocr import PaddleOCR
from pathlib import Path

def ocr_to_json(image_path, output_json, lang="ne", confidence_threshold=0.7):
    """
    Extract text from form image and generate JSON template automatically.
    Uses PaddleOCR v3.x with Nepali ('ne') language support.
    """
    # Initialize OCR (v3.x uses different parameters)
    ocr = PaddleOCR(
        lang=lang,
        use_textline_orientation=True,  # Only use this parameter in v3.x
        use_gpu=False,            # CPU mode for compatibility
        show_log=False            # Quiet mode
    )

    # Run OCR
    results = ocr.ocr(image_path, cls=True)

    fields = []
    field_id = 1

    for line in results[0]:
        bbox, (text, conf) = line
        if conf < confidence_threshold:
            continue

        text = text.strip()
        if not text or len(text) < 2:
            continue

        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        x, y = min(x_coords), min(y_coords)
        w, h = max(x_coords) - x, max(y_coords) - y

        field_type = "text_line"
        if w < 50 and h < 50:  # Likely a grid box (e.g., PAN, ID number)
            field_type = "box_grid"

        field = {
            "id": f"f{field_id:03d}",
            "name": text,
            "label": text,
            "type": field_type,
            "page": 1,
            "bbox": {
                "px": [int(x), int(y), int(w), int(h)],
                "mm": [0, 0, 0, 0]  # Will be calculated later from DPI
            },
            "ocr": {"lang": lang, "psm": 7},
            "conf": float(conf),
            "validate": {
                "req": True,
                "type": "string",
                "min_len": 2
            }
        }
        fields.append(field)
        field_id += 1

    template = {
        "form_name": Path(image_path).stem,
        "form_type": "auto-generated",
        "version": "1.0",
        "source": "Auto Template Generator",
        "metadata": {
            "image_path": image_path,
            "dpi": 300,
            "language": lang
        },
        "fields": fields
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent="\t")

    print(f"✅ Template saved to {output_json}")
    print(f"📊 Detected {len(fields)} fields with confidence ≥ {confidence_threshold}")


if __name__ == "__main__":
    ocr_to_json(
        "./templates/business_front.jpg",
        "./templates/business_front.json",
        lang="ne",
        confidence_threshold=0.7
    )