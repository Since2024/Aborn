import json
from paddleocr import PaddleOCR

def ocr_to_json(image_path, output_json, lang="nep", dpi=300):
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    results = ocr.ocr(image_path, cls=True)

    fields = []
    field_id = 1

    # Conversion factor px → mm (depends on scan DPI)
    pixel_to_mm = 25.4 / dpi  

    for line in results[0]:
        bbox, (text, conf) = line
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        x, y = min(x_coords), min(y_coords)
        w, h = max(x_coords) - x, max(y_coords) - y

        field = {
            "id": f"f{field_id:03d}",
            "name": text.strip(),
            "label": text.strip(),
            "type": "text_line",
            "page": 1,
            "bbox": {
                "px": [int(x), int(y), int(w), int(h)],
                "mm": [round(x*pixel_to_mm,2), round(y*pixel_to_mm,2), round(w*pixel_to_mm,2), round(h*pixel_to_mm,2)]
            },
            "ocr": {"lang": lang, "psm": 7},
            "conf": float(conf)
        }
        fields.append(field)
        field_id += 1

    template = {
        "form_name": image_path.split("/")[-1],
        "version": "auto-1.0",
        "fields": fields
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    print(f"✅ Template saved to {output_json}")


# Example usage
if __name__ == "__main__":
    ocr_to_json("./templates/business_front.jpg", "./templates/business_front.json", lang="nep")
