from paddleocr import PaddleOCR
import json
import os

# Initialize PaddleOCR with Devanagari support
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="devanagari",  # This is correct for Nepali (Devanagari script)
    use_gpu=False      # Set to True if you have CUDA
)

# Path to your form
image_path = "./templates/business_front.jpg"

# Perform OCR
results = ocr.ocr(image_path, cls=True)

# Save raw result for analysis
output_dir = "./data/output"
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/ocr_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ OCR completed! Results saved to ./data/output/ocr_result.json")
print("📌 Use this to auto-generate your JSON template.")