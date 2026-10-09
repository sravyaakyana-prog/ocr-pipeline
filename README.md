# Simple OCR Pipeline (OpenCV + Tesseract)

Takes an image, cleans it up, and returns the extracted text. Built to handle
**skewed, blurred and low-light** images, which plain OCR struggles with.

## Setup
```bash
# 1. Install Tesseract (Ubuntu: sudo apt install tesseract-ocr | Mac: brew install tesseract)
# 2. Install Python packages
pip install -r requirements.txt
```

## Usage
```bash
python ocr_pipeline.py samples/sample_input.png
python ocr_pipeline.py my_image.jpg --out result.txt --debug cleaned.png
```
Or from Python:
```python
from ocr_pipeline import run_ocr
print(run_ocr("samples/sample_input.png"))   # {'text': ..., 'confidence': ...}
```

## Sample
| Input (skewed, blurred, dark, noisy) | Pipeline's cleaned image |
|---|---|
| `samples/sample_input.png` | `samples/preprocessed.png` |

Output (`samples/sample_output.txt`):
```
PathPal Demo Sign
Exit 4 — Platform 2
Caution: Wet Floor
[average confidence: 93.0%]
```

## Approach
1. **Grayscale**: colour isn't needed for text.
2. **Contrast stretch**: spreads dark (low-light) pixels across the full range.
3. **Deskew**: tries rotations from -15° to +15° and keeps the one where text
   rows line up best (highest variance of row sums).
4. **Tesseract** (`--oem 3 --psm 6`) reads the text, and word confidences are averaged.

I first tried heavier steps (upscaling, denoising, thresholding, contour-based
deskew). On my tests they made accuracy *worse*, so I removed them. Tesseract
already binarizes well, and my first deskew rotated the wrong way.

## Results
`python evaluate.py` compares raw Tesseract with this pipeline on synthetic
degraded images (similarity to ground truth, 1.0 = perfect):

| Condition | Raw Tesseract | This pipeline |
|---|---|---|
| clean | 1.0 | 1.0 |
| skewed (12°) | 0.70 | 0.92 |
| blurred + dark | 1.0 | 1.0 |
| skew + blur + dark | 0.60 | 1.00 |

## Limitations
- Tested on synthetic images with one font, not real photos, so real-world
  results will be lower.
- Deskew only handles ±15° and one dominant text direction.
- Curved or perspective-distorted text isn't corrected.
- Next steps: try PaddleOCR/EasyOCR, add text detection for scene images, and
  test on real camera images.
