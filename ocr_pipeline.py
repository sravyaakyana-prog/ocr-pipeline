"""Small OCR pipeline: image -> preprocess -> Tesseract -> text."""
import argparse
import sys

import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def deskew(gray, max_angle=15, step=0.5):
    """Straighten text by trying small rotations and keeping the one where
    text lines are most horizontal (highest variance of row sums)."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5) if max(gray.shape) > 1000 else gray
    blur = cv2.GaussianBlur(255 - small, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = bw.shape
    best_angle, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + step, step):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rot = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST)
        score = np.var(rot.sum(axis=1).astype(float))
        if score > best_score:
            best_angle, best_score = a, score
    if abs(best_angle) < 0.3:
        return gray
    H, W = gray.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), best_angle, 1.0)
    return cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess(img):
    """Grayscale -> stretch contrast (low light) -> deskew."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return deskew(gray)


def extract_text(processed, lang="eng"):
    """Run Tesseract; return cleaned text and average word confidence."""
    cfg = "--oem 3 --psm 6"
    text = pytesseract.image_to_string(processed, lang=lang, config=cfg).strip()
    data = pytesseract.image_to_data(processed, lang=lang, config=cfg,
                                     output_type=pytesseract.Output.DICT)
    confs = [float(c) for c, t in zip(data["conf"], data["text"])
             if t.strip() and float(c) >= 0]
    return text, (sum(confs) / len(confs) if confs else 0.0)


def run_ocr(path, lang="eng", save_debug=None):
    """Full pipeline. Returns {'text': str, 'confidence': float}."""
    img = load_image(path)
    processed = preprocess(img)
    if save_debug:
        cv2.imwrite(save_debug, processed)
    text, conf = extract_text(processed, lang)
    return {"text": text, "confidence": round(conf, 1)}


def main():
    p = argparse.ArgumentParser(description="Simple OCR pipeline")
    p.add_argument("image", help="path to input image")
    p.add_argument("--lang", default="eng", help="Tesseract language (default: eng)")
    p.add_argument("--debug", help="save the preprocessed image to this path")
    p.add_argument("--out", help="save extracted text to this file")
    args = p.parse_args()
    try:
        result = run_ocr(args.image, args.lang, args.debug)
    except FileNotFoundError as e:
        sys.exit(str(e))
    print(result["text"])
    print(f"\n[average confidence: {result['confidence']}%]")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(result["text"] + "\n")


if __name__ == "__main__":
    main()
