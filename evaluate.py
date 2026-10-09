"""Compares raw Tesseract vs. this pipeline on synthetic degraded images."""
import difflib

import cv2
import numpy as np
import pytesseract

from ocr_pipeline import preprocess, extract_text

GT = "PathPal Demo Sign\nExit 4 - Platform 2\nCaution: Wet Floor"
CASES = {  # name: (skew deg, blur sigma, brightness, noise)
    "clean": (0, 0.1, 1.0, 0),
    "skewed (12 deg)": (12, 0.8, 0.8, 3),
    "blurred + dark": (4, 1.8, 0.35, 8),
    "skew + blur + dark": (10, 1.6, 0.35, 8),
}


def make(angle, blur, dark, noise):
    img = np.full((260, 620, 3), 235, np.uint8)
    for i, t in enumerate(GT.split("\n")):
        cv2.putText(img, t, (30, 70 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (30, 30, 30), 2, cv2.LINE_AA)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderValue=(235, 235, 235))
    img = (cv2.GaussianBlur(img, (0, 0), blur) * dark).astype(np.uint8)
    rng = np.random.default_rng(0)
    return np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)


def score(text):  # 1.0 = identical to ground truth (ignores case)
    return round(difflib.SequenceMatcher(
        None, GT.lower(), text.lower().replace("\u2014", "-")).ratio(), 2)


print(f"{'condition':22} {'raw Tesseract':>14} {'this pipeline':>14}")
for name, args in CASES.items():
    img = make(*args)
    raw = pytesseract.image_to_string(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), config="--oem 3 --psm 6").strip()
    ours, _ = extract_text(preprocess(img))
    print(f"{name:22} {score(raw):>14} {score(ours):>14}")
