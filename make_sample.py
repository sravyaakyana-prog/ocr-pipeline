"""Creates a hard sample image: skewed, blurred, low-light, noisy."""
import cv2
import numpy as np

lines = ["PathPal Demo Sign", "Exit 4 - Platform 2", "Caution: Wet Floor"]
img = np.full((260, 620, 3), 235, np.uint8)
for i, t in enumerate(lines):
    cv2.putText(img, t, (30, 70 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (30, 30, 30), 2, cv2.LINE_AA)
h, w = img.shape[:2]
M = cv2.getRotationMatrix2D((w / 2, h / 2), 10, 1.0)          # skew
img = cv2.warpAffine(img, M, (w, h), borderValue=(235, 235, 235))
img = cv2.GaussianBlur(img, (0, 0), 1.6)                      # blur
img = (img * 0.35).astype(np.uint8)                           # low light
noise = np.random.default_rng(0).normal(0, 8, img.shape)
img = np.clip(img + noise, 0, 255).astype(np.uint8)           # noise
cv2.imwrite("samples/sample_input.png", img)
