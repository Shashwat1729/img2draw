"""Generate diverse synthetic benchmark images (no external dataset needed)."""
import sys
from pathlib import Path

import cv2
import numpy as np

rng = np.random.default_rng(0)
out = Path(sys.argv[1] if len(sys.argv) > 1 else "examples")
out.mkdir(parents=True, exist_ok=True)

# 1. flat logo
img = np.full((128, 128, 3), 250, np.uint8)
cv2.circle(img, (64, 64), 40, (200, 30, 30), -1)
cv2.rectangle(img, (40, 50), (88, 78), (30, 30, 200), -1)
cv2.imwrite(str(out / "logo.png"), img[..., ::-1])

# 2. gradient landscape
yy, xx = np.mgrid[0:128, 0:128]
img = np.stack([yy * 2 % 256, (xx + yy) % 256, 255 - yy * 2 % 256], -1).astype(np.uint8)
cv2.imwrite(str(out / "gradient.png"), img[..., ::-1])

# 3. shaded illustration
img = np.full((128, 128, 3), (180, 220, 240), np.uint8)
cv2.circle(img, (64, 80), 30, (240, 200, 160), -1)
cv2.circle(img, (54, 75), 4, (20, 20, 20), -1)
cv2.circle(img, (74, 75), 4, (20, 20, 20), -1)
cv2.ellipse(img, (64, 95), (10, 5), 0, 0, 180, (150, 80, 80), 2)
cv2.imwrite(str(out / "face.png"), img[..., ::-1])

# 4. noisy photo-like
img = rng.normal(128, 40, (128, 128, 3)).clip(0, 255).astype(np.uint8)
img = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imwrite(str(out / "noisy.png"), img[..., ::-1])

# 5. hard edges / stripes
img = np.zeros((128, 128, 3), np.uint8)
img[:, ::16] = (255, 255, 255)
img[::16, :] = (200, 50, 50)
cv2.imwrite(str(out / "stripes.png"), img[..., ::-1])

# 6. soft shading sphere
yy, xx = np.mgrid[0:128, 0:128]
d = np.sqrt((xx - 64) ** 2 + (yy - 64) ** 2) / 50
z = np.sqrt(np.clip(1 - d ** 2, 0, 1))
img = np.stack([z * 255, z * z * 255, np.full_like(z, 128)], -1).astype(np.uint8)
cv2.imwrite(str(out / "sphere.png"), img[..., ::-1])

print(f"wrote {len(list(out.glob('*.png')))} images to {out}")
