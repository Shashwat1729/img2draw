"""Reconstruction fidelity metrics. LPIPS optional (needs lpips+torch)."""
from __future__ import annotations

import math

import cv2
import numpy as np


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(d * d))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    m = mse(a, b)
    # cap at 100 dB so reports stay JSON-serializable (inf is lossy in JSON)
    return 100.0 if m == 0 else min(100.0, 10 * math.log10(255.0 ** 2 / m))


def _ssim_channel(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_x, mu_y = cv2.GaussianBlur(x, (11, 11), 1.5), cv2.GaussianBlur(y, (11, 11), 1.5)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
    sig_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
    sig_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
    ssim = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2))
    return float(np.mean(ssim))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([_ssim_channel(a[..., c], b[..., c]) for c in range(3)]))


def ms_ssim(a: np.ndarray, b: np.ndarray, levels: int = 3) -> float:
    vals = []
    for _ in range(levels):
        vals.append(ssim(a, b))
        a = cv2.pyrDown(a)
        b = cv2.pyrDown(b)
        if min(a.shape[:2]) < 16:
            break
    return float(np.mean(vals))


def edge_difference(a: np.ndarray, b: np.ndarray) -> float:
    ea = cv2.Canny(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 50, 150)
    eb = cv2.Canny(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), 50, 150)
    return float(np.mean(np.abs(ea.astype(float) - eb.astype(float))) / 255.0)


def color_difference(a: np.ndarray, b: np.ndarray) -> float:
    la = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(float)
    lb = cv2.cvtColor(b, cv2.COLOR_RGB2LAB).astype(float)
    return float(np.mean(np.linalg.norm(la - lb, axis=-1)))


def lpips(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        import lpips as lpips_pkg
        import torch
    except Exception:
        return None
    net = lpips_pkg.LPIPS(net="alex", verbose=False)
    ta = torch.from_numpy(a.transpose(2, 0, 1)).float()[None] / 127.5 - 1
    tb = torch.from_numpy(b.transpose(2, 0, 1)).float()[None] / 127.5 - 1
    with torch.no_grad():
        return float(net(ta, tb).item())


def error_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(int) - b.astype(int)).astype(np.uint8)


def error_heatmap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    err = np.abs(a.astype(int) - b.astype(int)).sum(-1).astype(np.uint8)
    return cv2.applyColorMap(err, cv2.COLORMAP_JET)[..., ::-1]


def structural_difference(a: np.ndarray, b: np.ndarray) -> float:
    sa = cv2.Sobel(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 0)
    sb = cv2.Sobel(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), cv2.CV_64F, 1, 0)
    return float(np.mean(np.abs(sa - sb)) / 255.0)


def full_report(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    return {
        "mae": mae(original, reconstructed),
        "mse": mse(original, reconstructed),
        "psnr": psnr(original, reconstructed),
        "ssim": ssim(original, reconstructed),
        "ms_ssim": ms_ssim(original, reconstructed),
        "edge_difference": edge_difference(original, reconstructed),
        "color_difference": color_difference(original, reconstructed),
        "structural_difference": structural_difference(original, reconstructed),
        "lpips": lpips(original, reconstructed),
    }
