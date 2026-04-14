"""
face_recognizer.py
------------------
Pure-NumPy Local Binary Pattern Histogram (LBPH) face recogniser.
Provides the same train / predict / read / write interface as
cv2.face.LBPHFaceRecognizer so the rest of the code works unchanged
whether or not OpenCV-contrib is available.
"""

import os
import pickle
import numpy as np
import cv2


# ── LBP helpers ──────────────────────────────────────────────────────────────

def _lbp_image(gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """Return the LBP-coded image for a single-channel (gray) image."""
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    angles = [2 * np.pi * i / n_points for i in range(n_points)]
    for i, angle in enumerate(angles):
        dx = radius * np.cos(angle)
        dy = -radius * np.sin(angle)
        # Bilinear sampling of neighbour
        x0 = int(np.floor(dx))
        x1 = x0 + 1
        y0 = int(np.floor(dy))
        y1 = y0 + 1
        # Weights
        wx = dx - x0
        wy = dy - y0
        # Shift the image for each corner and blend
        def shift(img, sx, sy):
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            return cv2.warpAffine(img, M, (w, h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)

        neighbour = ((1 - wx) * (1 - wy) * shift(gray, x0, y0) +
                     wx       * (1 - wy) * shift(gray, x1, y0) +
                     (1 - wx) * wy       * shift(gray, x0, y1) +
                     wx       * wy       * shift(gray, x1, y1))
        lbp |= ((neighbour.astype(np.float32) >= gray.astype(np.float32))
                .astype(np.uint8) << i)
    return lbp


def _lbp_histogram(gray: np.ndarray,
                   grid_x: int = 8,
                   grid_y: int = 8,
                   radius: int = 1,
                   n_points: int = 8) -> np.ndarray:
    """Compute concatenated grid-of-cell LBP histograms for a face image."""
    lbp = _lbp_image(gray, radius, n_points)
    h, w = lbp.shape
    cy = h // grid_y
    cx = w // grid_x
    hist_parts = []
    for gy in range(grid_y):
        for gx in range(grid_x):
            cell = lbp[gy * cy:(gy + 1) * cy, gx * cx:(gx + 1) * cx]
            hist, _ = np.histogram(cell.ravel(), bins=256, range=(0, 256))
            # L1-normalise each cell
            s = hist.sum()
            hist = hist / (s + 1e-7)
            hist_parts.append(hist)
    return np.concatenate(hist_parts).astype(np.float32)


# ── LBPHFaceRecognizer class ──────────────────────────────────────────────────

class LBPHFaceRecognizer:
    """
    Minimal LBPH face recogniser compatible with the cv2.face API.

    Usage
    -----
    rec = LBPHFaceRecognizer(threshold=800)
    rec.train(images, labels)         # images: list of gray np arrays, labels: list of int
    label, confidence = rec.predict(face_roi)
    rec.write("trainer/trainer.pkl")
    rec.read("trainer/trainer.pkl")
    """

    def __init__(self, radius: int = 1, neighbors: int = 8,
                 grid_x: int = 8, grid_y: int = 8,
                 threshold: float = 800.0):
        self.radius    = radius
        self.neighbors = neighbors
        self.grid_x    = grid_x
        self.grid_y    = grid_y
        self.threshold = threshold
        self._histograms: list  = []   # list of 1-D np.float32 arrays
        self._labels: list      = []   # list of int labels

    # ------------------------------------------------------------------
    def train(self, images, labels):
        """
        Train on a list of grayscale face images.
        Can be called multiple times – new samples are *appended*.
        """
        for img, lbl in zip(images, labels):
            gray = self._preprocess(img)
            h = _lbp_histogram(gray, self.grid_x, self.grid_y,
                               self.radius, self.neighbors)
            self._histograms.append(h)
            self._labels.append(int(lbl))

    # ------------------------------------------------------------------
    def update(self, images, labels):
        """Alias for train (matches cv2.face API)."""
        self.train(images, labels)

    # ------------------------------------------------------------------
    def predict(self, img):
        """
        Return (predicted_label, confidence).
        Confidence is chi-squared distance (lower = more certain).
        Returns (-1, 100.0) when no model is loaded or no match within threshold.
        """
        if not self._histograms:
            return -1, 100.0

        gray = self._preprocess(img)
        query = _lbp_histogram(gray, self.grid_x, self.grid_y,
                               self.radius, self.neighbors)

        best_label = -1
        best_dist  = float("inf")

        for h, lbl in zip(self._histograms, self._labels):
            # Chi-squared distance
            num  = (h - query) ** 2
            den  = h + query + 1e-7
            dist = float(np.sum(num / den))
            if dist < best_dist:
                best_dist  = dist
                best_label = lbl

        # Map chi-squared distance to a 0-100 "confidence" score
        # so callers can apply the same threshold < 85 logic.
        confidence = best_dist * 100.0   # raw chi-sq × 100; no artificial cap

        if confidence > self.threshold:
            return -1, confidence
        return best_label, confidence

    # ------------------------------------------------------------------
    def write(self, path: str):
        """Serialise the model to a pickle file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        payload = {
            "radius":     self.radius,
            "neighbors":  self.neighbors,
            "grid_x":     self.grid_x,
            "grid_y":     self.grid_y,
            "threshold":  self.threshold,
            "histograms": self._histograms,
            "labels":     self._labels,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    # ------------------------------------------------------------------
    def read(self, path: str):
        """Load a previously saved model."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.radius      = payload["radius"]
        self.neighbors   = payload["neighbors"]
        self.grid_x      = payload["grid_x"]
        self.grid_y      = payload["grid_y"]
        self.threshold   = payload["threshold"]
        self._histograms = payload["histograms"]
        self._labels     = payload["labels"]

    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        """Convert to grayscale if needed, resize to 100×100, equalise."""
        if img is None:
            raise ValueError("Received None image")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (100, 100))
        # NOTE: equalization is applied ONCE in the caller (capture / engine)
        #       before the ROI reaches here.  Do NOT equalise again.
        return img

    # ------------------------------------------------------------------
    def getHistograms(self):
        return self._histograms

    def getLabels(self):
        return self._labels


# ── Factory (mirrors cv2.face.LBPHFaceRecognizer_create) ─────────────────────

def LBPHFaceRecognizer_create(radius=1, neighbors=8,
                              grid_x=8, grid_y=8,
                              threshold=800.0) -> LBPHFaceRecognizer:
    """
    Drop-in replacement for cv2.face.LBPHFaceRecognizer_create().
    First tries the real OpenCV implementation; falls back to the
    pure-NumPy version if not available.
    """
    try:
        rec = cv2.face.LBPHFaceRecognizer_create(
            radius, neighbors, grid_x, grid_y, threshold)
        return rec
    except AttributeError:
        pass
    return LBPHFaceRecognizer(radius, neighbors, grid_x, grid_y, threshold)
