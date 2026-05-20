from typing import Dict

import cv2
import numpy as np


class CellFeatureExtractor:
    MIN_SIZE = 10  # Minimum image size to process
    
    @staticmethod
    def _make_mask(cell_bgr: np.ndarray) -> np.ndarray:
        # Validate image size
        h, w = cell_bgr.shape[:2]
        if h < CellFeatureExtractor.MIN_SIZE or w < CellFeatureExtractor.MIN_SIZE:
            # Return empty mask for small images
            return np.zeros((h, w), dtype=np.uint8)
        
        try:
            gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            return mask
        except Exception as e:
            print(f"[WARN] Error creating mask: {e}")
            return np.zeros(cell_bgr.shape[:2], dtype=np.uint8)

    def extract(self, cell_bgr: np.ndarray) -> Dict[str, float]:
        mask = self._make_mask(cell_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area = 0.0
        perimeter = 0.0
        circularity = 0.0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter > 0:
                circularity = float((4.0 * np.pi * area) / (perimeter ** 2))

        mean_bgr = cv2.mean(cell_bgr, mask=mask)[:3]
        hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]

        texture_laplacian = float(cv2.Laplacian(cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())

        return {
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "mean_b": float(mean_bgr[0]),
            "mean_g": float(mean_bgr[1]),
            "mean_r": float(mean_bgr[2]),
            "mean_h": float(mean_hsv[0]),
            "mean_s": float(mean_hsv[1]),
            "mean_v": float(mean_hsv[2]),
            "texture_laplacian_var": texture_laplacian,
        }
