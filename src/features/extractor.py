from typing import Dict

import cv2
import numpy as np


class CellFeatureExtractor:
    MIN_SIZE = 10  # Kích thước hình ảnh tối thiểu để xử lý
    
    @staticmethod
    def _make_mask(cell_bgr: np.ndarray) -> np.ndarray:
        # Xác thực kích thước hình ảnh
        h, w = cell_bgr.shape[:2]
        if h < CellFeatureExtractor.MIN_SIZE or w < CellFeatureExtractor.MIN_SIZE:
            # Trả về mặt nạ rỗng cho các hình ảnh nhỏ
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

    @staticmethod
    def _calculate_eccentricity(contour: np.ndarray) -> float:
        """
        Tính độ lệch tâm (bộ mô tả hình dạng).
        Đo lường mức độ kéo dài của một hình dạng. 
        Hình tròn: ~0, Kéo dài: gần với 1
        Giúp phân biệt WBC (Bạch cầu - thường bất thường hơn) với RBC (Hồng cầu - tròn hơn)
        """
        if len(contour) < 5:
            return 0.0
        
        try:
            ellipse = cv2.fitEllipse(contour)
            (_, _), (major, minor), _ = ellipse
            
            if minor == 0:
                return 0.0
            
            # Độ lệch tâm = sqrt(1 - (trục_nhỏ/trục_lớn)^2)
            ratio = minor / major if major > 0 else 0
            eccentricity = float(np.sqrt(1 - ratio**2))
            return eccentricity
        except Exception:
            return 0.0
    
    @staticmethod
    def _calculate_solidity(contour: np.ndarray, area: float) -> float:
        """
        Tính độ đặc (độ nén/độ rắn).
        Độ đặc = diện_tích_đường_viền / diện_tích_bao_lồi
        Giá trị càng gần 1 = hình dạng càng đặc/nén
        RBC thường có độ đặc cao hơn WBC
        """
        if area == 0:
            return 0.0
        
        try:
            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            
            if hull_area == 0:
                return 0.0
            
            solidity = area / hull_area
            return float(solidity)
        except Exception:
            return 0.0
    
    @staticmethod
    def _calculate_hu_moments(contour: np.ndarray) -> tuple:
        """
        Tính toán các Mô-men Hu để khớp hình dạng.
        3 mô-men đầu tiên bất biến đối với phép tịnh tiến, thu phóng và phép quay.
        Giúp phân biệt các loại tế bào bằng đặc trưng hình dạng của chúng.
        """
        try:
            moments = cv2.HuMoments(contour)
            # Trả về 3 mô-men đầu tiên (có tính phân biệt cao nhất)
            return tuple(float(abs(m[0])) for m in moments[:3])
        except Exception:
            return (0.0, 0.0, 0.0)
    
    @staticmethod
    def _calculate_color_stats(cell_bgr: np.ndarray, mask: np.ndarray) -> dict:
        """
        Tính toán thống kê màu sắc bao gồm cả phương sai.
        WBC thường có cường độ/phương sai màu sắc khác với RBC.
        """
        if cv2.countNonZero(mask) == 0:
            return {
                "color_std_b": 0.0,
                "color_std_g": 0.0,
                "color_std_r": 0.0,
                "color_std_hsv": 0.0,
            }
        
        try:
            # Độ lệch chuẩn BGR
            b_std = float(np.std(cell_bgr[:, :, 0][mask > 0]))
            g_std = float(np.std(cell_bgr[:, :, 1][mask > 0]))
            r_std = float(np.std(cell_bgr[:, :, 2][mask > 0]))
            
            # Độ lệch chuẩn HSV
            hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
            h_std = float(np.std(hsv[:, :, 0][mask > 0]))
            
            return {
                "color_std_b": b_std,
                "color_std_g": g_std,
                "color_std_r": r_std,
                "color_std_hsv": h_std,
            }
        except Exception:
            return {
                "color_std_b": 0.0,
                "color_std_g": 0.0,
                "color_std_r": 0.0,
                "color_std_hsv": 0.0,
            }
    
    @staticmethod
    def _calculate_extent(contour: np.ndarray, area: float) -> float:
        """
        Tính toán phạm vi (diện tích đường viền / diện tích hình chữ nhật bao quanh).
        Giúp xác định mức độ đều đặn của hình dạng tế bào.
        """
        if area == 0:
            return 0.0
        
        try:
            x, y, w, h = cv2.boundingRect(contour)
            rect_area = w * h
            
            if rect_area == 0:
                return 0.0
            
            extent = area / rect_area
            return float(extent)
        except Exception:
            return 0.0

    def extract(self, cell_bgr: np.ndarray) -> Dict[str, float]:
        """
        Trích xuất tập hợp các đặc trưng toàn diện để phân loại tế bào.
        Được tăng cường với các bộ mô tả hình dạng để phân biệt tốt hơn RBC, WBC và Tiểu cầu (Platelets).
        """
        mask = self._make_mask(cell_bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        area = 0.0
        perimeter = 0.0
        circularity = 0.0
        eccentricity = 0.0
        solidity = 0.0
        extent = 0.0
        hu_m1, hu_m2, hu_m3 = 0.0, 0.0, 0.0
        
        if contours:
            contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter > 0:
                circularity = float((4.0 * np.pi * area) / (perimeter ** 2))
            
            # Các bộ mô tả hình dạng mới
            eccentricity = self._calculate_eccentricity(contour)
            solidity = self._calculate_solidity(contour, area)
            extent = self._calculate_extent(contour, area)
            hu_m1, hu_m2, hu_m3 = self._calculate_hu_moments(contour)

        mean_bgr = cv2.mean(cell_bgr, mask=mask)[:3]
        hsv = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV)
        mean_hsv = cv2.mean(hsv, mask=mask)[:3]

        texture_laplacian = float(cv2.Laplacian(cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        
        # Thống kê màu sắc
        color_stats = self._calculate_color_stats(cell_bgr, mask)

        return {
            # Các đặc trưng hình thái học ban đầu
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            
            # Các bộ mô tả hình dạng mới (giúp phân biệt WBC với RBC)
            "eccentricity": eccentricity,
            "solidity": solidity,
            "extent": extent,
            "hu_moment_1": hu_m1,
            "hu_moment_2": hu_m2,
            "hu_moment_3": hu_m3,
            
            # Các đặc trưng màu sắc ban đầu
            "mean_b": float(mean_bgr[0]),
            "mean_g": float(mean_bgr[1]),
            "mean_r": float(mean_bgr[2]),
            "mean_h": float(mean_hsv[0]),
            "mean_s": float(mean_hsv[1]),
            "mean_v": float(mean_hsv[2]),
            "texture_laplacian_var": texture_laplacian,
            
            # Các thống kê màu sắc mới
            "color_std_b": color_stats["color_std_b"],
            "color_std_g": color_stats["color_std_g"],
            "color_std_r": color_stats["color_std_r"],
            "color_std_hsv": color_stats["color_std_hsv"],
        }