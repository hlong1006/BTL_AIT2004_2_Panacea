"""So sánh phân loại YOLO-only vs Hybrid (YOLO + ML)."""

from typing import Dict, List

from src.detection.yolo_detector import Detection


CLASS_NAMES = ["Platelets", "RBC", "WBC"]


def compare_yolo_vs_hybrid(
    detections: List[Detection],
    ml_labels: List[str],
) -> Dict:
    """So sánh nhãn YOLO gốc với nhãn sau ML refinement."""
    yolo_labels = [det.class_name for det in detections]
    n = min(len(yolo_labels), len(ml_labels))

    agreements = sum(1 for i in range(n) if yolo_labels[i] == ml_labels[i])
    disagreements = n - agreements

    changes = []
    for i in range(n):
        if yolo_labels[i] != ml_labels[i]:
            changes.append({
                "cell_index": i,
                "yolo": yolo_labels[i],
                "hybrid": ml_labels[i],
                "yolo_confidence": round(detections[i].confidence, 4),
            })

    yolo_counts: Dict[str, int] = {}
    hybrid_counts: Dict[str, int] = {}
    for y, h in zip(yolo_labels[:n], ml_labels[:n]):
        yolo_counts[y] = yolo_counts.get(y, 0) + 1
        hybrid_counts[h] = hybrid_counts.get(h, 0) + 1

    agreement_rate = (agreements / n * 100) if n > 0 else 0.0

    return {
        "total_cells": n,
        "agreements": agreements,
        "disagreements": disagreements,
        "agreement_rate_pct": round(agreement_rate, 2),
        "yolo_counts": yolo_counts,
        "hybrid_counts": hybrid_counts,
        "changes": changes[:20],
        "summary": (
            f"YOLO-only và Hybrid đồng ý {agreements}/{n} tế bào ({agreement_rate:.1f}%). "
            f"ML thay đổi {disagreements} nhãn."
            if n > 0 else "Không có tế bào để so sánh."
        ),
    }
