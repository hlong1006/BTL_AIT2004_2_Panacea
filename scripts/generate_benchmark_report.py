#!/usr/bin/env python3
"""Sinh báo cáo benchmark cho slide bảo vệ BTL."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classification.ml_classifier import MLClassifier
from src.config.settings import PATHS
from src.detection.yolo_detector import YoloDetector
from src.pipeline.end_to_end import HybridCellPipeline


def main():
    output_dir = PATHS.root / "outputs" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_path = PATHS.yolo_models / "best.pt"
    ml_path = PATHS.ml_models / "best_ml_model.pt"
    report = {
        "yolo_model": str(yolo_path),
        "ml_model": str(ml_path),
        "yolo_defaults": {
            "conf": YoloDetector.DEFAULT_CONF,
            "iou": YoloDetector.DEFAULT_IOU,
            "imgsz": YoloDetector.DEFAULT_IMGSZ,
        },
        "test_images_analyzed": 0,
        "aggregate": {"total_cells": 0, "hybrid_agreement_pct": []},
        "per_image": [],
    }

    if not yolo_path.exists() or not ml_path.exists():
        report["error"] = "Model files missing"
        out = output_dir / "benchmark_report.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Models missing. Wrote stub to {out}")
        return

    pipeline = HybridCellPipeline(yolo_path, ml_path)
    test_images = list(PATHS.test_images.glob("*.jpg"))[:10]

    for img_path in test_images:
        out_img = output_dir / f"{img_path.stem}_bench.png"
        try:
            result = pipeline.run_on_image_full(img_path, out_img, output_dir / img_path.stem)
            entry = {
                "image": img_path.name,
                "total_cells": result.stats.total_cells,
                "cell_counts": result.stats.cell_counts,
                "hybrid_agreement": result.hybrid_comparison.get("agreement_rate_pct") if result.hybrid_comparison else None,
                "ground_truth_f1": result.ground_truth_metrics.get("f1_score") if result.ground_truth_metrics else None,
            }
            report["per_image"].append(entry)
            report["test_images_analyzed"] += 1
            report["aggregate"]["total_cells"] += result.stats.total_cells
            if entry["hybrid_agreement"] is not None:
                report["aggregate"]["hybrid_agreement_pct"].append(entry["hybrid_agreement"])
        except Exception as exc:
            report["per_image"].append({"image": img_path.name, "error": str(exc)})

    agreements = report["aggregate"]["hybrid_agreement_pct"]
    if agreements:
        report["aggregate"]["avg_hybrid_agreement_pct"] = round(sum(agreements) / len(agreements), 2)

    f1s = [e["ground_truth_f1"] for e in report["per_image"] if e.get("ground_truth_f1") is not None]
    if f1s:
        report["aggregate"]["avg_ground_truth_f1"] = round(sum(f1s) / len(f1s), 4)

    out = output_dir / "benchmark_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# Benchmark Report — Panacea",
        f"- Ảnh test phân tích: {report['test_images_analyzed']}",
        f"- Tổng tế bào: {report['aggregate']['total_cells']}",
    ]
    if "avg_hybrid_agreement_pct" in report["aggregate"]:
        md_lines.append(f"- YOLO vs Hybrid agreement TB: {report['aggregate']['avg_hybrid_agreement_pct']}%")
    if "avg_ground_truth_f1" in report["aggregate"]:
        md_lines.append(f"- Ground Truth F1 TB: {report['aggregate']['avg_ground_truth_f1']}")
    md_lines.append("\n## Chi tiết từng ảnh\n")
    for e in report["per_image"]:
        md_lines.append(f"- **{e.get('image')}**: {e.get('total_cells', 'ERR')} cells, F1={e.get('ground_truth_f1', 'N/A')}")

    (output_dir / "benchmark_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Benchmark saved to {output_dir}")


if __name__ == "__main__":
    main()
