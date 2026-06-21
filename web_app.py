#!/usr/bin/env python3
"""Ứng dụng Web Phát hiện & Phân loại Tế bào Máu — Panacea v2."""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.classification.ml_classifier import MLClassifier
from src.config.settings import PATHS
from src.pipeline.end_to_end import HybridCellPipeline

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path(__file__).parent / "uploads"
RESULTS_FOLDER = Path(__file__).parent / "results"
DEMO_FOLDER = PATHS.demo_images
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}
MAX_FILE_SIZE = 50 * 1024 * 1024

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)
DEMO_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE

YOLO_MODEL = PATHS.yolo_models / "best.pt"
ML_MODEL = PATHS.ml_models / "best_ml_model.pt"
_pipeline: Optional[HybridCellPipeline] = None
MODELS_LOADED = False

try:
    _pipeline = HybridCellPipeline(YOLO_MODEL, ML_MODEL)
    MODELS_LOADED = True
    print("Pipeline sẵn sàng.")
except Exception as exc:
    print(f"Lỗi tải model: {exc}")
    _pipeline = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _result_to_response(result, image_path: Path, output_dir: Path) -> dict:
    stem = image_path.stem
    resp = {
        "success": True,
        "image_name": image_path.name,
        "total_cells": result.stats.total_cells,
        "cell_counts": result.stats.cell_counts,
        "cell_percentages": result.stats.cell_percentages,
        "warnings": result.stats.warnings,
        "report": result.report_text,
        "health_diagnosis": result.health_report.to_text() if result.health_report else "",
        "health_status": result.health_report.overall_status.value if result.health_report else "",
        "health_dict": result.health_report.to_dict() if result.health_report else {},
        "plain_language": result.plain_language_summary or "",
        "confidence_stats": result.confidence_stats,
        "per_cell_confidence": result.per_cell_confidence[:50],
        "anomalies": result.anomalies,
        "explanation": result.explanation,
        "global_importance": result.global_importance,
        "cbc_estimate": result.cbc_estimate.to_dict() if result.cbc_estimate else {},
        "hybrid_comparison": result.hybrid_comparison,
        "ground_truth": result.ground_truth_metrics,
        "wbc_subtypes": result.wbc_subtypes,
        "output_image": str((output_dir / f"{stem}_analyzed.png").relative_to(Path(__file__).parent)),
        "chart_data": {
            "labels": list(result.stats.cell_counts.keys()),
            "counts": list(result.stats.cell_counts.values()),
            "percentages": [round(v, 2) for v in result.stats.cell_percentages.values()],
            "normal_ranges": BloodHealthAnalyzer_ranges(),
        },
    }
    for suffix, key in [
        ("_report.json", "report_json"),
        ("_summary.csv", "report_csv"),
        ("_health_diagnosis.txt", "health_diagnosis_file"),
        ("_report.pdf", "report_pdf"),
        ("_plain_language.txt", "plain_language_file"),
    ]:
        p = output_dir / f"{stem}{suffix}"
        if p.exists():
            resp[key] = str(p.relative_to(Path(__file__).parent))
    return resp


def BloodHealthAnalyzer_ranges():
    from src.diagnosis.blood_health_analyzer import BloodHealthAnalyzer
    return {k: {"min": v["min"], "max": v["max"]} for k, v in BloodHealthAnalyzer.NORMAL_PERCENTAGE.items()}


def analyze_image_file(filepath: Path, output_dir: Path, conf: float = 0.06, ml_conf: float = 0.0):
    if not MODELS_LOADED or _pipeline is None:
        return {"success": False, "error": "Models chưa tải."}
    try:
        _pipeline.set_conf_threshold(conf)
        output_image_path = output_dir / f"{filepath.stem}_analyzed.png"
        result = _pipeline.run_on_image_full(
            image_path=filepath,
            output_image_path=output_image_path,
            output_stats_dir=output_dir,
            ml_confidence_threshold=ml_conf,
        )
        return _result_to_response(result, filepath, output_dir)
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def _load_demo_manifest():
    manifest_path = DEMO_FOLDER / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"samples": []}


def _ensure_demo_images():
    manifest_path = DEMO_FOLDER / "manifest.json"
    if manifest_path.exists():
        return
    samples = [
        ("demo_normal.jpg", "BloodImage_00038_jpg.rf.ffa23e4b5b55b523367f332af726eae8.jpg", "Mẫu bình thường", "Tỷ lệ tế bào cân bằng"),
        ("demo_high_wbc.jpg", "BloodImage_00337_jpg.rf.b6cb228440b9158cafec01a0351e3aad.jpg", "Nhiều bạch cầu", "Có thể gợi ý tăng WBC"),
        ("demo_sparse.jpg", "BloodImage_00099_jpg.rf.e3c42cd68359527494a53843479dff5c.jpg", "Mẫu thưa tế bào", "Ít tế bào trong field"),
    ]
    demo_samples = []
    for demo_name, src_name, title, desc in samples:
        src = PATHS.test_images / src_name
        dst = DEMO_FOLDER / demo_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
        if dst.exists():
            demo_samples.append({"id": demo_name.replace(".jpg", ""), "file": demo_name, "title": title, "description": desc})
    manifest_path.write_text(json.dumps({"samples": demo_samples}, ensure_ascii=False, indent=2), encoding="utf-8")


@app.route("/")
def index():
    _ensure_demo_images()
    return render_template("index.html", models_loaded=MODELS_LOADED)


@app.route("/api/health")
def health_check():
    return jsonify({"status": "ok", "models_loaded": MODELS_LOADED, "timestamp": datetime.now().isoformat()})


@app.route("/api/info")
def system_info():
    return jsonify({
        "status": "ready" if MODELS_LOADED else "error",
        "system": "Panacea Blood Cell Analyzer v2",
        "version": "2.0.0",
        "features": [
            "hybrid_pipeline", "health_diagnosis", "anomaly_detection",
            "explainability", "cbc_estimate", "ground_truth", "batch_upload",
            "demo_mode", "pdf_export", "webcam", "wbc_subtype_heuristic",
        ],
        "models": {"yolo": "best.pt", "ml": "best_ml_model.pt"},
        "classes": {"0": "Platelets", "1": "RBC", "2": "WBC"},
    })


@app.route("/api/demo/samples")
def demo_samples():
    _ensure_demo_images()
    return jsonify({"success": True, "samples": _load_demo_manifest().get("samples", [])})


@app.route("/api/demo/run/<sample_id>", methods=["POST"])
def run_demo(sample_id):
    if not MODELS_LOADED:
        return jsonify({"success": False, "error": "Models chưa tải."}), 503
    _ensure_demo_images()
    conf = float(request.json.get("conf", 0.06)) if request.is_json else 0.06
    for s in _load_demo_manifest().get("samples", []):
        if s["id"] == sample_id:
            src = DEMO_FOLDER / s["file"]
            if not src.exists():
                return jsonify({"success": False, "error": "File demo không tồn tại."}), 404
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_")
            dst = UPLOAD_FOLDER / f"{ts}demo_{s['file']}"
            shutil.copy2(src, dst)
            out = RESULTS_FOLDER / dst.stem
            out.mkdir(parents=True, exist_ok=True)
            result = analyze_image_file(dst, out, conf=conf)
            result["demo_id"] = sample_id
            result["demo_title"] = s.get("title", "")
            return jsonify(result)
    return jsonify({"success": False, "error": "Không tìm thấy mẫu demo."}), 404


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if not MODELS_LOADED:
        return jsonify({"success": False, "error": "Models chưa tải."}), 503
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Không có file."}), 400
    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "File không hợp lệ."}), 400

    conf = float(request.form.get("conf", 0.06))
    ml_conf = float(request.form.get("ml_conf", 0.0))

    filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(file.filename)
    filepath = UPLOAD_FOLDER / filename
    file.save(str(filepath))
    output_dir = RESULTS_FOLDER / filepath.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return jsonify(analyze_image_file(filepath, output_dir, conf=conf, ml_conf=ml_conf))


@app.route("/api/upload/batch", methods=["POST"])
def upload_batch():
    if not MODELS_LOADED:
        return jsonify({"success": False, "error": "Models chưa tải."}), 503
    files = request.files.getlist("files")
    if not files:
        return jsonify({"success": False, "error": "Không có file."}), 400

    conf = float(request.form.get("conf", 0.06))
    results = []
    for file in files:
        if not file.filename or not allowed_file(file.filename):
            results.append({"success": False, "image_name": file.filename, "error": "Invalid"})
            continue
        filename = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        output_dir = RESULTS_FOLDER / filepath.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        r = analyze_image_file(filepath, output_dir, conf=conf)
        results.append(r)

    ok = [r for r in results if r.get("success")]
    return jsonify({
        "success": True,
        "total": len(results),
        "successful": len(ok),
        "failed": len(results) - len(ok),
        "results": results,
        "comparison": _batch_comparison(ok),
    })


def _batch_comparison(ok_results):
    if not ok_results:
        return {}
    totals = {"RBC": 0, "WBC": 0, "Platelets": 0}
    for r in ok_results:
        for k, v in r.get("cell_counts", {}).items():
            totals[k] = totals.get(k, 0) + v
    return {"aggregate_counts": totals, "images": len(ok_results)}


@app.route("/api/analyze-frame", methods=["POST"])
def analyze_frame():
    if not MODELS_LOADED:
        return jsonify({"success": False, "error": "Models chưa tải."}), 503
    if "file" not in request.files:
        return jsonify({"success": False, "error": "Không có frame."}), 400
    file = request.files["file"]
    conf = float(request.form.get("conf", 0.06))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_")
    filepath = UPLOAD_FOLDER / f"{ts}webcam_frame.jpg"
    file.save(str(filepath))
    output_dir = RESULTS_FOLDER / filepath.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    return jsonify(analyze_image_file(filepath, output_dir, conf=conf))


@app.route("/api/results")
def get_results():
    try:
        results = []
        for result_dir in sorted(RESULTS_FOLDER.iterdir(), reverse=True):
            if not result_dir.is_dir():
                continue
            json_file = result_dir / f"{result_dir.name}_report.json"
            if json_file.exists():
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                img = result_dir / f"{result_dir.name}_analyzed.png"
                results.append({
                    "name": result_dir.name,
                    "data": data,
                    "thumbnail": str(img.relative_to(Path(__file__).parent)) if img.exists() else None,
                })
        return jsonify({"success": True, "count": len(results), "results": results[:20]})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/results/<path:filename>")
def serve_result(filename):
    fp = RESULTS_FOLDER / filename
    if fp.exists() and fp.is_file():
        return send_file(str(fp))
    return "Not found", 404


@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    fp = UPLOAD_FOLDER / filename
    if fp.exists():
        return send_file(str(fp))
    return "Not found", 404


@app.route("/demo/<path:filename>")
def serve_demo(filename):
    fp = DEMO_FOLDER / filename
    if fp.exists():
        return send_file(str(fp))
    return "Not found", 404


if __name__ == "__main__":
    _ensure_demo_images()
    print("=" * 60)
    print(" Panacea Web v2 — http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
