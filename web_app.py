#!/usr/bin/env python3
"""
Blood Cell Detection & Classification Web Application
Flask-based web interface for the Hybrid AI Pipeline
"""

import os
import json
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
import cv2
import numpy as np

from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS

from src.classification.ml_classifier import MLClassifier
from src.config.settings import PATHS
from src.detection.yolo_detector import YoloDetector
from src.pipeline.end_to_end import HybridCellPipeline

# ============================================================================
# FLASK APP SETUP
# ============================================================================

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
RESULTS_FOLDER = Path(__file__).parent / 'results'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
app.config['JSON_SORT_KEYS'] = False

# ============================================================================
# LOAD MODELS
# ============================================================================

try:
    yolo_model_path = PATHS.yolo_models / "best.pt"
    ml_model_path = PATHS.ml_models / "best_ml_model.pt"
    
    print("🔄 Initializing Pipeline...")
    pipeline = HybridCellPipeline(yolo_model_path, ml_model_path)
    print("✅ Pipeline ready!")
    MODELS_LOADED = True
except Exception as e:
    print(f"❌ Error loading models: {e}")
    MODELS_LOADED = False
    pipeline = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_image_file(filepath, output_dir):
    """Analyze single image and return results"""
    if not MODELS_LOADED or pipeline is None:
        return {
            "success": False,
            "error": "Models not loaded. Please restart application."
        }
    
    try:
        image_path = Path(filepath)
        output_image_path = output_dir / f"{image_path.stem}_analyzed.png"
        
        # Run pipeline
        result = pipeline.run_on_image_full(
            image_path=image_path,
            output_image_path=output_image_path,
            output_stats_dir=output_dir,
        )
        
        # Prepare response
        response = {
            "success": True,
            "image_name": image_path.name,
            "total_cells": result.stats.total_cells,
            "cell_counts": result.stats.cell_counts,
            "cell_percentages": result.stats.cell_percentages,
            "warnings": result.stats.warnings,
            "report": result.report_text,
            "output_image": str(output_image_path.relative_to(Path(__file__).parent)),
        }
        
        # Add download links
        if (output_dir / f"{image_path.stem}_report.json").exists():
            response["report_json"] = str((output_dir / f"{image_path.stem}_report.json").relative_to(Path(__file__).parent))
        if (output_dir / f"{image_path.stem}_summary.csv").exists():
            response["report_csv"] = str((output_dir / f"{image_path.stem}_summary.csv").relative_to(Path(__file__).parent))
            
        return response
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', models_loaded=MODELS_LOADED)


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "models_loaded": MODELS_LOADED,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/info', methods=['GET'])
def system_info():
    """Get system information"""
    return jsonify({
        "status": "ready" if MODELS_LOADED else "error",
        "system": "Blood Cell Detection & Classification",
        "version": "1.0.0",
        "models": {
            "yolo": "best.pt",
            "ml": "best_ml_model.pt"
        },
        "classes": {
            0: "RBC (Hồng cầu)",
            1: "WBC (Bạch cầu)",
            2: "Platelets (Tiểu cầu)"
        }
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis"""
    
    if not MODELS_LOADED:
        return jsonify({
            "success": False,
            "error": "System not ready - models not loaded"
        }), 503
    
    # Check file
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "error": "No file provided"
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "No file selected"
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        # Create analysis output directory
        output_dir = RESULTS_FOLDER / filename.split('.')[0]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze
        result = analyze_image_file(filepath, output_dir)
        
        if result['success']:
            # Generate thumbnail
            img = cv2.imread(str(filepath))
            if img is not None:
                height = img.shape[0]
                scale = 300 / height if height > 300 else 1
                thumb = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
                thumb_path = output_dir / 'thumbnail.jpg'
                cv2.imwrite(str(thumb_path), thumb)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get list of analysis results"""
    try:
        results = []
        for result_dir in sorted(RESULTS_FOLDER.iterdir(), reverse=True):
            if result_dir.is_dir():
                json_file = result_dir / f"{result_dir.name}_report.json"
                if json_file.exists():
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    results.append({
                        "name": result_dir.name,
                        "timestamp": result_dir.name.split('_')[0:3],
                        "data": data
                    })
        
        return jsonify({
            "success": True,
            "count": len(results),
            "results": results[:10]  # Return last 10
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """Download result file"""
    try:
        file_path = RESULTS_FOLDER / filename
        if file_path.exists() and file_path.is_file():
            return send_file(str(file_path), as_attachment=True)
        else:
            return jsonify({
                "success": False,
                "error": "File not found"
            }), 404
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/results/<path:filename>', methods=['GET'])
def serve_result(filename):
    """Serve result image"""
    try:
        file_path = RESULTS_FOLDER / filename
        if file_path.exists() and file_path.is_file():
            return send_file(str(file_path))
        else:
            return "File not found", 404
    except Exception as e:
        return str(e), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Server error"}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("="*60)
    print("🩺 Blood Cell Analysis Web App")
    print("="*60)
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print(f"Results folder: {RESULTS_FOLDER}")
    print(f"Models loaded: {MODELS_LOADED}")
    print("\n🚀 Starting Flask server...")
    print("   URL: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
