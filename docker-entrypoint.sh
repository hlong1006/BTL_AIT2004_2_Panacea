#!/bin/bash
# Docker entrypoint script for flexible command execution

# Set Python path
export PYTHONPATH=/app:$PYTHONPATH

# If first argument is empty or --help, show help
if [ -z "$1" ] || [ "$1" = "--help" ]; then
    echo "=================================================="
    echo "Blood Cell Analysis System - Docker Container"
    echo "=================================================="
    echo ""
    echo "Available commands:"
    echo ""
    echo "  Web interface (Flask):"
    echo "    docker run -p 5000:5000 -v \$(pwd)/models:/app/models blood-cell-analyzer:latest python web_app.py"
    echo ""
    echo "  Single image analysis:"
    echo "    docker run -v \$(pwd)/models:/app/models -v \$(pwd)/outputs:/app/outputs blood-cell-analyzer:latest --image data/train/images/sample.jpg --output outputs/analysis_results/"
    echo ""
    echo "  Batch processing:"
    echo "    docker run -v \$(pwd):/app/data app:latest --folder data/samples --output data/results/ --recursive"
    echo ""
    echo "  Train ML model:"
    echo "    docker run -v \$(pwd):/app app:latest python scripts/run_ml_pipeline.py"
    echo ""
    echo "  Interactive mode:"
    echo "    docker run -it -v \$(pwd):/app/data app:latest bash"
    echo ""
    echo "  Using docker-compose:"
    echo "    docker-compose up"
    echo ""
    echo "=================================================="
    exit 0
fi

# If first argument starts the web server or another Python script, run it directly
if [ "$1" = "bash" ] || [ "$1" = "sh" ] || [ "$1" = "python" ] || [ "$1" = "python3" ]; then
    exec "$@"
fi

# Otherwise, run app.py with all arguments
exec python /app/app.py "$@"
