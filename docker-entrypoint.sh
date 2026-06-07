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
    echo "  Single image analysis:"
    echo "    docker run -v \$(pwd):/app/data app:latest --image data/sample.png --output data/results/"
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

# If first argument is 'bash' or 'sh', start interactive shell
if [ "$1" = "bash" ] || [ "$1" = "sh" ] || [ "$1" = "python" ]; then
    exec "$@"
fi

# Otherwise, run app.py with all arguments
exec python /app/app.py "$@"
