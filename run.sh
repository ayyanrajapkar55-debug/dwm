#!/usr/bin/env bash
set -e

# Multi-Source Precision Agriculture Intelligence Runner
echo "=========================================================="
echo "🌾 MULTI-SOURCE PRECISION AGRICULTURE INTELLIGENCE"
echo "   Data Warehousing & Data Mining Mini-Project"
echo "=========================================================="

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "Installing requirements..."
    .venv/bin/pip install -r requirements.txt
fi

# Run ETL and train models if warehouse or models don't exist
if [ ! -f "warehouse.db" ]; then
    echo "Generating datasets..."
    .venv/bin/python3 generate_data.py
    echo "Running ETL pipeline to build star schema warehouse..."
    .venv/bin/python3 etl.py
fi

if [ ! -f "trained_models.joblib" ]; then
    echo "Training data mining and ML models..."
    .venv/bin/python3 ml_models.py
fi

echo "Launching Streamlit Web Application..."
.venv/bin/streamlit run app.py
