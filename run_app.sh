#!/bin/bash
# Launch CS2-Predictor GUI on macOS/Linux
cd "$(dirname "$0")"
pip install -r requirements.txt
streamlit run app.py
