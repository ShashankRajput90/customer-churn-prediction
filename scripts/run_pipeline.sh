#!/bin/bash

# Script to run the complete ML pipeline
# Author: Shashank Lodhi
# Date: November 2025

echo "========================================"
echo "Customer Churn Prediction - ML Pipeline"
echo "========================================"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated"
    echo "Activating venv..."
    source venv/bin/activate
fi

# Check if dataset exists
if [ ! -f "data/raw/Telco-Customer-Churn.csv" ]; then
    echo "❌ Error: Dataset not found!"
    echo "Please download dataset and place in data/raw/"
    echo "See data/README.md for instructions"
    exit 1
fi

echo "📂 Dataset found"
echo ""

# Run ML pipeline
echo "🤖 Running ML pipeline..."
echo ""
python src/churn_prediction_pipeline.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Pipeline completed successfully!"
    echo "========================================"
    echo ""
    echo "Generated files:"
    echo "  - Plots in outputs/plots/"
    echo "  - Reports in outputs/reports/"
    echo "  - Models in models/"
    echo ""
    echo "To launch dashboard, run:"
    echo "  streamlit run src/streamlit_app.py"
    echo ""
else
    echo ""
    echo "❌ Pipeline failed!"
    echo "Check error messages above"
    exit 1
fi
