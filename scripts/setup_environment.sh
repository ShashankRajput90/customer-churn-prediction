#!/bin/bash

# Setup script for Customer Churn Prediction project
# Author: Shashank Lodhi
# Date: November 2025

echo "========================================"
echo "Customer Churn Prediction - Setup"
echo "========================================"
echo ""

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if [[ $(python3 -c "import sys; print(sys.version_info >= (3, 8))") == "False" ]]; then
    echo "❌ Error: Python 3.8+ required"
    exit 1
fi

echo "✅ Python version OK"
echo ""

# Create virtual environment
echo "🌐 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing..."
    rm -rf venv
fi

python3 -m venv venv
echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "🔼 Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✅ pip upgraded"
echo ""

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create directory structure
echo "📁 Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p outputs/plots
mkdir -p outputs/reports
echo "✅ Directories created"
echo ""

# Download dataset (if Kaggle API is configured)
if command -v kaggle &> /dev/null; then
    echo "📊 Downloading dataset from Kaggle..."
    kaggle datasets download -d blastchar/telco-customer-churn -p data/raw/
    
    if [ -f "data/raw/telco-customer-churn.zip" ]; then
        unzip -q data/raw/telco-customer-churn.zip -d data/raw/
        rm data/raw/telco-customer-churn.zip
        echo "✅ Dataset downloaded and extracted"
    else
        echo "⚠️  Dataset download failed. Please download manually."
    fi
else
    echo "⚠️  Kaggle CLI not found. Please download dataset manually:"
    echo "    https://www.kaggle.com/blastchar/telco-customer-churn"
    echo "    Place CSV in data/raw/ folder"
fi

echo ""
echo "========================================"
echo "✅ Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Download dataset (if not done): See data/README.md"
echo "  3. Run ML pipeline: python src/churn_prediction_pipeline.py"
echo "  4. Launch dashboard: streamlit run src/streamlit_app.py"
echo ""
