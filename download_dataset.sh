#!/bin/bash

# Telco Customer Churn Dataset Download Script
# This script downloads the dataset using Kaggle API

echo "=================================================="
echo "Telco Customer Churn - Dataset Download"
echo "=================================================="

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null
then
    echo "❌ Kaggle CLI not found. Installing..."
    pip install kaggle
fi

# Check if kaggle credentials exist
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "⚠️  Kaggle credentials not found!"
    echo ""
    echo "Please follow these steps:"
    echo "1. Go to https://www.kaggle.com/settings"
    echo "2. Scroll to 'API' section"
    echo "3. Click 'Create New API Token'"
    echo "4. Move the downloaded kaggle.json to ~/.kaggle/"
    echo "5. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p data

# Download dataset
echo ""
echo "📥 Downloading Telco Customer Churn dataset..."
kaggle datasets download -d blastchar/telco-customer-churn -p data/

# Unzip dataset
echo ""
echo "📦 Extracting dataset..."
unzip -o data/telco-customer-churn.zip -d data/

# Clean up zip file
rm data/telco-customer-churn.zip

# Verify download
if [ -f "data/WA_Fn-UseC_-Telco-Customer-Churn.csv" ]; then
    echo ""
    echo "✅ Dataset downloaded successfully!"
    echo "📁 Location: data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    echo ""
    echo "You can now run the EDA notebook:"
    echo "  jupyter notebook notebooks/01_telco_churn_eda.ipynb"
else
    echo ""
    echo "❌ Download failed. Please download manually from:"
    echo "   https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
fi

echo "=================================================="
