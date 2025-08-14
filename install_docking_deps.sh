#!/bin/bash

# PocketHunter Suite - Docking Dependencies Installation Script
# This script installs additional dependencies required for molecular docking

echo "🧬 PocketHunter Suite - Installing Docking Dependencies"
echo "========================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not in PATH. Please install conda first."
    exit 1
fi

# Check if the dockspot environment exists
if ! conda env list | grep -q "dockspot"; then
    echo "❌ Conda environment 'dockspot' not found. Please create it first."
    echo "   Run: conda create -n dockspot python=3.9"
    exit 1
fi

echo "✅ Found conda environment: dockspot"
echo "🔧 Installing additional dependencies..."

# Activate the environment and install dependencies
conda activate dockspot

# Install ProDy
echo "📦 Installing ProDy..."
conda install -c conda-forge prody -y

# Install OpenBabel
echo "📦 Installing OpenBabel..."
conda install -c conda-forge openbabel -y

# Install additional Python packages
echo "📦 Installing additional Python packages..."
pip install openbabel>=3.1.1

echo ""
echo "✅ Docking dependencies installation completed!"
echo ""
echo "📋 Next steps:"
echo "1. Install SMINA docking software:"
echo "   - Download from: https://sourceforge.net/projects/smina/"
echo "   - Or install via conda: conda install -c conda-forge smina"
echo ""
echo "2. Test the installation:"
echo "   - Run: python -c 'import prody; import openbabel; print(\"✅ All dependencies installed successfully!\")'"
echo ""
echo "3. Start the Streamlit app:"
echo "   - Run: streamlit run main.py"
echo ""
echo "🎯 The molecular docking functionality is now available in Step 4 of the PocketHunter Suite!" 