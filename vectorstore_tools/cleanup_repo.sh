#!/bin/bash
# Script to clean up temporary and backup files in the repository

# Make sure we're in the project root directory
cd "$(dirname "$0")/.."
ROOT_DIR=$(pwd)

echo "Cleaning up repository at $ROOT_DIR..."

# Remove backup files
echo "Removing backup files..."
find . -name "*.bak" -type f -delete
find . -name "*~" -type f -delete
find . -name "*.backup" -type f -delete
find . -name "*.swp" -type f -delete

# Remove macOS system files
echo "Removing macOS system files..."
find . -name ".DS_Store" -type f -delete
find . -name "._*" -type f -delete

# Remove Python cache files
echo "Removing Python cache files..."
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -type f -delete
find . -name "*.pyo" -type f -delete
find . -name "*.pyd" -type f -delete

# Remove temp config files if requested
if [ "$1" == "--clean-configs" ]; then
    echo "Removing temporary configuration files..."
    find ./config -name "*_updated.yaml" -type f -delete
fi

echo "Cleanup complete!" 