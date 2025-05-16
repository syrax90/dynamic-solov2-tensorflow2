#!/bin/bash

echo "Setting up Python environment..."

# Optional: check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "Packages installed successfully from requirements.txt."
else
    echo "Error: requirements.txt not found in the current directory."
    exit 1
fi
