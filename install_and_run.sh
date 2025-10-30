#!/bin/bash

echo "Installing Sensor Data Analyzer..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Install required packages
echo "Installing required packages..."
pip3 install -r requirements.txt

# Make the script executable
chmod +x sensor_analyzer.py

echo "Installation complete!"
echo "To run the application, execute: python3 sensor_analyzer.py"
echo ""
echo "Usage:"
echo "1. Click 'Upload IMU CSV' to load your IMU data file"
echo "2. Click 'Upload WAV File' to load your audio data file"
echo "3. Click 'Analyze Data' to process the data and extract timing information"
echo "4. View the results in the text area and visualization"
