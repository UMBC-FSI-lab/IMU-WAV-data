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
echo ""
echo "Usage:"
echo "python3 sensor_analyzer.py --imu <imu_file.csv> --wav <audio_file.wav> [--start <offset>]"
echo ""
echo "Example:"
echo "python3 sensor_analyzer.py --imu test8_imu.csv --wav test8.wav --start 30"
