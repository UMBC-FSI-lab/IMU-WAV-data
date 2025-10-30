# Sensor Data Analyzer

A standalone application for analyzing IMU and WAV sensor data to extract precise timing information for skin contact events.

## Features

- **IMU Data Analysis**: Detects movement patterns including:
  - When the phone starts moving towards the skin
  - When the phone is starting to touch the skin
  - When the phone has fully touched the skin
  - When the phone is stable at a constant force
  - When the phone is starting to move away from the skin
  - When the phone stops after moving away from the skin

- **Audio Chirp Detection**: Analyzes WAV files to detect chirps with:
  - Frequency range: 16-24 kHz
  - Bandwidth: 8 kHz
  - Chirp duration: 0.01 seconds
  - Chirp interval: 0.05 seconds

- **Visualization**: Creates sine wave graphs with highlighted chirp numbers and timing events

## Installation

### Option 1: Quick Install (Recommended)
```bash
./install_and_run.sh
```

### Option 2: Manual Install
```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Run the application
python3 sensor_analyzer.py
```

## Usage

1. **Launch the application**:
   ```bash
   python3 sensor_analyzer.py
   ```

2. **Load your data**:
   - Click "Upload IMU CSV" to load your IMU data file
   - Click "Upload WAV File" to load your audio data file

3. **Analyze the data**:
   - Click "Analyze Data" to process both files
   - View results in the text area
   - Examine the visualization with highlighted events

## Data Format Requirements

### IMU CSV File
The CSV file should contain columns for:
- `timestamp` (or similar time column)
- `accel_x`, `accel_y`, `accel_z` (acceleration data)
- `gyro_x`, `gyro_y`, `gyro_z` (gyroscope data)

Alternative column names are supported (e.g., `ax`, `ay`, `az`).

### WAV File
- Standard WAV format
- Any sample rate (automatically detected)
- Mono or stereo (automatically converted to mono)

## Output

The application provides:

1. **Text Results**: Detailed timing information for all detected events
2. **Visualization**: 
   - Top plot: IMU acceleration magnitude with timing event markers
   - Bottom plot: Audio signal with highlighted chirp numbers

## Technical Details

- **Movement Detection**: Uses acceleration magnitude and statistical thresholds
- **Chirp Detection**: Bandpass filtering (16-24 kHz) + energy-based detection
- **Visualization**: Real-time plotting with matplotlib
- **GUI**: Simple tkinter interface for easy file upload and results viewing

## Troubleshooting

- **Import Errors**: Make sure all dependencies are installed: `pip3 install -r requirements.txt`
- **File Format Issues**: Ensure CSV has proper headers and WAV is in standard format
- **No Data Detected**: Check that your data contains the expected column names and sufficient signal variation

## Example Output

```
=== SENSOR DATA ANALYSIS RESULTS ===

TIMING EVENTS:
Moving towards skin starts: 1.234s
Touching skin: 2.456s
Stable contact: 3.789s
Moving away: 4.123s
Stopped after moving away: 5.678s

CHIRP DETECTION:
Total chirps detected: 15
Chirp times (seconds):
  Chirp 1: 0.050s
  Chirp 2: 0.100s
  Chirp 3: 0.150s
  ...
```
