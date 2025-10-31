# Sensor Data Analyzer

A standalone application for analyzing IMU and WAV sensor data to extract precise timing information for phone movement and skin contact events.

## Features

- **IMU Data Analysis**: Detects movement patterns including:
   - When the phone starts moving towards the skin

   - When the phone is starting to touch the skin
    
   - When the phone has fully touched the skin
    
   - When the phone is stable at a constant force
    
    - When the phone is starting to move away from the skin
    
   - When the phone has just been touched on the skin
    
  - When the phone stops after moving away from the skin

- **Audio Chirp Detection**: Analyzes WAV files to detect chirps with:
  - Frequency range: 16-24 kHz
  - Bandwidth: 8 kHz
  - Chirp duration: 0.01 seconds
  - Chirp interval: 0.05 seconds

- **Timeline Output**: Generates a clear timeline showing all detected events with corresponding chirp numbers

## Installation

```bash
# Install Python dependencies
pip3 install -r requirements.txt
```

## Usage

Run from command line with your IMU and WAV files:

```bash
python3 sensor_analyzer.py --imu <path_to_imu.csv> --wav <path_to_audio.wav> --start <offset_in_seconds>
```

**Example:**
```bash
python3 sensor_analyzer.py --imu test8_imu.csv --wav test8.wav --start 30
```

### Arguments

- `--imu`: Path to IMU CSV file (required)
- `--wav`: Path to WAV audio file (required)  
- `--start`: Start offset in seconds for calibration (default: 0)

## Data Format Requirements

### IMU CSV File
The CSV file should contain columns:
- `timestamp_ms`: Timestamp in milliseconds
- `type`: Sensor type (e.g., 'ACC', 'GYR', 'MAG') - only 'ACC' data is used
- `x`, `y`, `z`: Acceleration values in m/s²

Example format:
```csv
timestamp_ms,type,x,y,z
1761331986648,ACC,0.049081136,9.84406,-0.13227965
1761331986652,ACC,0.029927522,9.851242,-0.12749124
```

### WAV File
- Standard WAV format
- Any sample rate (automatically detected)
- Mono or stereo (automatically converted to mono)

## Output

The application generates a timeline output showing:

```
Start point after calibration can be at 30 sec

• Second 2 (6th chirp) → Phone starts moving towards arm
• Second 2 → Its stable at constant force now
• Second 2 (6th chirp) → Phone starts moving away from arm
• Second 21 → Its stable at constant force now
• Second 21 (71st chirp) → Phone starts moving away from arm
...
```

The output is also saved to `timeline_output.txt` in the current directory.

## Technical Details

### Movement Detection
- Uses acceleration magnitude: `√(x² + y² + z²)`
- Detects transitions above/below median ± 0.5 m/s²
- Clusters events that are at least 15 seconds apart
- Identifies stable regions between movements
- Detects all 7 movement phases:
  - Phone starts moving towards the skin
  - Phone is starting to touch the skin
  - Phone has fully touched the skin
  - Phone is stable at constant force
  - Phone starts moving away from the skin
  - Phone has just been touched on the skin
  - Phone stops after moving away from the skin

### Chirp Detection
- Bandpass filtering: 6th order Butterworth (16-24 kHz)
- Energy-based detection with 90th percentile threshold
- Removes chirps closer than 0.03 seconds

### Output Format
- Timestamps are normalized to start at 0
- Chirp numbers show nearest chirp to each movement event
- Bullet points format for easy reading

## Example

```bash
python3 sensor_analyzer.py --imu "/Users/data/test8_imu.csv" --wav "/Users/data/test8.wav" --start 30

Analyzing IMU data...
Found 4 moving towards, 10 moving away, 10 stable regions
Analyzing audio data...
Found 1220 chirps

============================================================
TIMELINE OUTPUT:
============================================================
Start point after calibration can be at 30 sec

• Second 2 (6th chirp) → Phone starts moving towards arm
• Second 2 → Its stable at constant force now
• Second 2 (6th chirp) → Phone starts moving away from arm
...

✓ Saved to timeline_output.txt
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scipy
- matplotlib

Install all requirements:
```bash
pip3 install -r requirements.txt
```

## Troubleshooting

- **Import Errors**: Install dependencies: `pip3 install -r requirements.txt`
- **File Not Found**: Use absolute paths or ensure files are in current directory
- **No Events Detected**: Check that IMU data has sufficient variation in acceleration values
