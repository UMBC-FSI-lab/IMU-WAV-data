#!/usr/bin/env python3


import pandas as pd
import numpy as np
import scipy.signal
from scipy.io import wavfile
import sys
import argparse

def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"

def analyze_imu(imu_file):
    """Analyze IMU data - detect complete movement cycles"""
    df = pd.read_csv(imu_file)
    if 'type' in df.columns:
        df = df[df['type'] == 'ACC'].copy()
    
    x, y, z = df['x'].values, df['y'].values, df['z'].values
    mag = np.sqrt(x**2 + y**2 + z**2)
    
    if 'timestamp_ms' in df.columns:
        ts = df['timestamp_ms'].values / 1000.0
    else:
        ts = np.arange(len(mag)) / 100.0
    
    ts = ts - ts[0]
    
    # Use fixed thresholds based on median - more stable for flat data
    median = np.median(mag)
    std = np.std(mag)
    high = median + 0.5  # Fixed offset for "high" activity
    low = median - 0.5   # Fixed offset for "low" activity
    
    # Detect transitions
    moving_to_high = []
    moving_to_low = []
    prev_high, prev_low = False, False
    
    for i in range(1, len(mag)):
        if mag[i] > high and not prev_high:
            moving_to_high.append(i)
        elif mag[i] < low and not prev_low:
            moving_to_low.append(i)
        prev_high = mag[i] > high
        prev_low = mag[i] < low
    
    # Cluster events (min 15 seconds apart for significant movements only)
    min_samples = int((ts[1] - ts[0])**(-1) * 15) if len(ts) > 1 and ts[1] != ts[0] else 1500
    
    def cluster(indices):
        if len(indices) == 0:
            return []
        sorted_idx = sorted(indices)
        clustered = [sorted_idx[0]]
        for idx in sorted_idx[1:]:
            if idx - clustered[-1] >= min_samples:
                clustered.append(idx)
        return clustered
    
    moving_to_high = cluster(moving_to_high)
    moving_to_low = cluster(moving_to_low)
    
    # Get stable regions between movements
    stable_regions = []
    for low_idx in moving_to_low:
        next_high = [h for h in moving_to_high if h > low_idx]
        if next_high:
            stable_regions.append((low_idx, next_high[0]))
    
    # Detect "stopped after moving away"
    stopped_away = []
    for high_idx in moving_to_high:
        for i in range(high_idx, min(high_idx + min_samples, len(mag))):
            if abs(mag[i] - median) < 0.15:
                stopped_away.append(i)
                break
    stopped_away = cluster(stopped_away)
    
    # Organize into cycles: for each low->high->low pattern
    # This represents a complete interaction cycle
    moving_towards_list = []
    starting_touch_list = []
    fully_touched_list = []
    stable_contact_list = []
    moving_away_list = []
    just_touched_list = []
    stopped_away_list = []
    
    # Pair up low and high transitions to form cycles
    cycles = []
    for i, low_idx in enumerate(moving_to_low):
        # Find next high after this low
        highs_after = [h for h in moving_to_high if h > low_idx]
        if highs_after:
            # Find next low after that high (start of new cycle)
            next_low = [l for l in moving_to_low if l > highs_after[0]]
            cycles.append((low_idx, highs_after[0], next_low[0] if next_low else None))
    
    # Track used indices to avoid duplicates
    used_high = set()
    used_low = set()
    used_stopped = set()
    
    # For each cycle, extract all phases
    for touch_start, move_away, next_touch in cycles:
        # Phase 1: Moving towards (look for high before touch_start)
        before_touch = [h for h in moving_to_high if h < touch_start and h not in used_high]
        if before_touch:
            idx = before_touch[-1]
            moving_towards_list.append(ts[idx])
            used_high.add(idx)
        
        # Phase 2,3,6: Touch events
        starting_touch_list.append(ts[touch_start])
        fully_touched_list.append(ts[touch_start])
        just_touched_list.append(ts[touch_start])
        
        # Phase 4: Stable (from touch to moving away)
        if move_away:
            stable_contact_list.append((ts[touch_start], ts[move_away]))
        
        # Phase 5: Moving away
        if move_away and move_away not in used_high:
            moving_away_list.append(ts[move_away])
            used_high.add(move_away)
        
        # Phase 7: Stopped after moving away
        if move_away:
            stops = [s for s in stopped_away if s > move_away and s not in used_stopped]
            if stops:
                idx = stops[0]
                stopped_away_list.append(ts[idx])
                used_stopped.add(idx)
    
    return {
        'moving_towards': moving_towards_list,
        'starting_touch': starting_touch_list,
        'fully_touched': fully_touched_list,
        'stable_contact': stable_contact_list,
        'moving_away': moving_away_list,
        'just_touched': just_touched_list,
        'stopped_away': stopped_away_list
    }

def analyze_audio(wav_file):
    """Analyze audio and detect chirps"""
    sr, audio = wavfile.read(wav_file)
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1).astype(np.float64)
    else:
        audio = audio.astype(np.float64)
    
    audio = audio / np.max(np.abs(audio))
    
    nyquist = sr / 2
    b, a = scipy.signal.butter(6, [16000/nyquist, min(24000/nyquist, 0.99)], btype='band')
    filtered = scipy.signal.filtfilt(b, a, audio)
    
    dur_samples = int(0.01 * sr)
    interval_samples = int(0.05 * sr)
    
    energy = []
    times = []
    for i in range(0, len(filtered) - dur_samples, interval_samples):
        energy.append(np.sum(filtered[i:i+dur_samples]**2))
        times.append(i / sr)
    
    energy = np.array(energy)
    times = np.array(times)
    
    threshold = np.percentile(energy, 90)
    chirp_indices = np.where(energy > threshold)[0]
    chirp_times = times[chirp_indices]
    
    if len(chirp_times) > 1:
        valid = [chirp_times[0]]
        for i in range(1, len(chirp_times)):
            if chirp_times[i] - valid[-1] >= 0.03:
                valid.append(chirp_times[i])
        chirp_times = np.array(valid)
    
    return chirp_times

def generate_timeline(events, chirps, offset=0.0):
    """Generate timeline with all 7 phases"""
    all_items = []
    
    for t in events['moving_towards']:
        all_items.append((t, "Phone starts moving towards arm", True))
    
    for t in events['starting_touch']:
        all_items.append((t, "Phone is starting to touch the skin", True))
    
    for t in events['fully_touched']:
        all_items.append((t, "Phone has fully touched the skin", True))
    
    for t_start, t_end in events['stable_contact']:
        all_items.append((t_start, "Its stable at constant force now", False))
    
    for t in events['moving_away']:
        all_items.append((t, "Phone starts moving away from arm", True))
    
    for t in events['just_touched']:
        all_items.append((t, "Phone has just been touched on the skin", True))
    
    for t in events['stopped_away']:
        all_items.append((t, "Phone stops after moving away from arm", False))
    
    all_items.sort(key=lambda x: x[0])
    
    output = f"Start point after calibration can be at {int(round(offset))} sec\n\n"
    
    for t, label, show_chirp in all_items:
        t_cal = t
        chirp_part = ""
        
        if show_chirp and len(chirps) > 0:
            idx = np.argmin(np.abs(chirps - t))
            chirp_part = f" ({ordinal(idx+1)} chirp)"
        
        output += f"• Second {int(round(t_cal))}{chirp_part} → {label}\n"
    
    return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imu', required=True)
    parser.add_argument('--wav', required=True)
    parser.add_argument('--start', type=float, default=0.0)
    args = parser.parse_args()
    
    print("Analyzing IMU data...")
    events = analyze_imu(args.imu)
    print(f"Found {len(events['moving_towards'])} moving towards")
    print(f"Found {len(events['starting_touch'])} starting touch")
    print(f"Found {len(events['fully_touched'])} fully touched")
    print(f"Found {len(events['stable_contact'])} stable regions")
    print(f"Found {len(events['moving_away'])} moving away")
    print(f"Found {len(events['just_touched'])} just touched")
    print(f"Found {len(events['stopped_away'])} stopped away")
    
    print("Analyzing audio data...")
    chirps = analyze_audio(args.wav)
    print(f"Found {len(chirps)} chirps")
    
    timeline = generate_timeline(events, chirps, args.start)
    
    print("\n" + "="*60)
    print("TIMELINE OUTPUT:")
    print("="*60)
    print(timeline)
    
    with open("timeline_output.txt", 'w') as f:
        f.write(timeline)
    print("\n✓ Saved to timeline_output.txt")

if __name__ == "__main__":
    main()
