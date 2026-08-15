"""
Data Collection Script for ASL Sign Language Gesture Recognition.

This script uses a webcam to capture hand pose data using MediaPipe, extracts
feature vectors, and saves them along with user-specified labels to a CSV file.

Keyboard Controls:
- UP/DOWN arrow or 'n'/'p': Change current label
- SPACE: Toggle automatic capture
- 's': Single frame capture
- 'q' or ESC: Quit and save data
"""

import cv2
import time
import csv
import pandas as pd
from pathlib import Path

from config import (
    ALL_LABELS, DATA_FILE, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    CAPTURE_FPS, MP_MAX_HANDS, MP_DETECTION_CONFIDENCE,
    MP_TRACKING_CONFIDENCE, TOTAL_FEATURES
)
from utils import HandDetector, FeatureExtractor, Visualizer

def load_existing_counts(data_file):
    """Load existing sample counts from the CSV file."""
    counts = {label: 0 for label in ALL_LABELS}
    if data_file.exists():
        try:
            df = pd.read_csv(data_file)
            if 'label' in df.columns:
                counts_from_csv = df['label'].value_counts().to_dict()
                for label, count in counts_from_csv.items():
                    if label in counts:
                        counts[label] = count
        except Exception as e:
            print(f"Error loading existing data: {e}")
    return counts

def main():
    print("Initializing Hand Pose Data Collection...")
    print("Loading existing data counts...")
    
    counts = load_existing_counts(DATA_FILE)
    
    # Initialize components
    detector = HandDetector(
        max_num_hands=MP_MAX_HANDS,
        min_detection_confidence=MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=MP_TRACKING_CONFIDENCE
    )
    extractor = FeatureExtractor()
    
    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("\n--- Controls ---")
    print("UP/DOWN or N/P : Navigate labels")
    print("SPACE          : Toggle auto-capture")
    print("S              : Capture single sample")
    print("Q / ESC        : Quit")
    
    current_label_idx = 0
    is_recording = False
    
    last_capture_time = 0
    capture_interval = 1.0 / CAPTURE_FPS if CAPTURE_FPS > 0 else 0
    
    prev_frame_time = 0
    
    # Open CSV file for appending
    file_exists = DATA_FILE.exists()
    
    # Create parent directories if they don't exist
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(DATA_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if new file
            if not file_exists:
                header = [f"f_{i}" for i in range(TOTAL_FEATURES)] + ['label']
                writer.writerow(header)
                
            while True:
                success, frame = cap.read()
                if not success:
                    print("Error: Could not read frame from camera.")
                    break
                    
                # Mirror frame
                frame = cv2.flip(frame, 1)
                current_time = time.time()
                
                # Calculate FPS
                fps = 1 / (current_time - prev_frame_time) if prev_frame_time > 0 else 0
                prev_frame_time = current_time
                
                # Process frame
                results = detector.detect(frame)
                landmarks = detector.get_landmarks(results)
                
                frame = detector.draw_landmarks(frame, results)
                
                features = None
                if landmarks and len(landmarks) > 0:
                    # Extract from the first detected hand
                    features = extractor.extract(landmarks[0])
                
                current_label = ALL_LABELS[current_label_idx]
                
                # Handle capture logic
                save_frame = False
                
                # Auto capture
                if is_recording and current_time - last_capture_time >= capture_interval:
                    if features is not None:
                        save_frame = True
                        last_capture_time = current_time
                
                # Input handling
                # waitKeyEx allows catching arrow keys on Windows
                key = cv2.waitKeyEx(1)
                if key == ord('q') or key == 27:  # q or ESC
                    break
                elif key == ord('n') or key == 2621440:  # 'n' or DOWN arrow
                    current_label_idx = (current_label_idx + 1) % len(ALL_LABELS)
                elif key == ord('p') or key == 2490368:  # 'p' or UP arrow
                    current_label_idx = (current_label_idx - 1) % len(ALL_LABELS)
                elif key == ord(' '):  # Space
                    is_recording = not is_recording
                    if is_recording:
                        last_capture_time = time.time()
                elif key == ord('s'):
                    if features is not None:
                        save_frame = True
                
                # If save frame triggered via auto or single capture
                if save_frame and features is not None:
                    row = features.tolist() + [current_label]
                    writer.writerow(row)
                    f.flush()
                    counts[current_label] += 1
                
                # UI Overlays
                if features is None and is_recording:
                    Visualizer.draw_no_hand_warning(frame)
                    
                Visualizer.draw_recording_indicator(frame, is_recording, current_label, counts[current_label])
                Visualizer.draw_label_selector(frame, ALL_LABELS, current_label_idx, counts, y_start=150)
                Visualizer.draw_controls(frame, mode='collect')
                Visualizer.draw_fps(frame, fps)
                
                cv2.imshow("Data Collection", frame)
                
    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nCleaning up resources...")
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n--- Session Summary ---")
        for label in ALL_LABELS:
            print(f"{label}: {counts[label]} samples")
        print(f"\nData saved to: {DATA_FILE}")

if __name__ == '__main__':
    main()
