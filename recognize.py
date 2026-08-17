"""
Real-time gesture recognition inference script.

This script captures webcam feed, extracts hand landmarks using MediaPipe,
and predicts American Sign Language (ASL) gestures using a trained machine learning model.
It features a prediction smoothing mechanism for stability, visual feedback,
and a simple sentence builder.

Controls:
- SPACE: Add current recognized gesture to the sentence
- BACKSPACE: Delete last character/word
- 'c': Clear the entire sentence
- 'q' / ESC: Quit the application
"""

import cv2
import numpy as np
import time
import joblib
import os
import sys

# Try to import tensorflow for keras model support
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import config
import utils

def main():
    print("Loading models and preprocessing objects...")
    
    model = None
    is_keras = False
    
    # 1. Try loading sklearn best model
    if os.path.exists(config.BEST_MODEL_PATH):
        try:
            model = joblib.load(config.BEST_MODEL_PATH)
            print(f"Loaded scikit-learn model from {config.BEST_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading scikit-learn model: {e}")
            
    # 2. If not found or failed, try loading keras model
    if model is None and TF_AVAILABLE and os.path.exists(config.MLP_MODEL_PATH):
        try:
            model = load_model(config.MLP_MODEL_PATH)
            is_keras = True
            print(f"Loaded Keras model from {config.MLP_MODEL_PATH}")
        except Exception as e:
            print(f"Error loading Keras model: {e}")
            
    # 3. If no model loaded, exit
    if model is None:
        print("Error: Could not load any trained model.")
        print("Please run train_model.py first to generate the models.")
        return
        
    # Load scaler and label encoder
    try:
        scaler = joblib.load(config.SCALER_PATH)
        label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
        print("Successfully loaded scaler and label encoder.")
    except FileNotFoundError:
        print("Error: Scaler or Label Encoder not found.")
        print("Please run train_model.py first to generate these files.")
        return

    print("\nInitializing webcam...")
    cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize components
    detector = utils.HandDetector(
        max_num_hands=config.MP_MAX_HANDS,
        min_detection_confidence=config.MP_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MP_TRACKING_CONFIDENCE
    )
    extractor = utils.FeatureExtractor()
    smoother = utils.PredictionSmoother(config.SMOOTHING_WINDOW)

    sentence = ""
    prev_time = 0
    
    print("\n--- Started Recognition ---")
    print("Press SPACE to add word, BACKSPACE to delete, 'c' to clear, 'q' or ESC to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            # Hand Detection
            results = detector.detect(frame)
            frame = detector.draw_landmarks(frame, results)
            
            landmarks_list = detector.get_landmarks(results)
            
            if landmarks_list and len(landmarks_list) > 0:
                # Use the first hand detected
                landmarks_21x3 = landmarks_list[0]
                features = extractor.extract(landmarks_21x3)
                
                if features is not None:
                    # Prepare for prediction
                    features_scaled = scaler.transform([features])
                    
                    # Predict
                    if is_keras:
                        probas = model.predict(features_scaled, verbose=0)[0]
                        max_idx = np.argmax(probas)
                        confidence = float(probas[max_idx])
                        pred_class = max_idx
                    else:
                        if hasattr(model, "predict_proba"):
                            probas = model.predict_proba(features_scaled)[0]
                            max_idx = np.argmax(probas)
                            confidence = float(probas[max_idx])
                            pred_class = max_idx
                        elif hasattr(model, "decision_function"):
                            decision = model.decision_function(features_scaled)[0]
                            # Simple softmax approximation for confidence
                            exp_d = np.exp(decision - np.max(decision))
                            probas = exp_d / exp_d.sum()
                            max_idx = np.argmax(probas)
                            confidence = float(probas[max_idx])
                            pred_class = max_idx
                        else:
                            pred_class = model.predict(features_scaled)[0]
                            confidence = 1.0
                            
                    # Decode label
                    try:
                        label = label_encoder.inverse_transform([pred_class])[0]
                    except:
                        # Fallback if prediction is string already (some models)
                        label = str(pred_class)
                        
                    # Smooth prediction
                    smoother.add(label, confidence)
                    smoothed_label, smoothed_conf = smoother.get_smoothed()
                    stability = smoother.get_stability()
                    
                    # Display prediction
                    if smoother.is_ready():
                        if smoothed_conf >= config.CONFIDENCE_THRESHOLD:
                            utils.Visualizer.draw_prediction(
                                frame, smoothed_label, smoothed_conf, 10, 150
                            )
                            # Draw stability bar
                            utils.Visualizer.draw_confidence_bar(
                                frame, stability, 10, 220, 200, 15
                            )
                        else:
                            # Low confidence
                            cv2.putText(frame, f"? ({smoothed_label} - low conf)", (10, 150), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_WARNING, 2)
                    else:
                        # Warming up
                        cv2.putText(frame, "Warming up...", (10, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_WARNING, 2)
            else:
                # No hand detected
                smoother.reset()
                utils.Visualizer.draw_no_hand_warning(frame)

            # Draw UI overlays
            utils.Visualizer.draw_sentence(frame, sentence, 10, 40)
            utils.Visualizer.draw_fps(frame, int(fps))
            utils.Visualizer.draw_controls(frame, mode='recognize')
            
            # Show the frame
            cv2.imshow("ASL Gesture Recognition", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE
                if smoother.is_ready():
                    s_label, s_conf = smoother.get_smoothed()
                    if s_conf >= config.CONFIDENCE_THRESHOLD:
                        if len(s_label) == 1: # Single letter
                            sentence += s_label
                        else: # Phrase/Word
                            if sentence and sentence[-1] != ' ':
                                sentence += " " + s_label + " "
                            else:
                                sentence += s_label + " "
            elif key == 8:  # BACKSPACE
                if len(sentence) > 0:
                    # Basic word deletion or char deletion
                    if sentence[-1] == ' ' and len(sentence) > 1:
                        sentence = sentence[:-2]
                        # Try to remove full word
                        while len(sentence) > 0 and sentence[-1] != ' ':
                            sentence = sentence[:-1]
                    else:
                        sentence = sentence[:-1]
            elif key == ord('c'):  # Clear
                sentence = ""
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
